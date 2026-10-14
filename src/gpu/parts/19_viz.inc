// ---------------- runtime visualization taps (QWENOX_VIZ=1) ------------------
// Off by default; with the variable unset nothing below allocates or launches
// and every kernel gets an empty VizTap (one uniform branch, no stores).
//
// What is recorded, per slot, per absolute token position p (ring of kRing):
//   records 0..L-1  trunk layer l routing of the row processed at p:
//                   10 expert ids, 10 renormalized weights, the top-k
//                   probability mass and the router softmax entropy
//   record  L       top-5 of the trunk logits computed at p (= candidates for
//                   the token at p+1) with their softmax probabilities
// The router kernels write plain stores into a device-memory mirror of the
// rings; after the last layer's router one k_viz_publish launch copies the
// pass's records to the pinned host rings (one system-scope fence wait per
// pass instead of one per layer: per-layer fences cost ~1.5% decode on
// Windows). The top-5 kernel writes the host ring directly. Both are
// addressed by the same position source the engine's own kernels use
// (*d_pos for single-token steps, *d_vbase or the scalar base + row for
// batched ones), so graph replays land at the right position and no decode
// loop needs a hook. Multi-segment verify batches (rows of several sequences,
// eager only) pass a per-row (slot, position) table instead. MTP draft layers
// and the draft lm_head are not tapped. All slots share one allocation sized
// by QWENOX_PARALLEL (1..8) at the first ensure().
//
// Correctness rule (reader side): a speculative round also processes drafted
// rows that end up rejected, so a position's record is final only once the
// token at that position has been committed AND processed. The last emitted
// token of a request is not processed yet (or holds a rejected draft's row),
// hence it is never exposed. Records carry a tag (p + 1) written after a
// release fence; the reader takes a record only if the tag matches before and
// after copying (positions being rewritten are skipped).
namespace viz {

constexpr int kRing = 2048;  // positions per slot (power of 2)
constexpr int kRecW = 16;    // uint32 words per record

struct VizTap {
  uint32_t* ring = nullptr;    // all slot rings: device mirror, or pinned host (sys)
  bool sys = false;            // ring is host memory: fenced tag protocol
  const int* dpos = nullptr;   // device position of row 0, or null
  const int* rowtab = nullptr; // per row (slot << 24 | position), -1 = skip
  int base = 0;                // added to *dpos (or the position itself)
  int slot = 0;                // ring of rows without a rowtab
  int layer = 0;               // record index inside a position
  int recs = 0;                // records per position
  int skip_rows = 0;           // rows below this are not recorded: a batch
                               // longer than the ring keeps its last kRing rows
                               // (rows r and r + kRing would share a record)
};

__device__ inline uint32_t* rec_at(const VizTap& t, int row, int* pos) {
  int p, s;
  if (row < t.skip_rows) return nullptr;
  if (t.rowtab) {
    const int v = t.rowtab[row];
    if (v < 0) return nullptr;
    s = v >> 24;
    p = v & 0xffffff;
  } else {
    s = t.slot;
    p = (t.dpos ? *t.dpos : 0) + t.base + row;
  }
  *pos = p;
  if (p < 0) return nullptr;
  return t.ring + (((size_t)s * kRing + (p & (kRing - 1))) * t.recs + t.layer) * kRecW;
}

// Routing record: [0] tag p+1, [1..5] ids (u16 pairs), [6..10] weights (u16
// pairs, w * 65535), [11] top-k mass (f32), [12] entropy in nats (f32).
__device__ inline void store_route(const VizTap& t, int row, const int* ids,
                                   const float* ws, int k, float mass, float ent) {
  int p;
  uint32_t* r = rec_at(t, row, &p);
  if (!r) return;
  volatile uint32_t* v = r;
  if (t.sys) {
    v[0] = 0u;
    __threadfence_system();
  }
  for (int j = 0; j < 5; j++) {
    const int a = 2 * j, b = 2 * j + 1;
    const uint32_t ia = a < k ? (uint32_t)ids[a] & 0xffffu : 0xffffu;
    const uint32_t ib = b < k ? (uint32_t)ids[b] & 0xffffu : 0xffffu;
    v[1 + j] = ia | (ib << 16);
    const float wa = a < k ? fminf(fmaxf(ws[a], 0.f), 1.f) : 0.f;
    const float wb = b < k ? fminf(fmaxf(ws[b], 0.f), 1.f) : 0.f;
    v[6 + j] = (uint32_t)__float2uint_rn(wa * 65535.f) |
               ((uint32_t)__float2uint_rn(wb * 65535.f) << 16);
  }
  v[11] = __float_as_uint(mass);
  v[12] = __float_as_uint(ent);
  if (t.sys) __threadfence_system();
  v[0] = (uint32_t)(p + 1);
}

}  // namespace viz
using viz::VizTap;

// Publish one pass's routing records (rows x layers) from the device mirror d
// to the host rings h (same addressing). One thread per record; every thread
// runs the fenced tag protocol, so the fence waits overlap.
__global__ void __launch_bounds__(256) k_viz_publish(VizTap d, VizTap h, int rows,
                                                     int layers) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * layers) return;
  const int row = i / layers;
  d.layer = h.layer = i - row * layers;
  int p, ph;
  const uint32_t* s = viz::rec_at(d, row, &p);
  uint32_t* o = viz::rec_at(h, row, &ph);
  if (!s || !o || s[0] != (uint32_t)(p + 1)) return;
  uint32_t w[viz::kRecW];
#pragma unroll
  for (int k = 1; k < viz::kRecW; k++) w[k] = s[k];
  volatile uint32_t* v = o;
  v[0] = 0u;
  __threadfence_system();
#pragma unroll
  for (int k = 1; k < viz::kRecW; k++) v[k] = w[k];
  __threadfence_system();
  v[0] = (uint32_t)(p + 1);
}

// Top-5 of each logits row + softmax probabilities (record t.layer of the
// row's position): [0] tag, [1..5] ids, [6..10] probs (f32), [11] logsumexp.
// One block per row; reads the row once (online max/sum-exp + per-thread
// top-5), thread 0 merges the 256 x 5 candidates. Launched only when tapped.
__global__ void __launch_bounds__(256) k_viz_top5(const float* __restrict__ logits,
                                                  int vocab, VizTap t) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  logits += (size_t)row * vocab;
  float tv[5];
  int ti[5];
#pragma unroll
  for (int j = 0; j < 5; j++) { tv[j] = -INFINITY; ti[j] = -1; }
  float mx = -INFINITY, se = 0.f;
  for (int i = tid; i < vocab; i += 256) {
    const float v = logits[i];
    if (!(v > -INFINITY)) continue;  // -inf / NaN (masked) entries
    if (v > mx) { se = se * __expf(mx - v) + 1.f; mx = v; }
    else se += __expf(v - mx);
    if (v > tv[4]) {
      int j = 4;
      while (j > 0 && v > tv[j - 1]) { tv[j] = tv[j - 1]; ti[j] = ti[j - 1]; j--; }
      tv[j] = v; ti[j] = i;
    }
  }
  __shared__ float smx[256], sse[256];
  __shared__ float rv[256];
  __shared__ int rt[256], ri[256];
  smx[tid] = mx;
  sse[tid] = se;
  __syncthreads();
  for (int off = 128; off > 0; off >>= 1) {
    if (tid < off) {
      const float a = smx[tid], b = smx[tid + off];
      const float m = fmaxf(a, b);
      float s = 0.f;
      if (m > -INFINITY) {
        if (a > -INFINITY) s += sse[tid] * __expf(a - m);
        if (b > -INFINITY) s += sse[tid + off] * __expf(b - m);
      }
      smx[tid] = m;
      sse[tid] = s;
    }
    __syncthreads();
  }
  const float M = smx[0], S = sse[0];
  // merge: 5 rounds of a block argmax over each thread's current list head
  // (largest value, ties -> lowest token id); the winner pops its head
  float bv[5];
  int bi[5];
  int head = 0;
  for (int j = 0; j < 5; j++) {
    rv[tid] = head < 5 ? tv[head] : -INFINITY;
    ri[tid] = head < 5 ? ti[head] : -1;
    rt[tid] = tid;
    __syncthreads();
    for (int off = 128; off > 0; off >>= 1) {
      if (tid < off) {
        const float a = rv[tid], b = rv[tid + off];
        const int ia = ri[tid], ib = ri[tid + off];
        if (ib >= 0 && (ia < 0 || b > a || (b == a && ib < ia))) {
          rv[tid] = b;
          ri[tid] = ib;
          rt[tid] = rt[tid + off];
        }
      }
      __syncthreads();
    }
    bv[j] = rv[0];
    bi[j] = ri[0];
    if (rt[0] == tid && ri[0] >= 0) head++;
    __syncthreads();
  }
  if (tid != 0) return;
  int p;
  uint32_t* r = viz::rec_at(t, row, &p);
  if (!r) return;
  volatile uint32_t* v = r;
  v[0] = 0u;
  __threadfence_system();
  const float lse = M + __logf(S > 0.f ? S : 1.f);
  for (int j = 0; j < 5; j++) {
    v[1 + j] = (uint32_t)bi[j];
    v[6 + j] = __float_as_uint(bi[j] >= 0 ? __expf(bv[j] - lse) : 0.f);
  }
  v[11] = __float_as_uint(lse);
  __threadfence_system();
  v[0] = (uint32_t)(p + 1);
}

// ---- host side: per-slot rings + the token log the VIZ verb reads ----------
namespace viz {

inline bool on() {
  static const bool e = [] {
    const char* s = getenv("QWENOX_VIZ");
    return s && *s && strcmp(s, "0") != 0;
  }();
  return e;
}

struct SlotLog {
  std::mutex mu;
  uint32_t* ring = nullptr;  // this slot's part of g_block
  long long req = -1;
  int state = 0;              // 0 idle, 1 prefill, 2 decode, 3 done
  int n_prompt = 0;
  int fresh_from = 0;         // first position computed by the current request
  std::vector<int> toks;      // prompt + emitted tokens, index = position
  std::vector<double> ms;     // emit time (ms since request start), per generated token
  std::chrono::steady_clock::time_point t0{};
};

inline std::mutex g_mu;
inline std::vector<std::unique_ptr<SlotLog>> g_slots;
inline std::atomic<uint32_t*> g_block{nullptr};  // [g_nslots][kRing][recs][kRecW]
inline uint32_t* g_dev = nullptr;                // device mirror, same layout
inline int g_nslots = 0;
inline bool g_alloc_tried = false;
constexpr int kTabs = 4, kTabRows = 128;
inline int* g_rowtab = nullptr;  // [kTabs][kTabRows] pinned, device-visible
inline int g_tab_next = 0;

inline int recs() { return g_cfg.layers + 1; }
inline size_t ring_bytes() { return (size_t)kRing * recs() * kRecW * 4; }
inline size_t slot_words() { return (size_t)kRing * recs() * kRecW; }

inline SlotLog* slot(int s) {
  if (s < 0 || s > 64) return nullptr;
  std::lock_guard<std::mutex> lk(g_mu);
  if ((size_t)s >= g_slots.size()) g_slots.resize((size_t)s + 1);
  if (!g_slots[(size_t)s]) g_slots[(size_t)s] = std::make_unique<SlotLog>();
  return g_slots[(size_t)s].get();
}

inline void* pinned_mapped(size_t bytes) {
  void* p = nullptr;
  if (hipHostMalloc(&p, bytes, hipHostMallocMapped) != hipSuccess || !p) {
    (void)hipGetLastError();
    return nullptr;
  }
  void* dp = nullptr;
  if (hipHostGetDevicePointer(&dp, p, 0) != hipSuccess || dp != p) {
    // the taps store through the host address: it must be the device one too
    (void)hipGetLastError();
    (void)hipHostFree(p);
    return nullptr;
  }
  memset(p, 0, bytes);
  return p;
}

// Allocate every slot's ring (first call) and register slot s. Call outside
// stream capture (bind, build_graphs, request start); taps never allocate.
inline void ensure(int s) {
  if (!on()) return;
  {
    std::lock_guard<std::mutex> lk(g_mu);
    if (!g_alloc_tried) {
      g_alloc_tried = true;
      const char* e = getenv("QWENOX_PARALLEL");
      int n = e ? atoi(e) : 1;
      n = n < 1 ? 1 : n > 8 ? 8 : n;
      void* b = pinned_mapped(ring_bytes() * (size_t)n);
      g_rowtab = (int*)pinned_mapped((size_t)kTabs * kTabRows * 4);
      void* dv = nullptr;
      if (hipMalloc(&dv, ring_bytes() * (size_t)n) != hipSuccess ||
          hipMemset(dv, 0, ring_bytes() * (size_t)n) != hipSuccess) {
        (void)hipGetLastError();
        dv = nullptr;
      }
      g_dev = (uint32_t*)dv;
      if (!b || !g_rowtab || !g_dev) {
        fprintf(stderr, "viz: pinned allocation failed; viz taps off\n");
      } else {
        g_nslots = n;
        g_block.store((uint32_t*)b);
        fprintf(stderr, "viz: taps on, %d slot ring(s) x %.1f MiB (%d positions x %d records)\n",
                n, ring_bytes() / 1048576.0, kRing, recs());
      }
    }
  }
  if (s < 0 || s >= g_nslots) return;
  SlotLog* sl = slot(s);
  std::lock_guard<std::mutex> lk(sl->mu);
  if (!sl->ring) sl->ring = g_block.load() + (size_t)s * slot_words();
}

// Device-visible base of all host rings, or null when slot s has none.
inline uint32_t* ring(int s) {
  if (!on() || s < 0 || s >= g_nslots) return nullptr;
  return g_block.load(std::memory_order_relaxed);
}
// Base of the device mirror (router taps), or null when slot s has none.
inline uint32_t* dev(int s) {
  if (!on() || s < 0 || s >= g_nslots) return nullptr;
  return g_dev;
}

// Per-row (slot, position) table for a multi-segment batch (eager path: each
// verify round ends with a blocking readback, so kTabs tables are never in
// use when one is rewritten). Returns null when the rows cannot be tagged.
inline const int* rows_table(const int* Ps, const int* bases, const int* sl,
                             const int* voffs, int nseg, int R) {
  if (!g_rowtab || R > kTabRows) return nullptr;
  int* t = g_rowtab + (size_t)g_tab_next * kTabRows;
  g_tab_next = (g_tab_next + 1) % kTabs;
  for (int r = 0; r < R; r++) t[r] = -1;
  for (int s = 0; s < nseg; s++) {
    const bool ok = sl[s] >= 0 && sl[s] < g_nslots;
    for (int r = 0; r < Ps[s]; r++) {
      const int p = bases[s] + r;
      if (ok && p >= 0 && p < (1 << 24)) t[voffs[s] + r] = (sl[s] << 24) | p;
    }
  }
  return t;
}

// Request start on slot s. fresh = positions >= fresh_from are computed by
// this request; keep_ring = the slot continues its live state (positions below
// fresh_from were computed by earlier turns of the same sequence and stay
// valid). Otherwise the ring is wiped (a cache restore did not compute them).
inline void begin(int s, long long req, const std::vector<int>& ids, int fresh_from,
                  bool keep_ring) {
  if (!on()) return;
  ensure(s);
  SlotLog* sl = slot(s);
  if (!sl) return;
  std::lock_guard<std::mutex> lk(sl->mu);
  if (sl->ring) {
    if (!keep_ring) {
      memset(sl->ring, 0, ring_bytes());  // only this slot's part
    } else {
      // continuing the live sequence: records of positions >= fresh_from
      // belong to the previous turn's tail and are about to be recomputed
      for (size_t r = 0; r < (size_t)kRing * recs(); r++) {
        uint32_t* rec = sl->ring + r * kRecW;
        if (rec[0] && (long long)rec[0] - 1 >= fresh_from) rec[0] = 0;
      }
    }
  }
  sl->req = req;
  sl->state = 1;
  sl->n_prompt = (int)ids.size();
  sl->fresh_from = fresh_from;
  sl->toks = ids;
  sl->ms.clear();
  sl->t0 = std::chrono::steady_clock::now();
}

inline void token(int s, int tok) {
  if (!on()) return;
  SlotLog* sl = slot(s);
  if (!sl) return;
  const auto now = std::chrono::steady_clock::now();
  std::lock_guard<std::mutex> lk(sl->mu);
  sl->state = 2;
  sl->toks.push_back(tok);
  sl->ms.push_back(std::chrono::duration<double, std::milli>(now - sl->t0).count());
}

inline void finish(int s) {
  if (!on()) return;
  SlotLog* sl = slot(s);
  if (!sl) return;
  std::lock_guard<std::mutex> lk(sl->mu);
  sl->state = 3;
}

inline void jint(std::string& o, long long v) {
  char b[24];
  snprintf(b, sizeof b, "%lld", v);
  o += b;
}

// VIZ -> summary of every slot.
inline std::string summary_json() {
  std::string o = "{\"on\":";
  o += on() ? "true" : "false";
  o += ",\"ring\":"; jint(o, kRing);
  o += ",\"layers\":"; jint(o, g_cfg.layers);
  o += ",\"experts\":"; jint(o, g_cfg.experts);
  o += ",\"topk\":"; jint(o, g_cfg.topk);
  o += ",\"qsa\":[";
  bool first = true;
  for (int l = 0; l < g_cfg.layers; l++)
    if (is_qsa(l)) { if (!first) o += ','; first = false; jint(o, l); }
  o += "],\"slots\":[";
  std::vector<SlotLog*> v;
  {
    std::lock_guard<std::mutex> lk(g_mu);
    for (auto& p : g_slots) v.push_back(p.get());
  }
  first = true;
  for (size_t s = 0; s < v.size(); s++) {
    if (!v[s]) continue;
    std::lock_guard<std::mutex> lk(v[s]->mu);
    if (!first) o += ',';
    first = false;
    o += "{\"slot\":"; jint(o, (long long)s);
    o += ",\"req\":"; jint(o, v[s]->req);
    o += ",\"state\":"; jint(o, v[s]->state);
    o += ",\"n_prompt\":"; jint(o, v[s]->n_prompt);
    o += ",\"n_tok\":"; jint(o, (long long)v[s]->toks.size());
    o += ",\"fresh_from\":"; jint(o, v[s]->fresh_from);
    o += ",\"ring_ok\":"; o += v[s]->ring ? "true" : "false";
    o += '}';
  }
  o += "]}";
  return o;
}

// Copy record (pos, layer) out if its tag says it is position pos's and it
// did not change while being copied.
inline bool read_rec(const uint32_t* ring, int pos, int layer, uint32_t* out) {
  const volatile uint32_t* r =
      ring + ((size_t)(pos & (kRing - 1)) * recs() + layer) * kRecW;
  const uint32_t want = (uint32_t)(pos + 1);
  if (r[0] != want) return false;
  std::atomic_thread_fence(std::memory_order_acquire);
  for (int i = 0; i < kRecW; i++) out[i] = r[i];
  std::atomic_thread_fence(std::memory_order_acquire);
  return r[0] == want && out[0] == want;
}

// VIZ <slot> <from> <max> -> positions [from, from+max) of slot's request.
// Per position: p, t (token id), m (emit ms for generated tokens, else -1),
// and when final: e (L*10 expert ids, layer-major), w (L*10 weights * 255),
// c (L top-k mass * 1000), h (L entropy * 1000), k/q (top-5 ids / probs *
// 10000, candidates for position p+1). Layers whose record is missing get
// ids -1. Positions with no record at all have no e/w/c/h fields.
inline std::string range_json(int s, int from, int maxn) {
  SlotLog* sl = nullptr;
  {
    std::lock_guard<std::mutex> lk(g_mu);
    if (s >= 0 && (size_t)s < g_slots.size()) sl = g_slots[(size_t)s].get();
  }
  if (!sl) return "{\"error\":\"no such slot\"}";
  if (maxn < 1) maxn = 1;
  if (maxn > 256) maxn = 256;
  std::vector<int> toks;
  std::vector<double> ms;
  long long req;
  int state, n_prompt, fresh_from;
  uint32_t* ring;
  {
    std::lock_guard<std::mutex> lk(sl->mu);
    req = sl->req;
    state = sl->state;
    n_prompt = sl->n_prompt;
    fresh_from = sl->fresh_from;
    ring = sl->ring;
    const int n = (int)sl->toks.size();
    const int a = std::max(0, std::min(from, n));
    const int b = std::min(n, a + maxn);
    toks.assign(sl->toks.begin() + a, sl->toks.begin() + b);
    ms = sl->ms;
    from = a;
  }
  const int ntot = n_prompt + (int)ms.size();
  // the last emitted token is not final (see the rule at the top); during
  // the prompt ingest only the positions the slot already held are
  const int final_end = !ms.empty() ? ntot - 1 : state == 1 ? fresh_from : ntot;
  const int L = g_cfg.layers, K = std::min(g_cfg.topk, 10);
  std::string o;
  o.reserve(toks.size() * 4200 + 256);
  o = "{\"slot\":"; jint(o, s);
  o += ",\"req\":"; jint(o, req);
  o += ",\"state\":"; jint(o, state);
  o += ",\"n_prompt\":"; jint(o, n_prompt);
  o += ",\"n_tok\":"; jint(o, ntot);
  o += ",\"fresh_from\":"; jint(o, fresh_from);
  o += ",\"final_end\":"; jint(o, final_end);
  o += ",\"from\":"; jint(o, from);
  o += ",\"items\":[";
  uint32_t rec[kRecW];
  std::vector<uint32_t> lay((size_t)L * kRecW);
  std::vector<char> have((size_t)L);
  char nb[32];
  for (size_t i = 0; i < toks.size(); i++) {
    const int p = from + (int)i;
    if (i) o += ',';
    o += "{\"p\":"; jint(o, p);
    o += ",\"t\":"; jint(o, toks[i]);
    o += ",\"m\":";
    if (p >= n_prompt && p - n_prompt < (int)ms.size()) {
      snprintf(nb, sizeof nb, "%.2f", ms[(size_t)(p - n_prompt)]);
      o += nb;
    } else {
      o += "-1";
    }
    if (ring && p < final_end) {
      int nh = 0;
      for (int l = 0; l < L; l++) {
        have[(size_t)l] = read_rec(ring, p, l, &lay[(size_t)l * kRecW]);
        nh += have[(size_t)l];
      }
      if (nh) {
        o += ",\"e\":[";
        for (int l = 0; l < L; l++)
          for (int j = 0; j < 10; j++) {
            if (l || j) o += ',';
            if (!have[(size_t)l] || j >= K) { o += "-1"; continue; }
            const uint32_t w = lay[(size_t)l * kRecW + 1 + j / 2];
            jint(o, (j & 1) ? (w >> 16) : (w & 0xffff));
          }
        o += "],\"w\":[";
        for (int l = 0; l < L; l++)
          for (int j = 0; j < 10; j++) {
            if (l || j) o += ',';
            if (!have[(size_t)l] || j >= K) { o += '0'; continue; }
            const uint32_t w = lay[(size_t)l * kRecW + 6 + j / 2];
            const unsigned q = (j & 1) ? (w >> 16) : (w & 0xffff);
            jint(o, (q * 255 + 32767) / 65535);
          }
        o += "],\"c\":[";
        for (int l = 0; l < L; l++) {
          if (l) o += ',';
          float f;
          memcpy(&f, &lay[(size_t)l * kRecW + 11], 4);
          jint(o, have[(size_t)l] ? (long long)lrintf(f * 1000.f) : -1);
        }
        o += "],\"h\":[";
        for (int l = 0; l < L; l++) {
          if (l) o += ',';
          float f;
          memcpy(&f, &lay[(size_t)l * kRecW + 12], 4);
          jint(o, have[(size_t)l] ? (long long)lrintf(f * 1000.f) : -1);
        }
        o += ']';
      }
      if (read_rec(ring, p, L, rec)) {
        o += ",\"k\":[";
        for (int j = 0; j < 5; j++) { if (j) o += ','; jint(o, (int)rec[1 + j]); }
        o += "],\"q\":[";
        for (int j = 0; j < 5; j++) {
          if (j) o += ',';
          float f;
          memcpy(&f, &rec[6 + j], 4);
          jint(o, (long long)lrintf(f * 10000.f));
        }
        o += ']';
      }
    }
    o += '}';
  }
  o += "]}";
  return o;
}

}  // namespace viz
