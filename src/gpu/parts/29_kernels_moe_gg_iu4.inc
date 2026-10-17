// --- GGUF-native W4A4 routed experts (QWENOX_GG_A4) ----------------------------
//
// 已废弃（2026-10-17）：禁止用于生产或测速口径，仅留档。证据：
// - 精度：IQ 码表（IQ4_NL/XS 非线性 16 值、IQ3_S 符号+网格）数学上无法由
//   单个线性 int4 无损表示；权重线性化 rel-L2 11-14%（CPU 逐元素对拍
//   tools/gg_lin_decomp.cpp，与 kernel 输出误差吻合到 5% 内，排除实现 bug）。
//   KLD 64x512：0.1002 -> 0.1661 / 0.1736 / 0.2306（=1/2/3），是 hgn A4
//   （+0.035 已超 +0.002 门禁）的 2-4 倍。div 扫描 7-12，9.0 即最优点。
// - 收益：kernel 仅 1.29x（hgn A4 为 1.63x）——IQ 解码与 qtab LDS 查表吃掉
//   大半 tensor 收益；64K e2e +3.6~4.6%，8K +4.6%。
// - 无损替代不存在：gfx1151 实测 iu8 53.7 TOPS ≈ f16 53.6，只有 iu4 106 有
//   2x（tools/wmma_int_probe.cu）；双 pass 高低 nibble 同样跌回 f16 速率。
//   GGUF prefill 的剩余空间在 F16 解码路径（当前约 43% 峰值），不在 int4。
//
// The hgn v2 W4A4 kernel (27_kernels_moe_lut.inc k_moe_iu4) assumes the
// weights are already linear int4 (i4r). GGUF IQ experts are not: IQ4_NL /
// IQ4_XS are non-linear 16-entry codebooks and IQ3_S is sign+magnitude with a
// 3-bit grid, all with one scale per 32 elements. A single int4 WMMA pass
// therefore has to linearize each 128-element group on the fly:
//
//   per (row, 128-K group):
//     S = amax / div                       (amax over the 128 actual weights)
//     q_b(c) = clamp(rint(d_b * v(c) / S), -8, 7)   (16-entry table per 32-block)
//     q_i    = q_b(code_i)                 (pair-order nibbles, two's complement)
//
// then the ordinary W4A4 pipeline: k_quant_a4 activations, 8 iu4 MMAs per
// 128-K stage, acc += int32 * (S * act_scale). The raw GGUF blocks stay the
// source of truth; no model conversion and no persistent int4 copy. The
// temporary int4 codes live only in LDS for the current tile.
//
// Quality note: this is the same lossy linearization as an offline i4r
// conversion (IQ3_S ~11-12%, IQ4 ~10% extra weight error) but without the
// +4.7 GiB model copy. For lossless weights use two int4 passes (hi/lo
// nibbles) or int8; both lose the 2x tensor rate.
//
// Gate/up: grid (m / 128, tiles), two weight passes (gate then up), plain
//   hid = up * silu(gate) epilogue (no hgn v2 rotations).
// Down: grid (m / 128, tiles), f16 pairs output like k_moe_iu4<false>.

#pragma once

namespace ggiu4 {

constexpr int kIQ4NL = 20, kIQ4XS = 23, kIQ3S = 21;

template <int T>
__host__ __device__ constexpr size_t row_bytes(size_t k) {
  return T == kIQ4NL ? (k / 32) * 18
         : T == kIQ4XS ? (k / 256) * 136
                       : (k / 256) * 110;
}

// Fetch one 32-element sub-block of `row` (global sub-block index ib) and
// return its 16 pair-order code bytes (byte b = elements 2b | 2b+1 << 4).
// sc = d_b, the format's own per-32 scale.
template <int T>
__device__ __forceinline__ uint4 fetch_sub(const uint8_t* row, int ib, float& sc) {
  if constexpr (T == kIQ4NL) {
    const uint8_t* blk = row + (size_t)ib * 18;
    uint16_t dh;
    __builtin_memcpy(&dh, blk, 2);
    sc = __half2float(__builtin_bit_cast(__half, dh));
    uint4 q;
    __builtin_memcpy(&q, blk + 2, 16);
    return moelut::iq_to_pairs(q);
  } else if constexpr (T == kIQ4XS) {
    const int sb = ib >> 3, sub = ib & 7;
    const uint8_t* blk = row + (size_t)sb * 136;
    uint16_t dh, sh;
    __builtin_memcpy(&dh, blk, 2);
    __builtin_memcpy(&sh, blk + 2, 2);
    const uint8_t* sl = blk + 4;
    const int ls = ((sl[sub >> 1] >> (4 * (sub & 1))) & 0xF) |
                   (((sh >> (2 * sub)) & 3) << 4);
    sc = __half2float(__builtin_bit_cast(__half, dh)) * (float)(ls - 32);
    uint4 q;
    __builtin_memcpy(&q, blk + 8 + 16 * sub, 16);
    return moelut::iq_to_pairs(q);
  } else {
    const int sb = ib >> 3, sub = ib & 7;
    const uint8_t* blk = row + (size_t)sb * 110;
    uint16_t dh;
    __builtin_memcpy(&dh, blk, 2);
    const int sc4 = (blk[106 + (sub >> 1)] >> (4 * (sub & 1))) & 0xF;
    sc = __half2float(__builtin_bit_cast(__half, dh)) * (float)(1 + 2 * sc4);
    const uint8_t* qs = blk + 2 + 8 * sub;
    const uint32_t qh = blk[66 + sub];
    const uint8_t* sg = blk + 74 + 4 * sub;
    uint32_t w[4];
#pragma unroll
    for (int l = 0; l < 4; ++l) {
      const uint32_t g1 = moegg::c_iq3s_grid[qs[2 * l] | ((qh << (8 - 2 * l)) & 256)];
      const uint32_t g2 = moegg::c_iq3s_grid[qs[2 * l + 1] | ((qh << (7 - 2 * l)) & 256)];
      w[l] = moelut::iq3s_word(g1, g2, sg[l]);
    }
    return make_uint4(w[0], w[1], w[2], w[3]);
  }
}

// Map one word of 8 pair-order codes through a 256-entry byte table: one LDS
// lookup per code byte (two nibbles), like the F16 path's s_cbp.
__device__ __forceinline__ uint32_t map_word(uint32_t w, const uint8_t* qbp) {
  return (uint32_t)qbp[w & 0xFF] | ((uint32_t)qbp[(w >> 8) & 0xFF] << 8) |
         ((uint32_t)qbp[(w >> 16) & 0xFF] << 16) | ((uint32_t)qbp[w >> 24] << 24);
}

__device__ __forceinline__ uint4 map_codes(uint4 c, const uint8_t* qbp) {
  return make_uint4(map_word(c.x, qbp), map_word(c.y, qbp), map_word(c.z, qbp),
                    map_word(c.w, qbp));
}

}  // namespace ggiu4

// div: S = amax / div. 9.0 is close to the MSE-searched 128-group scale on
// IQ3_S / IQ4_NL samples (a small candidate search can be added later).
template <int kType, bool kPair>
__launch_bounds__(256) __global__
    void k_moe_gg_iu4(const uint8_t* __restrict__ w, const uint8_t* __restrict__ w_up,
                      const uint8_t* __restrict__ x4, const float* __restrict__ xs,
                      const int* __restrict__ tokidx, const MoeTile* __restrict__ tiles,
                      const int* __restrict__ ntiles, __half* __restrict__ out, int m, int k,
                      const uint8_t* __restrict__ qtab, float div) {
  using namespace ggiu4;
  constexpr int BM = 128, BN = 64, T = BN / 16;
  constexpr int kActStrideB = 80;
  constexpr int kCodeBytes = BM * 64, kWscBytes = BM * 2, kActBytes = BN * kActStrideB,
                kSaBytes = BN * 4;
  constexpr int kStage = kCodeBytes + kWscBytes + kActBytes + kSaBytes;
  constexpr int kMidRow = BM + 8;
  constexpr int kPlaneB = BN * kMidRow * 2;
  constexpr int kStageOff = kPair ? kPlaneB : 0;
  constexpr int kLds = kPair ? kStageOff + (kStage > kPlaneB ? kStage : kPlaneB) : kStage;
  __shared__ __attribute__((aligned(16))) uint8_t lds[kLds];
  __shared__ uint8_t s_qtab[8192];  // [sign][16 ratio levels][256 code-pair bytes]
  uint4* s_codes = reinterpret_cast<uint4*>(lds + kStageOff);
  __half* s_wsc = reinterpret_cast<__half*>(lds + kStageOff + kCodeBytes);
  uint8_t* s_act = lds + kStageOff + kCodeBytes + kWscBytes;
  float* s_sa = reinterpret_cast<float*>(lds + kStageOff + kCodeBytes + kWscBytes + kActBytes);

  if ((int)blockIdx.y >= *ntiles) return;
  const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31, sub = lane & 15, half = lane >> 4;
  {
    const int base = tid * 32;
#pragma unroll
    for (int i = 0; i < 2; ++i)
      *reinterpret_cast<uint4*>(s_qtab + base + i * 16) =
          *reinterpret_cast<const uint4*>(qtab + base + i * 16);
  }
  __syncthreads();

  const MoeTile tile = tiles[blockIdx.y];
  const int slot0 = tile.first, nrows = tile.count, live = (nrows + 15) / 16;
  const int r_block = (int)blockIdx.x * BM;
  const int nst = k / 128, nb = k / 128;
  const size_t wrb = row_bytes<kType>((size_t)k);

  const int fs = tid >> 1, fpart = tid & 1;
  const int flr = (fs & ~15) | ((fs & 1) << 3) | ((fs & 15) >> 1);
  const uint8_t* fw;
  int frow;
  const auto set_fetch = [&](int pass) {
    const uint8_t* base = (kPair && pass) ? w_up : w;
    frow = tile.expert * m + r_block + flr;
    if (frow >= (tile.expert + 1) * m) frow = (tile.expert + 1) * m - 1;
    fw = base + (size_t)frow * wrb;
  };
  const int at = tid >> 2, ac = tid & 3;
  const uint8_t* fa = nullptr;
  const float* fas = nullptr;
  if (at < nrows) {
    const int src = tokidx ? tokidx[slot0 + at] : slot0 + at;
    fa = x4 + (size_t)src * (k / 2) + ac * 16;
    fas = xs + (size_t)src * nb;
  }
  uint4 ra;
  float ras = 0.f;
  const auto fetch_act = [&](int st) {
    ra = fa ? *reinterpret_cast<const uint4*>(fa + st * 64) : make_uint4(0u, 0u, 0u, 0u);
    if (ac == 0) ras = fa ? fas[st] : 0.f;
  };
  const auto swz = [](int slot, int c) { return slot * 4 + (c ^ ((slot >> 1) & 3)); };

  uint4 q0, q1;
  __half sw;
  const auto fetch_w = [&](int st) {
    const int ib0 = 4 * st + 2 * fpart;
    float d0, d1;
    const uint4 c0 = fetch_sub<kType>(fw, ib0, d0);
    const uint4 c1 = fetch_sub<kType>(fw, ib0 + 1, d1);
    constexpr float kvmax = kType == kIQ3S ? 15.f : 127.f;
    const float ad0 = fabsf(d0), ad1 = fabsf(d1);
    float md = fmaxf(ad0, ad1);
    md = fmaxf(md, __shfl_xor_sync(0xffffffffffffffffULL, md, 1));
    const float sf = md > 0.f ? md * kvmax / div : 0.f;
    const float sf16 = __half2float(__float2half(sf));
    int l0 = md > 0.f ? (int)rintf(ad0 / md * 15.f) : 0;
    int l1 = md > 0.f ? (int)rintf(ad1 / md * 15.f) : 0;
    l0 = l0 < 0 ? 0 : l0 > 15 ? 15 : l0;
    l1 = l1 < 0 ? 0 : l1 > 15 ? 15 : l1;
    const int s0 = d0 < 0.f ? 16 : 0, s1 = d1 < 0.f ? 16 : 0;
    q0 = map_codes(c0, s_qtab + (s0 + l0) * 256);
    q1 = map_codes(c1, s_qtab + (s1 + l1) * 256);
    sw = __float2half(sf16);
  };
  const auto commit = [&]() {
    s_codes[swz(fs, fpart * 2)] = q0;
    s_codes[swz(fs, fpart * 2 + 1)] = q1;
    if (fpart == 0) s_wsc[flr] = sw;
    *reinterpret_cast<uint4*>(s_act + at * kActStrideB + ac * 16) = ra;
    if (ac == 0) s_sa[at] = ras;
  };

  moegg::v8f acc[T];
  const int slot = wave * 16 + sub;
  const auto compute = [&]() {
    uint4 a[4];
#pragma unroll
    for (int c = 0; c < 4; ++c) a[c] = s_codes[swz(slot, c)];
    const uint4 wsv = *reinterpret_cast<const uint4*>(s_wsc + wave * 16 + half * 8);
    const uint32_t wsw[4] = {wsv.x, wsv.y, wsv.z, wsv.w};
    float ws[8];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const __half2 h = __builtin_bit_cast(__half2, wsw[q]);
      ws[2 * q] = __low2float(h);
      ws[2 * q + 1] = __high2float(h);
    }
#pragma unroll
    for (int j = 0; j < T; ++j) {
      if (j >= live) continue;
      const int t = j * 16 + sub;
      const uint8_t* ap = s_act + t * kActStrideB;
      moeiu4::v8i ci = moeiu4::v8i{0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const uint4 b = *reinterpret_cast<const uint4*>(ap + c * 16);
        ci = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, moeiu4::v2i{(int)a[c].x, (int)a[c].y}, true,
                                                        moeiu4::v2i{(int)b.x, (int)b.y}, ci, false);
        ci = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, moeiu4::v2i{(int)a[c].z, (int)a[c].w}, true,
                                                        moeiu4::v2i{(int)b.z, (int)b.w}, ci, false);
      }
      const float sa = s_sa[t];
#pragma unroll
      for (int l = 0; l < 8; ++l) acc[j][l] = fmaf((float)ci[l], ws[l] * sa, acc[j][l]);
    }
  };

  const auto spill = [&](__half* plane) {
#pragma unroll
    for (int j = 0; j < T; ++j) {
      if (j >= live) break;
      const int t = j * 16 + sub;
      if (t >= nrows) continue;
      __half2 h2[4];
#pragma unroll
      for (int q = 0; q < 4; ++q)
        h2[q] = __halves2half2(__float2half(acc[j][2 * q]), __float2half(acc[j][2 * q + 1]));
      uint4 v;
      __builtin_memcpy(&v, h2, 16);
      *reinterpret_cast<uint4*>(plane + t * kMidRow + wave * 16 + half * 8) = v;
    }
  };

  for (int pass = 0; pass < (kPair ? 2 : 1); ++pass) {
    set_fetch(pass);
#pragma unroll
    for (int j = 0; j < T; ++j) acc[j] = moegg::v8f{0, 0, 0, 0, 0, 0, 0, 0};
    fetch_w(0);
    fetch_act(0);
    for (int st = 0; st < nst; ++st) {
      commit();
      __syncthreads();
      if (st + 1 < nst) {
        fetch_w(st + 1);
        fetch_act(st + 1);
      }
      compute();
      __syncthreads();
    }
    if constexpr (kPair) spill(reinterpret_cast<__half*>(lds) + (pass ? kPlaneB / 2 : 0));
  }

  if constexpr (kPair) {
    __syncthreads();
    const __half* gplane = reinterpret_cast<const __half*>(lds);
    const __half* uplane = gplane + kPlaneB / 2;
    for (int s = wave; s < nrows; s += 8) {
      const __half* gr = gplane + s * kMidRow;
      const __half* ur = uplane + s * kMidRow;
      float g[4], u[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        g[j] = __half2float(gr[j * 32 + lane]);
        u[j] = __half2float(ur[j * 32 + lane]);
      }
      __half* o = out + (size_t)(slot0 + s) * m + r_block;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float v = g[j] / (1.f + __expf(-g[j])) * u[j];
        o[j * 32 + lane] = __float2half(v);
      }
    }
  } else {
    const int rr = r_block + wave * 16 + half * 8;
#pragma unroll
    for (int j = 0; j < T; ++j) {
      if (j >= live) break;
      const int t = j * 16 + sub;
      if (t >= nrows) continue;
      __half2 h2[4];
#pragma unroll
      for (int q = 0; q < 4; ++q) h2[q] = __floats2half2_rn(acc[j][2 * q], acc[j][2 * q + 1]);
      uint4 v;
      __builtin_memcpy(&v, h2, 16);
      *reinterpret_cast<uint4*>(out + (size_t)(slot0 + t) * m + rr) = v;
    }
  }
}

// Host launchers. Return false when the type/shape has no instance.
static bool moe_gg_iu4_up(int type, const uint8_t* gate, const uint8_t* up,
                          const uint8_t* x4, const float* xs, const int* tokidx,
                          const MoeTile* tiles, const int* ntiles, int max_tiles,
                          __half* hid, int m, int k, const uint8_t* qtab, float div,
                          hipStream_t st) {
  if (m % 128 || k % 128) return false;
  dim3 grid(m / 128, max_tiles);
  if (type == moegg::kIq4nl)
    k_moe_gg_iu4<ggiu4::kIQ4NL, true><<<grid, 256, 0, st>>>(gate, up, x4, xs, tokidx, tiles, ntiles, hid, m, k, qtab, div);
  else if (type == moegg::kIq4xs)
    k_moe_gg_iu4<ggiu4::kIQ4XS, true><<<grid, 256, 0, st>>>(gate, up, x4, xs, tokidx, tiles, ntiles, hid, m, k, qtab, div);
  else if (type == moegg::kIq3s)
    k_moe_gg_iu4<ggiu4::kIQ3S, true><<<grid, 256, 0, st>>>(gate, up, x4, xs, tokidx, tiles, ntiles, hid, m, k, qtab, div);
  else
    return false;
  return true;
}

static bool moe_gg_iu4_down(int type, const uint8_t* dn, const uint8_t* x4, const float* xs,
                            const MoeTile* tiles, const int* ntiles, int max_tiles,
                            __half* pairs, int m, int k, const uint8_t* qtab, float div,
                            hipStream_t st) {
  if (m % 128 || k % 128) return false;
  dim3 grid(m / 128, max_tiles);
  if (type == moegg::kIq4nl)
    k_moe_gg_iu4<ggiu4::kIQ4NL, false><<<grid, 256, 0, st>>>(dn, nullptr, x4, xs, nullptr, tiles, ntiles, pairs, m, k, qtab, div);
  else if (type == moegg::kIq4xs)
    k_moe_gg_iu4<ggiu4::kIQ4XS, false><<<grid, 256, 0, st>>>(dn, nullptr, x4, xs, nullptr, tiles, ntiles, pairs, m, k, qtab, div);
  else
    return false;
  return true;
}


// Host-side fold table for the per-128-scale GGUF A4 kernel:
// tab[(sign * 16 + level) * 256 + byte] = two packed int4 q nibbles.
static int gg_host_value(int type, int c) {
  if (type == moegg::kIq3s) return (c & 8) ? -(2 * (c & 7) + 1) : (2 * c + 1);
  static const int kv[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                             1, 13, 25, 38, 53, 69, 89, 113};
  return kv[c];
}

static bool gg_init_qtab(int type, uint8_t* d_tab, float div, hipStream_t st) {
  if (type != moegg::kIq4nl && type != moegg::kIq4xs && type != moegg::kIq3s) return false;
  const int vmax = type == moegg::kIq3s ? 15 : 127;
  uint8_t h[8192];
  for (int sign = 0; sign < 2; ++sign)
    for (int lev = 0; lev < 16; ++lev) {
      const float r = lev / 15.f;
      uint8_t* row = h + (sign * 16 + lev) * 256;
      for (int byte = 0; byte < 256; ++byte) {
        const float sg = sign ? -1.f : 1.f;
        int qlo = (int)rintf(sg * r * (float)gg_host_value(type, byte & 15) * div / (float)vmax);
        int qhi = (int)rintf(sg * r * (float)gg_host_value(type, byte >> 4) * div / (float)vmax);
        qlo = qlo < -8 ? -8 : qlo > 7 ? 7 : qlo;
        qhi = qhi < -8 ? -8 : qhi > 7 ? 7 : qhi;
        row[byte] = (uint8_t)((qlo & 15) | ((qhi & 15) << 4));
      }
    }
  // Synchronous: the host buffer is local and must stay valid until the copy
  // completes; startup-only, so the sync is free.
  (void)st;
  return hipMemcpy(d_tab, h, sizeof h, hipMemcpyHostToDevice) == hipSuccess;
}
