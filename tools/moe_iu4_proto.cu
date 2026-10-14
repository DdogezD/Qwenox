// moe_iu4_proto.cu — W4A4 (int4 x int4 -> int32 WMMA) routed-expert prototype
// for hgn v2 i4r experts (10-14, hipfire IU4 route; dev-docs/HIPFIRE-PREFILL-GAP-GPUONLY.md §8).
//
// DEPRECATED（10-14 结论）：引擎开关 QWENOX_MOE_A4 已废弃——KLD 0.1075 →
// 0.1427（+0.035），并发 prefill 不安全，64K 只快 8.9%。本工具仅留档/复测。
//
// Kernels now live in 27_kernels_moe_lut.inc. Compares, on one real v2 layer and synthetic
// skewed top-10 routing, the production F16 WMMA kernels (moe_lut_up_mid /
// moe_lut_down f16 pairs, BN=64 tiles) with:
//   k_quant_a4  : f16 [N][K] -> int4 codes (pair order, two's complement) +
//                 one f32 scale per 128 elements (MSE-best clip of 5 candidates)
//   k_moe_iu4   : BM=128 x BN=64, one 128-K scale block per stage, 8
//                 v_wmma_i32_16x16x16_iu4 per (stage, token tile), then
//                 acc += float(int32) * (w_scale[row] * a_scale[tok]).
//                 i4r nibbles (nib-8 offset binary) become two's complement by
//                 ^0x8 — no decode. up = gate/up + the k_moe_lut kMid FWHT mid
//                 epilogue (verbatim math) -> hid f16; down -> f16 pairs.
// Checks: (1) exact-grid inputs (activations = int4 * 2^-e per block, so A4 is
//         lossless) -> iu4 vs f16 must agree to ~1e-3 (kernel correctness);
//         (2) gaussian x through up -> quant -> down vs the f16 chain
//         (quantization error); every output must be written and finite.
//
//   moe_iu4_proto --v2 models/qwen38-flash-next-v2.hgn [--layer 24] [--P 8192] [--iters 10]
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "../src/gguf.h"
#include "../src/hgn.h"

#define CK(x)                                                                    \
  do {                                                                           \
    hipError_t e_ = (x);                                                         \
    if (e_ != hipSuccess) {                                                      \
      fprintf(stderr, "HIP %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); \
      exit(2);                                                                   \
    }                                                                            \
  } while (0)

struct MoeTile { int expert, first, count; };
#include "../src/gpu/parts/26_kernels_moe_gguf.inc"
#include "../src/gpu/parts/27_kernels_moe_lut.inc"

// ---------------------------------------------------------------------------
static uint8_t* upload(const void* src, size_t n) {
  uint8_t* d;
  CK(hipMalloc(&d, n));
  CK(hipMemcpy(d, src, n, hipMemcpyHostToDevice));
  return d;
}

struct Err {
  double num = 0, den = 0;
  size_t bad = 0, n = 0;
  double rel() const { return den > 0 ? std::sqrt(num / den) : 0; }
};
static Err cmp(const __half* d_a, const __half* d_ref, size_t n) {
  std::vector<__half> a(n), r(n);
  CK(hipMemcpy(a.data(), d_a, n * 2, hipMemcpyDeviceToHost));
  CK(hipMemcpy(r.data(), d_ref, n * 2, hipMemcpyDeviceToHost));
  Err e;
  e.n = n;
  for (size_t i = 0; i < n; i++) {
    const double x = __half2float(a[i]), y = __half2float(r[i]);
    if (!std::isfinite(x) || !std::isfinite(y)) {
      e.bad++;
      continue;
    }
    e.num += (x - y) * (x - y);
    e.den += y * y;
  }
  return e;
}

int main(int argc, char** argv) {
  std::string v2_path;
  int layer = 24, P = 8192, iters = 10, topk = 10;
  unsigned seed = 1;
  for (int i = 1; i + 1 < argc; i += 2) {
    std::string a = argv[i];
    if (a == "--v2") v2_path = argv[i + 1];
    else if (a == "--layer") layer = atoi(argv[i + 1]);
    else if (a == "--P") P = atoi(argv[i + 1]);
    else if (a == "--iters") iters = atoi(argv[i + 1]);
    else if (a == "--seed") seed = (unsigned)atoi(argv[i + 1]);
    else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
  }
  if (v2_path.empty()) {
    fprintf(stderr, "usage: %s --v2 V2.hgn [--layer L] [--P N] [--iters N]\n", argv[0]);
    return 2;
  }
  hgn::Checkpoint ck(v2_path.c_str());
  char nm[128];
  snprintf(nm, sizeof nm, "layers.%d.mlp.experts.gate_up_proj.weight", layer);
  const hgn::Tensor& tgu = ck.at(nm);
  snprintf(nm, sizeof nm, "layers.%d.mlp.experts.down_proj.weight", layer);
  const hgn::Tensor& tdn = ck.at(nm);
  if (tgu.dtype != 23 || tdn.dtype != 23) {
    fprintf(stderr, "expected dtype 23 experts\n");
    return 2;
  }
  const int E = (int)tgu.dims[0], D = (int)tgu.dims[tgu.ndims - 1];
  const uint64_t rows_gu = tgu.numel() / (uint64_t)D;
  const int MID = (int)(rows_gu / (uint64_t)E / 2);
  const uint64_t rows_dn = tdn.numel() / (uint64_t)MID;
  if (D % 128 || MID % 128 || rows_dn != (uint64_t)E * D) {
    fprintf(stderr, "unexpected shapes\n");
    return 2;
  }
  const float cb[16] = {-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7};
  float* d_cb;
  CK(hipMalloc(&d_cb, sizeof cb));
  CK(hipMemcpy(d_cb, cb, sizeof cb, hipMemcpyHostToDevice));
  const uint8_t* dgu = upload(tgu.data, tgu.data_size);
  const uint8_t* ddn = upload(tdn.data, tdn.data_size);
  const moelut::LutW pgu = lut_i4r_view(dgu, rows_gu, (uint64_t)D, d_cb, 2 * MID, MID);
  const moelut::LutW pdn = lut_i4r_view(ddn, rows_dn, (uint64_t)MID, d_cb, D, 0);
  const Iu4W qgu{dgu, dgu + rows_gu * D / 2, 2 * MID, MID};
  const Iu4W qdn{ddn, ddn + rows_dn * MID / 2, D, 0};
  auto sidecar = [&](const char* suffix, size_t want) {
    snprintf(nm, sizeof nm, "layers.%d.mlp.experts.%s", layer, suffix);
    const hgn::Tensor& t = ck.at(nm);
    if (t.dtype != 2 || t.numel() != want) {
      fprintf(stderr, "bad sidecar %s\n", nm);
      exit(2);
    }
    std::vector<float> h(want);
    for (size_t i = 0; i < want; i++) h[i] = hgn::fp16_to_f32(((const uint16_t*)t.data)[i]);
    float* d;
    CK(hipMalloc(&d, want * 4));
    CK(hipMemcpy(d, h.data(), want * 4, hipMemcpyHostToDevice));
    return d;
  };
  float* d_svh = sidecar("gate_up_proj.svh", (size_t)2 * MID);
  float* d_suh = sidecar("down_proj.suh", (size_t)MID);
  printf("v2 layer %d: E=%d D=%d MID=%d P=%d top%d\n", layer, E, D, MID, P, topk);

  // routing (mild skew, as tools/moe_lut_test.cu)
  std::mt19937 rng(seed);
  std::vector<double> pop(E);
  for (int e = 0; e < E; e++) pop[e] = 1.0 / (1.0 + ((e * 2654435761u) % E) / 64.0);
  std::discrete_distribution<int> pick(pop.begin(), pop.end());
  std::vector<int> ids((size_t)P * topk);
  for (int t = 0; t < P; t++)
    for (int s = 0; s < topk; s++) {
      int e;
      bool dup;
      do {
        e = pick(rng);
        dup = false;
        for (int q = 0; q < s; q++) dup |= ids[(size_t)t * topk + q] == e;
      } while (dup);
      ids[(size_t)t * topk + s] = e;
    }
  const int npairs = P * topk;
  std::vector<int> eoff(E + 1, 0), tokidx(npairs);
  for (int p = 0; p < npairs; p++) eoff[ids[p] + 1]++;
  for (int e = 0; e < E; e++) eoff[e + 1] += eoff[e];
  {
    std::vector<int> cur(eoff.begin(), eoff.end() - 1);
    for (int t = 0; t < P; t++)
      for (int s = 0; s < topk; s++) tokidx[cur[ids[(size_t)t * topk + s]]++] = t;
  }
  std::vector<MoeTile> tiles;
  for (int e = 0; e < E; e++)
    for (int r = eoff[e]; r < eoff[e + 1]; r += 64) tiles.push_back({e, r, std::min(64, eoff[e + 1] - r)});
  const int ntiles = (int)tiles.size();
  const int max_tiles = (npairs + 63) / 64 + E;
  printf("tiles %d (max %d)\n", ntiles, max_tiles);

  int *d_tok, *d_nt;
  MoeTile* d_tiles;
  CK(hipMalloc(&d_tok, npairs * 4));
  CK(hipMalloc(&d_nt, 4));
  CK(hipMalloc(&d_tiles, (size_t)max_tiles * sizeof(MoeTile)));
  CK(hipMemcpy(d_tok, tokidx.data(), npairs * 4, hipMemcpyHostToDevice));
  CK(hipMemcpy(d_nt, &ntiles, 4, hipMemcpyHostToDevice));
  CK(hipMemcpy(d_tiles, tiles.data(), ntiles * sizeof(MoeTile), hipMemcpyHostToDevice));

  const size_t nx = (size_t)P * D, nh = (size_t)npairs * MID, npr = (size_t)npairs * D;
  __half *d_x, *d_hid_ref, *d_hid_q, *d_pr_ref, *d_pr_q, *d_pr_chain, *d_hidx;
  uint8_t *d_x4, *d_h4;
  float *d_xs, *d_hs;
  CK(hipMalloc(&d_x, nx * 2));
  CK(hipMalloc(&d_hidx, nh * 2));
  CK(hipMalloc(&d_hid_ref, nh * 2));
  CK(hipMalloc(&d_hid_q, nh * 2));
  CK(hipMalloc(&d_pr_ref, npr * 2));
  CK(hipMalloc(&d_pr_q, npr * 2));
  CK(hipMalloc(&d_pr_chain, npr * 2));
  CK(hipMalloc(&d_x4, nx / 2));
  CK(hipMalloc(&d_h4, nh / 2));
  CK(hipMalloc(&d_xs, nx / 128 * 4));
  CK(hipMalloc(&d_hs, nh / 128 * 4));

  hipEvent_t e0, e1;
  CK(hipEventCreate(&e0));
  CK(hipEventCreate(&e1));
  auto timeit = [&](auto&& fn) {
    fn();
    CK(hipDeviceSynchronize());
    CK(hipEventRecord(e0, 0));
    for (int r = 0; r < iters; r++) fn();
    CK(hipEventRecord(e1, 0));
    CK(hipEventSynchronize(e1));
    float ms;
    CK(hipEventElapsedTime(&ms, e0, e1));
    return ms / iters;
  };
  auto nan_fill = [&](void* p, size_t bytes) { CK(hipMemset(p, 0xFF, bytes)); };
  auto quant = [&](const __half* src, int N, int K, uint8_t* q, float* s) {
    const long waves = (long)N * (K / 128);
    k_quant_a4<<<(unsigned)((waves + 7) / 8), 256>>>(src, N, K, q, s);
  };
  const dim3 gup(MID / 128, max_tiles), gdn(D / 128, max_tiles);
  auto ref_up = [&](const __half* x, __half* hid) {
    if (!moe_lut_up_mid(pgu, x, d_tok, d_tiles, d_nt, max_tiles, hid, d_svh, d_suh, MID, D, 0)) exit(3);
  };
  auto ref_dn = [&](const __half* hid, __half* pr) {
    if (!moe_lut_down(moelut::kI4R, pdn, hid, d_tiles, d_nt, max_tiles, (float*)pr, D, MID, 0, true)) exit(3);
  };
  auto q_up = [&](__half* hid) {
    k_moe_iu4<true><<<gup, 256>>>(qgu, d_x4, d_xs, d_tok, d_tiles, d_nt, hid, MID, D, d_svh, d_suh);
  };
  auto q_dn = [&](__half* pr) {
    k_moe_iu4<false><<<gdn, 256>>>(qdn, d_h4, d_hs, nullptr, d_tiles, d_nt, pr, D, MID, nullptr, nullptr);
  };

  bool ok = true;
  // (1) exact-grid inputs: per 128-block ints in [-7,7] (one +-7) * 2^-e.
  {
    auto gen_exact = [&](size_t n, std::vector<__half>& h) {
      h.resize(n);
      std::uniform_int_distribution<int> qi(-7, 7), ex(2, 6);
      for (size_t b = 0; b < n / 128; b++) {
        const float s = std::ldexp(1.f, -ex(rng));
        for (int i = 0; i < 128; i++) h[b * 128 + i] = __float2half((float)qi(rng) * s);
        h[b * 128 + (b % 128)] = __float2half((b & 1 ? 7.f : -7.f) * s);
      }
    };
    std::vector<__half> hx, hh;
    gen_exact(nx, hx);
    gen_exact(nh, hh);
    CK(hipMemcpy(d_x, hx.data(), nx * 2, hipMemcpyHostToDevice));
    CK(hipMemcpy(d_hidx, hh.data(), nh * 2, hipMemcpyHostToDevice));
    nan_fill(d_hid_ref, nh * 2);
    nan_fill(d_hid_q, nh * 2);
    nan_fill(d_pr_ref, npr * 2);
    nan_fill(d_pr_q, npr * 2);
    ref_up(d_x, d_hid_ref);
    ref_dn(d_hidx, d_pr_ref);
    quant(d_x, P, D, d_x4, d_xs);
    quant(d_hidx, npairs, MID, d_h4, d_hs);
    q_up(d_hid_q);
    q_dn(d_pr_q);
    CK(hipGetLastError());
    CK(hipDeviceSynchronize());
    const Err eu = cmp(d_hid_q, d_hid_ref, nh), ed = cmp(d_pr_q, d_pr_ref, npr);
    const bool pass = eu.bad == 0 && ed.bad == 0 && eu.rel() < 5e-3 && ed.rel() < 5e-3;
    printf("EXACT up rel %.2e (nonfinite %zu)  down rel %.2e (nonfinite %zu)  %s\n", eu.rel(), eu.bad,
           ed.rel(), ed.bad, pass ? "PASS" : "FAIL");
    ok &= pass;
  }
  // (2) gaussian activations: quantization error of the A4 chain.
  {
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<__half> hx(nx);
    for (auto& v : hx) v = __float2half(nd(rng));
    CK(hipMemcpy(d_x, hx.data(), nx * 2, hipMemcpyHostToDevice));
    nan_fill(d_hid_ref, nh * 2);
    nan_fill(d_hid_q, nh * 2);
    nan_fill(d_pr_ref, npr * 2);
    nan_fill(d_pr_q, npr * 2);
    nan_fill(d_pr_chain, npr * 2);
    ref_up(d_x, d_hid_ref);
    ref_dn(d_hid_ref, d_pr_ref);
    quant(d_x, P, D, d_x4, d_xs);
    q_up(d_hid_q);
    quant(d_hid_ref, npairs, MID, d_h4, d_hs);
    q_dn(d_pr_q);  // down alone: A4 of the reference hid
    quant(d_hid_q, npairs, MID, d_h4, d_hs);
    q_dn(d_pr_chain);  // full chain
    CK(hipGetLastError());
    CK(hipDeviceSynchronize());
    const Err eu = cmp(d_hid_q, d_hid_ref, nh), ed = cmp(d_pr_q, d_pr_ref, npr),
              ec = cmp(d_pr_chain, d_pr_ref, npr);
    printf("GAUSS up rel %.4f  down rel %.4f  chain rel %.4f  (nonfinite %zu/%zu/%zu)\n", eu.rel(), ed.rel(),
           ec.rel(), eu.bad, ed.bad, ec.bad);
    ok &= eu.bad == 0 && ed.bad == 0 && ec.bad == 0;
  }
  // timing (inputs from (2))
  const float t_ru = timeit([&] { ref_up(d_x, d_hid_ref); });
  const float t_rd = timeit([&] { ref_dn(d_hid_ref, d_pr_ref); });
  const float t_qx = timeit([&] { quant(d_x, P, D, d_x4, d_xs); });
  const float t_qu = timeit([&] { q_up(d_hid_q); });
  const float t_qh = timeit([&] { quant(d_hid_q, npairs, MID, d_h4, d_hs); });
  const float t_qd = timeit([&] { q_dn(d_pr_chain); });
  printf("TIME f16  up %.3f ms  down %.3f ms  total %.3f\n", t_ru, t_rd, t_ru + t_rd);
  printf("TIME iu4  up %.3f ms  down %.3f ms  quant_x %.3f  quant_h %.3f  total %.3f\n", t_qu, t_qd, t_qx, t_qh,
         t_qu + t_qd + t_qx + t_qh);
  printf("SPEEDUP up %.2fx  down %.2fx  total(incl quant) %.2fx\n", t_ru / t_qu, t_rd / t_qd,
         (t_ru + t_rd) / (t_qu + t_qd + t_qx + t_qh));
  printf("PROTO %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
