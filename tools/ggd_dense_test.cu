// ggd_dense_test.cu — GGUF native dense kernels (src/gpu/parts/26_kernels_dense_gguf.inc)
// on the real non-expert matrices of a GGUF file.
//
//   ggd_dense_test FIRST_SHARD.gguf [--rows N] [--iters N] [--only NAME]
//
// For the largest dense matrix of each native type (Q4_K Q5_K Q6_K IQ4_XS IQ4_NL)
// plus output.weight:
//   deq   k_ggd_deq_bf16 == f2bf(gguf::dequant_row) bit for bit (first N rows)
//   gemv  P = 1 vs an f64 CPU reference on the CPU-dequantized weights (rel-L2)
//   rowinv  P in {2,3,5,8,13,24,48} (<= 8-token passes, as ggd_gemv): every
//         token's output bit-identical to its own P = 1 run, f32 and bf16 x
//   time  P = 1 over the full matrix, GB/s of native bytes
// Prints RESULT PASS / FAIL.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "../src/gguf.h"

__device__ inline uint16_t f2bf(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  u += 0x7FFF + ((u >> 16) & 1);
  return (uint16_t)(u >> 16);
}
static uint16_t f2bf_h(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  u += 0x7FFF + ((u >> 16) & 1);
  return (uint16_t)(u >> 16);
}
static float bf2f_h(uint16_t b) {
  uint32_t u = (uint32_t)b << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}
#include "../src/gpu/parts/26_kernels_dense_gguf.inc"

#define CK(x)                                                                         \
  do {                                                                                \
    hipError_t e_ = (x);                                                              \
    if (e_ != hipSuccess) {                                                           \
      fprintf(stderr, "HIP %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); \
      exit(2);                                                                        \
    }                                                                                 \
  } while (0)

template <class F>
static void with_type(int t, F&& f) {
  switch (t) {
    case ggd::kQ4K: f(std::integral_constant<int, ggd::kQ4K>{}); break;
    case ggd::kQ5K: f(std::integral_constant<int, ggd::kQ5K>{}); break;
    case ggd::kQ6K: f(std::integral_constant<int, ggd::kQ6K>{}); break;
    case ggd::kIQ4NL: f(std::integral_constant<int, ggd::kIQ4NL>{}); break;
    default: f(std::integral_constant<int, ggd::kIQ4XS>{}); break;
  }
}
// same pass structure as ggd_gemv (30_host_util.inc)
static void run_gemv(int t, const uint8_t* w, uint64_t rb, const void* x, bool xb, uint64_t xs,
                     float* y, uint64_t ys, int P, int R, int C) {
  const unsigned blocks = (unsigned)((R + 3) / 4);
  for (int off = 0; off < P; off += 8) {
    const int pn = std::min(8, P - off);
    const void* xp = xb ? (const void*)((const uint16_t*)x + (size_t)off * xs)
                        : (const void*)((const float*)x + (size_t)off * xs);
    float* yp = y + (size_t)off * ys;
    with_type(t, [&](auto tc_) {
      constexpr int T = decltype(tc_)::value;
#define G(NP)                                                                                   \
  case NP:                                                                                      \
    if (xb) k_ggd_gemv<T, NP, true><<<blocks, 128>>>(w, rb, xp, xs, yp, ys, R, C);             \
    else k_ggd_gemv<T, NP, false><<<blocks, 128>>>(w, rb, xp, xs, yp, ys, R, C);               \
    break;
      switch (pn) { G(1) G(2) G(3) G(4) G(5) G(6) G(7) G(8) }
#undef G
    });
  }
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s FIRST_SHARD.gguf [--rows N] [--iters N] [--only NAME]\n", argv[0]);
    return 2;
  }
  int max_rows = 4096, iters = 30;
  std::string only;
  for (int i = 2; i + 1 < argc; i += 2) {
    std::string a = argv[i];
    if (a == "--rows") max_rows = atoi(argv[i + 1]);
    else if (a == "--iters") iters = atoi(argv[i + 1]);
    else if (a == "--only") only = argv[i + 1];
    else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
  }
  gguf::File f(argv[1]);
  // largest dense 2-D matrix per native type (+ output.weight)
  std::vector<const gguf::Tensor*> pick;
  const int types[5] = {ggd::kQ4K, ggd::kQ5K, ggd::kQ6K, ggd::kIQ4XS, ggd::kIQ4NL};
  for (int t : types) {
    const gguf::Tensor* best = nullptr;
    for (const auto& n : f.order()) {
      const gguf::Tensor& x = f.at(n);
      if ((int)x.type != t || x.nd != 2 || n.find("_exps") != std::string::npos ||
          n == "token_embd.weight" || n == "output.weight" || n.find("per_layer") != std::string::npos)
        continue;
      if (!best || x.nbytes > best->nbytes) best = &x;
    }
    if (best) pick.push_back(best);
  }
  if (const gguf::Tensor* o = f.find("output.weight"); o && ggd::type_ok((int)o->type)) pick.push_back(o);
  if (!only.empty()) {
    pick.clear();
    pick.push_back(&f.at(only));
  }
  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0.f, 1.f);
  bool ok = true;
  constexpr int PMAX = 48;
  for (const gguf::Tensor* tp : pick) {
    const gguf::Tensor& T = *tp;
    const int t = (int)T.type, C = (int)T.ne[0];
    const uint64_t R = T.rows(), rb = T.row_bytes();
    if (rb != ggd::row_bytes(t, C)) { printf("%s: row_bytes mismatch\n", T.name.c_str()); ok = false; continue; }
    const int Rc = (int)std::min<uint64_t>(R, (uint64_t)max_rows);
    printf("== %-28s %-6s %llu x %d (%.1f MiB)\n", T.name.c_str(), gguf::type_name(t),
           (unsigned long long)R, C, T.nbytes / 1048576.0);
    uint8_t* dw;
    CK(hipMalloc(&dw, T.nbytes + 256));
    CK(hipMemcpy(dw, T.data, T.nbytes, hipMemcpyHostToDevice));
    // CPU dequant of the first Rc rows
    std::vector<float> wf((size_t)Rc * C);
    for (int r = 0; r < Rc; r++) gguf::dequant_row(t, T.data + (uint64_t)r * rb, &wf[(size_t)r * C], C);
    // deq
    uint16_t* dq;
    CK(hipMalloc(&dq, (size_t)Rc * C * 2));
    with_type(t, [&](auto tc_) {
      constexpr int TT = decltype(tc_)::value;
      const uint64_t n = (uint64_t)Rc * (C / 32);
      k_ggd_deq_bf16<TT><<<(unsigned)((n + 255) / 256), 256>>>(dw, rb, dq, Rc, C);
    });
    CK(hipDeviceSynchronize());
    std::vector<uint16_t> hq((size_t)Rc * C);
    CK(hipMemcpy(hq.data(), dq, hq.size() * 2, hipMemcpyDeviceToHost));
    size_t bad = 0;
    for (size_t i = 0; i < hq.size(); i++) bad += hq[i] != f2bf_h(wf[i]);
    printf("   deq: %zu / %zu bf16 differ from CPU dequant %s\n", bad, hq.size(), bad ? "FAIL" : "ok");
    if (bad) ok = false;
    // x
    std::vector<float> x((size_t)PMAX * C);
    for (auto& v : x) v = nd(rng);
    std::vector<uint16_t> xb(x.size());
    for (size_t i = 0; i < x.size(); i++) xb[i] = f2bf_h(x[i]);
    float* dx; uint16_t* dxb; float *dy, *dy1;
    CK(hipMalloc(&dx, x.size() * 4));
    CK(hipMalloc(&dxb, xb.size() * 2));
    CK(hipMemcpy(dx, x.data(), x.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(dxb, xb.data(), xb.size() * 2, hipMemcpyHostToDevice));
    CK(hipMalloc(&dy, (size_t)PMAX * Rc * 4));
    CK(hipMalloc(&dy1, (size_t)PMAX * Rc * 4));
    // gemv P = 1 accuracy (token 0)
    run_gemv(t, dw, rb, dx, false, C, dy, Rc, 1, Rc, C);
    CK(hipDeviceSynchronize());
    std::vector<float> y(Rc);
    CK(hipMemcpy(y.data(), dy, Rc * 4, hipMemcpyDeviceToHost));
    double e2 = 0, r2 = 0;
    for (int r = 0; r < Rc; r++) {
      double s = 0;
      for (int c = 0; c < C; c++) s += (double)wf[(size_t)r * C + c] * x[c];
      e2 += (y[r] - s) * (y[r] - s);
      r2 += s * s;
    }
    const double rel = sqrt(e2 / std::max(r2, 1e-300));
    printf("   gemv P=1: rel-L2 vs f64 CPU %.3e %s\n", rel, rel < 1e-5 ? "ok" : "FAIL");
    if (!(rel < 1e-5)) ok = false;
    // row invariance
    for (int xbm = 0; xbm < 2; xbm++) {
      const void* xp = xbm ? (const void*)dxb : (const void*)dx;
      for (int p = 0; p < PMAX; p++)  // per-token P = 1 references
        run_gemv(t, dw, rb, xbm ? (const void*)(dxb + (size_t)p * C) : (const void*)(dx + (size_t)p * C),
                 xbm, C, dy1 + (size_t)p * Rc, Rc, 1, Rc, C);
      CK(hipDeviceSynchronize());
      std::vector<float> ref((size_t)PMAX * Rc), got((size_t)PMAX * Rc);
      CK(hipMemcpy(ref.data(), dy1, ref.size() * 4, hipMemcpyDeviceToHost));
      int nbad = 0;
      for (int P : {2, 3, 5, 8, 13, 24, 48}) {
        CK(hipMemset(dy, 0xFF, (size_t)P * Rc * 4));
        run_gemv(t, dw, rb, xp, xbm, C, dy, Rc, P, Rc, C);
        CK(hipDeviceSynchronize());
        CK(hipMemcpy(got.data(), dy, (size_t)P * Rc * 4, hipMemcpyDeviceToHost));
        if (memcmp(got.data(), ref.data(), (size_t)P * Rc * 4)) nbad++;
      }
      printf("   rowinv x=%s: %s\n", xbm ? "bf16" : "f32 ", nbad ? "FAIL" : "P 2..48 == P=1 bitwise");
      if (nbad) ok = false;
    }
    // timing, full matrix P = 1, cycling over copies (>= 256 MiB total) so the
    // weights come from DRAM, not the 32 MiB MALL
    const int ncp = (int)std::min<uint64_t>(64, ((256ull << 20) + T.nbytes - 1) / T.nbytes);
    const size_t cstride = (T.nbytes + 4095) & ~(size_t)4095;
    uint8_t* dwc;
    CK(hipMalloc(&dwc, cstride * ncp));
    for (int i = 0; i < ncp; i++) CK(hipMemcpy(dwc + i * cstride, dw, T.nbytes, hipMemcpyDeviceToDevice));
    float* dyf;
    CK(hipMalloc(&dyf, R * 4 * 2));
    hipEvent_t a, b;
    CK(hipEventCreate(&a));
    CK(hipEventCreate(&b));
    for (int i = 0; i < 3; i++) run_gemv(t, dwc, rb, dx, false, C, dyf, R, 1, (int)R, C);
    CK(hipEventRecord(a));
    for (int i = 0; i < iters; i++)
      run_gemv(t, dwc + (i % ncp) * cstride, rb, dx, false, C, dyf, R, 1, (int)R, C);
    CK(hipEventRecord(b));
    CK(hipEventSynchronize(b));
    float ms;
    CK(hipEventElapsedTime(&ms, a, b));
    ms /= iters;
    const double q8 = (double)R * C * 34 / 32;
    printf("   time P=1: %.1f us, %.0f GB/s native (q8g32 would read %.2fx the bytes)\n", ms * 1e3,
           T.nbytes / (ms * 1e-3) / 1e9, q8 / T.nbytes);
    CK(hipFree(dw)); CK(hipFree(dq)); CK(hipFree(dx)); CK(hipFree(dxb));
    CK(hipFree(dy)); CK(hipFree(dy1)); CK(hipFree(dyf)); CK(hipFree(dwc));
  }
  printf("%s\n", ok ? "RESULT PASS" : "RESULT FAIL");
  return ok ? 0 : 1;
}
