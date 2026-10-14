// wmma_int_probe.cu — gfx1151 WMMA throughput (f16 / iu8 / iu4) + iu4 fragment
// layout check (10-14, hipfire IU4 feasibility, dev-docs/HIPFIRE-PREFILL-GAP-GPUONLY.md §8).
//
// Build (Windows Git Bash, from repo root, same toolchain as build_win.sh):
//   "$HIPCC" -O3 --offload-arch=gfx1151 tools/wmma_int_probe.cu -o build/wmma_int_probe.exe
// Prints TOPS per instruction type and LAYOUT PASS/FAIL for iu4 / iu8.
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

typedef _Float16 v16h __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef int v8i __attribute__((ext_vector_type(8)));
typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v2i __attribute__((ext_vector_type(2)));

#define CK(x)                                                                 \
  do {                                                                        \
    hipError_t e = (x);                                                       \
    if (e != hipSuccess) {                                                    \
      fprintf(stderr, "HIP %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); \
      exit(1);                                                                \
    }                                                                         \
  } while (0)

constexpr int kChains = 8;

template <int T>  // 0 f16, 1 iu8, 2 iu4
__global__ void __launch_bounds__(256) k_tput(int iters, int seed, float* out) {
  const int lane = threadIdx.x & 31;
  v8f accf[kChains];
  v8i acci[kChains];
#pragma unroll
  for (int c = 0; c < kChains; ++c) {
    accf[c] = v8f{0, 0, 0, 0, 0, 0, 0, 0};
    acci[c] = v8i{0, 0, 0, 0, 0, 0, 0, 0};
  }
  v16h ah, bh;
  v4i a8, b8;
  v2i a4, b4;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    ah[i] = (_Float16)(((lane + i + seed) & 7) * 0.125f);
    bh[i] = (_Float16)(((lane * 3 + i) & 7) * 0.125f);
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    a8[i] = 0x01020304 * (lane + i + seed);
    b8[i] = 0x02030405 * (lane + i);
  }
  a4[0] = 0x12345678 ^ (lane + seed);
  a4[1] = 0x23456789 ^ lane;
  b4[0] = 0x3456789A ^ lane;
  b4[1] = 0x456789AB ^ (lane * 7);
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int c = 0; c < kChains; ++c) {
      if constexpr (T == 0)
        accf[c] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ah, bh, accf[c]);
      else if constexpr (T == 1)
        acci[c] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a8, true, b8, acci[c], false);
      else
        acci[c] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a4, true, b4, acci[c], false);
    }
  }
  float s = 0;
#pragma unroll
  for (int c = 0; c < kChains; ++c)
#pragma unroll
    for (int e = 0; e < 8; ++e) s += accf[c][e] + (float)acci[c][e];
  if (s == 12345.f) out[0] = s;  // keep live
}

// Layout check: A[16][16], B[16][16] (B stored as [col][k]), int values.
// Assumed fragment layout (same as the f16 path in 26/27_kernels):
//   A: lane l (both halves) holds row l & 15, K 0..15 packed little-endian
//      (iu4: nibble k of dword k/8; iu8: byte k of dword k/4);
//   B: lane l holds column l & 15, same K packing;
//   C: v8i[e] = (row 2e + (l >> 4), col l & 15).
template <int T>
__global__ void k_layout(const int8_t* A, const int8_t* B, int* C) {
  const int lane = threadIdx.x & 31, r = lane & 15;
  v8i acc = v8i{0, 0, 0, 0, 0, 0, 0, 0};
  if constexpr (T == 2) {
    v2i a, b;
    for (int d = 0; d < 2; ++d) {
      uint32_t wa = 0, wb = 0;
      for (int n = 0; n < 8; ++n) {
        wa |= (uint32_t)(A[r * 16 + d * 8 + n] & 15) << (4 * n);
        wb |= (uint32_t)(B[r * 16 + d * 8 + n] & 15) << (4 * n);
      }
      a[d] = (int)wa;
      b[d] = (int)wb;
    }
    acc = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a, true, b, acc, false);
  } else {
    v4i a, b;
    for (int d = 0; d < 4; ++d) {
      uint32_t wa = 0, wb = 0;
      for (int n = 0; n < 4; ++n) {
        wa |= (uint32_t)(uint8_t)A[r * 16 + d * 4 + n] << (8 * n);
        wb |= (uint32_t)(uint8_t)B[r * 16 + d * 4 + n] << (8 * n);
      }
      a[d] = (int)wa;
      b[d] = (int)wb;
    }
    acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, acc, false);
  }
  for (int e = 0; e < 8; ++e) C[(2 * e + (lane >> 4)) * 16 + r] = acc[e];
}

template <int T>
static bool layout_check() {
  std::mt19937 g(7 + T);
  std::vector<int8_t> A(256), B(256);
  for (auto& v : A) v = T == 2 ? (int8_t)((int)(g() % 16) - 8) : (int8_t)((int)(g() % 256) - 128);
  for (auto& v : B) v = T == 2 ? (int8_t)((int)(g() % 16) - 8) : (int8_t)((int)(g() % 256) - 128);
  int8_t *dA, *dB;
  int* dC;
  CK(hipMalloc(&dA, 256));
  CK(hipMalloc(&dB, 256));
  CK(hipMalloc(&dC, 1024));
  CK(hipMemcpy(dA, A.data(), 256, hipMemcpyHostToDevice));
  CK(hipMemcpy(dB, B.data(), 256, hipMemcpyHostToDevice));
  k_layout<T><<<1, 32>>>(dA, dB, dC);
  std::vector<int> C(256);
  CK(hipMemcpy(C.data(), dC, 1024, hipMemcpyDeviceToHost));
  int bad = 0;
  for (int i = 0; i < 16; ++i)
    for (int j = 0; j < 16; ++j) {
      int s = 0;
      for (int k = 0; k < 16; ++k) s += A[i * 16 + k] * B[j * 16 + k];
      if (s != C[i * 16 + j]) ++bad;
    }
  CK(hipFree(dA));
  CK(hipFree(dB));
  CK(hipFree(dC));
  printf("LAYOUT %s %s (%d/256 mismatched)\n", T == 2 ? "iu4" : "iu8", bad ? "FAIL" : "PASS", bad);
  return !bad;
}

template <int T>
static double tput(const char* name) {
  float* d;
  CK(hipMalloc(&d, 4));
  const int blocks = 40 * 16, iters = 4096;
  k_tput<T><<<blocks, 256>>>(64, 1, d);  // warm
  CK(hipDeviceSynchronize());
  double best = 0;
  for (int r = 0; r < 3; ++r) {
    auto t0 = std::chrono::steady_clock::now();
    k_tput<T><<<blocks, 256>>>(iters, r, d);
    CK(hipDeviceSynchronize());
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    double ops = (double)blocks * 8 * iters * kChains * 16 * 16 * 16 * 2;
    best = std::max(best, ops / s / 1e12);
  }
  printf("TPUT %-4s %.1f TOPS\n", name, best);
  CK(hipFree(d));
  return best;
}

int main() {
  const bool l4 = layout_check<2>();
  const bool l8 = layout_check<1>();
  const double f = tput<0>("f16"), i8 = tput<1>("iu8"), i4 = tput<2>("iu4");
  printf("RATIO iu8/f16 %.2f  iu4/f16 %.2f\n", i8 / f, i4 / f);
  printf("PROBE %s\n", (l4 && l8) ? "PASS" : "FAIL");
  return 0;
}
