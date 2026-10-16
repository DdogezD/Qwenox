// gpuload.hip -- GPU-only memory load for N seconds (no NPU). Used to see
// which PM-table entries follow FCLK under load.
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
__global__ void Copy(const uint4* __restrict__ s, uint4* __restrict__ d, size_t n) {
  for (size_t i = blockIdx.x * size_t{blockDim.x} + threadIdx.x; i < n; i += size_t{gridDim.x} * blockDim.x) d[i] = s[i];
}
int main(int argc, char** argv) {
  const double secs = argc > 1 ? std::atof(argv[1]) : 10;
  const size_t bytes = size_t{1} << 30;
  void *a, *b;
  if (hipMalloc(&a, bytes) != hipSuccess || hipMalloc(&b, bytes) != hipSuccess) return 1;
  hipMemset(a, 1, bytes);
  const auto t0 = std::chrono::steady_clock::now();
  int it = 0;
  for (;; ++it) {
    Copy<<<2048, 256>>>((const uint4*)a, (uint4*)b, bytes / 16);
    if (it % 8 == 7) {
      hipDeviceSynchronize();
      if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() >= secs) break;
    }
  }
  const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("gpuload: %.1f s, %.1f GB/s\n", s, 2.0 * bytes * (it + 1) / s / 1e9);
  return 0;
}
