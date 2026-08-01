// Equivalence test: fused forward_pass vs the cuBLAS + hardgelu reference.
// Run before deleting the old path; rerun whenever forward_pass changes.
//
// Fills x/W0/W1/Wout with seeded pseudo-random values, runs both paths on
// identical inputs, and compares all five output buffers. Exit 0 = match.

#include <openntc/ntc_kernel.cuh>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

// must match ntc_kernel.cu
#define FEAT_DIM 57
#define OUT_DIM 9

#define CUDA_CHECK()                                                       \
  do {                                                                     \
    cudaError_t e = cudaGetLastError();                                    \
    if (e != cudaSuccess) {                                                \
      std::printf("CUDA error: %s @ %s:%d\n", cudaGetErrorString(e),       \
                  __FILE__, __LINE__);                                     \
      std::exit(2);                                                        \
    }                                                                      \
  } while (0)

// Row-major helpers, identical to openntc.cpp.
static void matmulAB(cublasHandle_t handle, int n, int m, int k, float* A,
                     float* B, float* C)
{
  float alpha = 1.0f, beta = 0.0f;
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, B, m, A, k,
              &beta, C, m);
}

static float* device_upload(std::mt19937& gen, int count)
{
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
  std::vector<float> host(count);
  for (float& v : host)
    v = dist(gen);
  float* dev = nullptr;
  cudaMalloc(&dev, count * sizeof(float));
  cudaMemcpy(dev, host.data(), count * sizeof(float),
             cudaMemcpyHostToDevice);
  return dev;
}

static float* device_alloc(int count)
{
  float* dev = nullptr;
  cudaMalloc(&dev, count * sizeof(float));
  cudaMemset(dev, 0, count * sizeof(float));
  return dev;
}

// Compares two device buffers; returns number of mismatches (prints first 5).
static int compare(const char* name, float* a_dev, float* b_dev, int rows,
                   int n)
{
  int count = rows * n;
  std::vector<float> a(count), b(count);
  cudaMemcpy(a.data(), a_dev, count * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(b.data(), b_dev, count * sizeof(float), cudaMemcpyDeviceToHost);
  int bad = 0;
  float max_diff = 0.0f;
  for (int i = 0; i < count; i++)
  {
    float diff = fabsf(a[i] - b[i]);
    float tol = 1e-5f * fmaxf(1.0f, fmaxf(fabsf(a[i]), fabsf(b[i])));
    if (diff > max_diff) max_diff = diff;
    if (diff > tol)
    {
      if (bad < 5)
        std::printf("  %s MISMATCH at row %d texel %d: ref %.7f fused %.7f\n",
                    name, i / n, i % n, a[i], b[i]);
      bad++;
    }
  }
  std::printf("%-8s %s  (max |diff| %.3e, %d/%d bad)\n", name,
              bad ? "FAIL" : "PASS", max_diff, bad, count);
  return bad;
}

int main(int argc, char** argv)
{
  int n = argc > 1 ? std::atoi(argv[1]) : 8 * 256 * 256;  // production shape
  std::mt19937 gen(42);

  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);

  float* x = device_upload(gen, FEAT_DIM * n);
  float* W0 = device_upload(gen, 64 * FEAT_DIM);
  float* W1 = device_upload(gen, 64 * 64);
  float* Wout = device_upload(gen, OUT_DIM * 64);

  // reference path buffers
  float* r_W0x = device_alloc(64 * n);
  float* r_W0xa = device_alloc(64 * n);
  float* r_W1x = device_alloc(64 * n);
  float* r_W1xa = device_alloc(64 * n);
  float* r_Woutx = device_alloc(OUT_DIM * n);
  // fused path buffers
  float* f_W0x = device_alloc(64 * n);
  float* f_W0xa = device_alloc(64 * n);
  float* f_W1x = device_alloc(64 * n);
  float* f_W1xa = device_alloc(64 * n);
  float* f_Woutx = device_alloc(OUT_DIM * n);
  CUDA_CHECK();

  // --- reference: cuBLAS GEMMs + hardgelu kernels (the old path, verbatim)
  matmulAB(handle, 64, n, FEAT_DIM, W0, x, r_W0x);
  launch_forward_hardgelu(64 * n, r_W0x, r_W0xa);
  matmulAB(handle, 64, n, 64, W1, r_W0xa, r_W1x);
  launch_forward_hardgelu(64 * n, r_W1x, r_W1xa);
  matmulAB(handle, OUT_DIM, n, 64, Wout, r_W1xa, r_Woutx);
  CUDA_CHECK();

  // --- fused
  launch_forward_pass(n, x, W0, W1, Wout, f_W0x, f_W0xa, f_W1x, f_W1xa,
                      f_Woutx);
  cudaDeviceSynchronize();
  CUDA_CHECK();

  int bad = 0;
  bad += compare("W0x", r_W0x, f_W0x, 64, n);
  bad += compare("W0xa", r_W0xa, f_W0xa, 64, n);
  bad += compare("W1x", r_W1x, f_W1x, 64, n);
  bad += compare("W1xa", r_W1xa, f_W1xa, 64, n);
  bad += compare("Woutx", r_Woutx, f_Woutx, OUT_DIM, n);

  std::printf(bad ? "\nFAILED\n" : "\nALL PASS\n");
  return bad ? 1 : 0;
}
