// Equivalence test: fused backward_pass vs the cuBLAS + backward_hardgelu
// reference chain. Compares dz2 (o_dLdW1x), dz1 (o_dLdW0x) and dLdx.
//
// Pre-activations are drawn from (-2.5, 2.5) so all three hardGELU-derivative
// branches (dead / quadratic / identity) are exercised.

#include <openntc/ntc_kernel.cuh>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

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

// Row-major helper, identical to openntc.cpp: C(n x m) = A^T B, A is (k x n).
static void matmulATB(cublasHandle_t handle, int n, int m, int k, float* A,
                      float* B, float* C)
{
  float alpha = 1.0f, beta = 0.0f;
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha, B, m, A, n,
              &beta, C, m);
}

static float* device_upload(std::mt19937& gen, int count, float lo, float hi)
{
  std::uniform_real_distribution<float> dist(lo, hi);
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
  std::printf("%-6s %s  (max |diff| %.3e, %d/%d bad)\n", name,
              bad ? "FAIL" : "PASS", max_diff, bad, count);
  return bad;
}

int main(int argc, char** argv)
{
  int n = argc > 1 ? std::atoi(argv[1]) : 8 * 256 * 256;
  std::mt19937 gen(1337);

  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);

  float* W0 = device_upload(gen, 64 * FEAT_DIM, -0.5f, 0.5f);
  float* W1 = device_upload(gen, 64 * 64, -0.5f, 0.5f);
  float* Wout = device_upload(gen, OUT_DIM * 64, -0.5f, 0.5f);
  float* dLdPred = device_upload(gen, OUT_DIM * n, -1.0f, 1.0f);
  // pre-activations spanning all three derivative branches
  float* W0x = device_upload(gen, 64 * n, -2.5f, 2.5f);
  float* W1x = device_upload(gen, 64 * n, -2.5f, 2.5f);

  // reference buffers
  float* r_dh2 = device_alloc(64 * n);
  float* r_dz2 = device_alloc(64 * n);
  float* r_dh1 = device_alloc(64 * n);
  float* r_dz1 = device_alloc(64 * n);
  float* r_dLdx = device_alloc(FEAT_DIM * n);
  // fused buffers
  float* f_dz2 = device_alloc(64 * n);
  float* f_dz1 = device_alloc(64 * n);
  float* f_dLdx = device_alloc(FEAT_DIM * n);
  CUDA_CHECK();

  // --- reference: the cuBLAS + backward_hardgelu chain (old path, verbatim)
  matmulATB(handle, 64, n, OUT_DIM, Wout, dLdPred, r_dh2);
  launch_backward_hardgelu(64 * n, W1x, r_dh2, r_dz2);
  matmulATB(handle, 64, n, 64, W1, r_dz2, r_dh1);
  launch_backward_hardgelu(64 * n, W0x, r_dh1, r_dz1);
  matmulATB(handle, FEAT_DIM, n, 64, W0, r_dz1, r_dLdx);
  CUDA_CHECK();

  // --- fused
  launch_backward_pass(n, W0, W1, Wout, dLdPred, W0x, W1x, f_dLdx, f_dz1,
                       f_dz2);
  cudaDeviceSynchronize();
  CUDA_CHECK();

  int bad = 0;
  bad += compare("dz2", r_dz2, f_dz2, 64, n);
  bad += compare("dz1", r_dz1, f_dz1, 64, n);
  bad += compare("dLdx", r_dLdx, f_dLdx, FEAT_DIM, n);

  std::printf(bad ? "\nFAILED\n" : "\nALL PASS\n");
  return bad ? 1 : 0;
}
