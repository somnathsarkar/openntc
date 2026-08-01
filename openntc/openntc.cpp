#include <openntc/ntc_kernel.cuh>
#include <openntc/dataloader.h>
#include <iostream>
#include <random>
#include <algorithm>
#include <cassert>

#include <cublas_v2.h>
#include <curand_kernel.h>

void initialize_decoder_weights(std::mt19937& gen, int feature_dim, int out_dim, float** W0, float** W1, float** Wout)
{
  *W0 = new float[64 * feature_dim];
  *W1 = new float[64 * 64];
  *Wout = new float[out_dim * 64];

  // bounds = +-1/sqrt(feature_dim) (feature_dim = 57)
  std::uniform_real_distribution<float> dist_W0(-0.1325f, 0.1325f);

  for (int i = 0; i < 64; i++)
  {
    for (int j = 0; j < feature_dim; j++)
    {
      (*W0)[i * feature_dim + j] = dist_W0(gen);
    }
  }
  
  // bounds = +-1/sqrt(64)
  std::uniform_real_distribution<float> dist_W1out(-0.125f, 0.125f);

  for (int i = 0; i < 64; i++)
  {
    for (int j = 0; j < 64; j++)
    {
      (*W1)[i * 64 + j] = dist_W1out(gen);
    }
  }

  for (int i = 0; i < out_dim; i++)
  {
    for (int j = 0; j < 64; j++)
    {
      (*Wout)[i * 64 + j] = dist_W1out(gen);
    }
  }
}

// C = A * B, where A, B, C are row-major. C is n x m, A is n x k, B is k x m

void matmulAB(cublasHandle_t handle, int n, int m, int k, float* A, float* B, float* C)
{
  float sgemm_alpha = 1.0f, sgemm_beta = 0.0f;
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &sgemm_alpha, B, m, A, k, &sgemm_beta, C, m);
}

// C = A^T * B, where A, B, C are row-major. C is n x m, A is k x n, B is k x m

void matmulATB(cublasHandle_t handle, int n, int m, int k, float* A, float* B, float* C)
{
  float sgemm_alpha = 1.0f, sgemm_beta = 0.0f;
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &sgemm_alpha, B, m, A, n, &sgemm_beta, C, m);
}

// C = A * B^T, where A, B, C are row-major. C is n x m, A is n x k, B is m x k

void matmulABT(cublasHandle_t handle, int n, int m, int k, float* A, float* B, float* C)
{
  float sgemm_alpha = 1.0f, sgemm_beta = 0.0f;
  cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &sgemm_alpha, B, k, A, k, &sgemm_beta, C, m);
}

float sinc(float x)
{
  const float pi = acosf(-1.0f);
  return sinf(pi * x) / (pi * x);
}

float lanczos(float x, float a)
{
  return sinc(x) * sinc(x / a);
}

void initialize_lanczos_weights(int a, int s, float** w)
{
  int n = 2 * a * s;
  *w = new float[n];
  for (int i = 0; i < n; i++)
  {
    float off = (i + ((1 - n) * 0.5f)) / s;
    (*w)[i] = lanczos(off, (float) a);
  }
}

void initialize_grid(std::mt19937& gen, float** g, int n, int m, int k)
{
  *g = new float[n * m * k];
  
  std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

  for (int i = 0; i < n; i++)
  {
    for (int j = 0; j < m; j++)
    {
      for (int l = 0; l < k; l++)
      {
        (*g)[i * m * k + j * k + l] = dist(gen);
      }
    }
  }
}

float cosine_annealing(float lr_min, float lr_max, int t_max, int t_cur)
{
  float lr = lr_min + 0.5f * (lr_max - lr_min) * (1.0f + cosf(t_cur * acosf(-1.0f) / t_max));
  return lr;
}

const char* map_phases_to_labels[7] = {
  "draw_features",
  "draw_targets",
  "forward_pass",
  "loss",
  "backward_pass",
  "accumulate_grid_gradients",
  "update_adam"};

class PerfTimer
{
  public:
    static const int num_phases = 7;
    static const int len_window = 500;
    static const int num_warmup = 50;
    PerfTimer()
    {
      for (int i = 0; i < len_window; i++)
      {
        for (int j = 0; j < num_phases + 1; j++)
        {
          cudaEventCreate(&events[i][j]);
        }
        lods[i] = -1;
      }
    }
    ~PerfTimer()
    {
      for (int i = 0; i < len_window; i++)
      {
        for (int j = 0; j < num_phases + 1; j++)
        {
          cudaEventDestroy(events[i][j]);
        }
      }
    }
    void log(int batch_i, int phase, int lod)
    {
      int step = batch_i - num_warmup;
      if (step < 0 || step >= len_window) return;
      assert(lods[step] == -1 || lods[step] == lod);
      cudaEventRecord(events[step][phase]);
      lods[step] = lod;
    }
    void harvest()
    {
      cudaEventSynchronize(events[len_window - 1][num_phases]);
      for (int step = 0; step < len_window; step++)
      {
        for (int phase = 0; phase < num_phases; phase++)
        {
          float ms;
          cudaEventElapsedTime(&ms, events[step][phase], events[step][phase + 1]);
          accum_ms[phase] += ms;
        }
      }
      double total = 0.0;
      for (int i = 0; i < num_phases; i++)
      {
        accum_ms[i] /= len_window;
        total += accum_ms[i];
        printf("%s: %f ms/step\n", map_phases_to_labels[i], accum_ms[i]);
      }
      printf("Total: %f ms/step\n", total);
    }
  private:
    cudaEvent_t events[len_window][num_phases + 1];
    double accum_ms[num_phases] = {};
    int lods[len_window];
};

int main()
{
  // std::random_device rd;
  std::mt19937 gen(123);

  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);

  int g0_num_bytes = 2;
  int g1_num_bytes = 4;
  float g0_delta = 2.0f / powf(2.0f, (float) g0_num_bytes);
  float g1_delta = 2.0f / powf(2.0f, (float) g1_num_bytes);
  int g0_dim = 8;
  int g1_dim = 12;
  int mip_dim[9] = {1024, 512, 256, 128, 64, 32, 16, 8, 4};
  int g0_grid_dim[4] = { 256, 64, 16, 4 };
  int g1_grid_dim[4] = { 128, 32, 8, 2 };
  int feature_dim = 4 * g0_dim + g1_dim + 12 + 1;
  int max_batch_dim = 8 * 256 * 256;
  int out_dim = 9;

  float* g0_host[4];
  float* g1_host[4];
  float* g0_dev[4];
  float* g1_dev[4];
  float* dLdG0_dev[4];
  float* dLdG1_dev[4];
  float* mG0_dev[4];
  float* vG0_dev[4];
  float* mG1_dev[4];
  float* vG1_dev[4];
  float* g0_noise_dev;
  float* g1_noise_dev;
  curandState* rstate_dev;
  for (int i = 0; i < 4; i++)
  {
    initialize_grid(gen, &g0_host[i], g0_grid_dim[i], g0_grid_dim[i], g0_dim);
    initialize_grid(gen, &g1_host[i], g1_grid_dim[i], g1_grid_dim[i], g1_dim);
    cudaMalloc(&g0_dev[i], g0_grid_dim[i] * g0_grid_dim[i] * g0_dim * sizeof(float));
    cudaMalloc(&g1_dev[i], g1_grid_dim[i] * g1_grid_dim[i] * g1_dim * sizeof(float));
    cudaMemcpy(g0_dev[i], g0_host[i], g0_grid_dim[i] * g0_grid_dim[i] * g0_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(g1_dev[i], g1_host[i], g1_grid_dim[i] * g1_grid_dim[i] * g1_dim * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&dLdG0_dev[i], g0_grid_dim[i] * g0_grid_dim[i] * g0_dim * sizeof(float));
    cudaMalloc(&dLdG1_dev[i], g1_grid_dim[i] * g1_grid_dim[i] * g1_dim * sizeof(float));
    cudaMalloc(&mG0_dev[i], g0_grid_dim[i] * g0_grid_dim[i] * g0_dim * sizeof(float));
    cudaMemset(mG0_dev[i], 0, g0_grid_dim[i] * g0_grid_dim[i] * g0_dim * sizeof(float));
    cudaMalloc(&vG0_dev[i], g0_grid_dim[i] * g0_grid_dim[i] * g0_dim * sizeof(float));
    cudaMemset(vG0_dev[i], 0, g0_grid_dim[i] * g0_grid_dim[i] * g0_dim * sizeof(float));
    cudaMalloc(&mG1_dev[i], g1_grid_dim[i] * g1_grid_dim[i] * g1_dim * sizeof(float));
    cudaMemset(mG1_dev[i], 0, g1_grid_dim[i] * g1_grid_dim[i] * g1_dim * sizeof(float));
    cudaMalloc(&vG1_dev[i], g1_grid_dim[i] * g1_grid_dim[i] * g1_dim * sizeof(float));
    cudaMemset(vG1_dev[i], 0, g1_grid_dim[i] * g1_grid_dim[i] * g1_dim * sizeof(float));
  }

  cudaMalloc(&g0_noise_dev, g0_grid_dim[0] * g0_grid_dim[0] * g0_dim * sizeof(float));
  cudaMalloc(&g1_noise_dev, g1_grid_dim[0] * g1_grid_dim[0] * g1_dim * sizeof(float));
  cudaMalloc(&rstate_dev, g0_grid_dim[0] * g0_grid_dim[0] * g0_dim * sizeof(curandState));
  launch_initialize_rand(g0_grid_dim[0] * g0_grid_dim[0] * g0_dim, rstate_dev);


  MipChain mips = load_mip_chain("C:/Code/openntc/data/mips/Bricks101_1K-JPG");

  float *W0_host, *W1_host, *Wout_host;
  initialize_decoder_weights(gen, feature_dim, out_dim, &W0_host, &W1_host, &Wout_host);
  float *x_dev;
  cudaMalloc(&x_dev, feature_dim * max_batch_dim * sizeof(float));
  float *W0_dev, *W1_dev, *Wout_dev;
  cudaMalloc(&W0_dev, 64 * feature_dim * sizeof(float));
  cudaMalloc(&W1_dev, 64 * 64 * sizeof(float));
  cudaMalloc(&Wout_dev, out_dim * 64 * sizeof(float));
  cudaMemcpy(W0_dev, W0_host, 64 * feature_dim * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(W1_dev, W1_host, 64 * 64 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(Wout_dev, Wout_host, out_dim * 64 * sizeof(float), cudaMemcpyHostToDevice);
  float *W0x_dev;
  cudaMalloc(&W0x_dev, 64 * max_batch_dim * sizeof(float));
  float *W0xa_dev;
  cudaMalloc(&W0xa_dev, 64 * max_batch_dim * sizeof(float));
  float *W1x_dev;
  cudaMalloc(&W1x_dev, 64 * max_batch_dim * sizeof(float));
  float *W1xa_dev;
  cudaMalloc(&W1xa_dev, 64 * max_batch_dim * sizeof(float));
  float *Woutx_dev;
  cudaMalloc(&Woutx_dev, out_dim * max_batch_dim * sizeof(float));
  float *lossdiff_dev;
  cudaMalloc(&lossdiff_dev, out_dim * max_batch_dim * sizeof(float));
  float* dLdPred_dev;
  cudaMalloc(&dLdPred_dev, out_dim * max_batch_dim * sizeof(float));
  float* dLdWout_dev;
  cudaMalloc(&dLdWout_dev, out_dim * 64 * sizeof(float));
  float* dLdW1xa_dev;
  cudaMalloc(&dLdW1xa_dev, 64 * max_batch_dim * sizeof(float));
  float* dLdW1x_dev;
  cudaMalloc(&dLdW1x_dev, 64 * max_batch_dim *sizeof(float));
  float* dLdW1_dev;
  cudaMalloc(&dLdW1_dev, 64 * 64 * sizeof(float));
  float* dLdW0xa_dev;
  cudaMalloc(&dLdW0xa_dev, 64 * max_batch_dim * sizeof(float));
  float* dLdW0x_dev;
  cudaMalloc(&dLdW0x_dev, 64 * max_batch_dim * sizeof(float));
  float* dLdW0_dev;
  cudaMalloc(&dLdW0_dev, 64 * feature_dim * sizeof(float));
  float* dLdx_dev;
  cudaMalloc(&dLdx_dev, feature_dim * max_batch_dim * sizeof(float));
  float* mWout_dev;
  cudaMalloc(&mWout_dev, out_dim * 64 * sizeof(float));
  cudaMemset(mWout_dev, 0, out_dim * 64 * sizeof(float));
  float* vWout_dev;
  cudaMalloc(&vWout_dev, out_dim * 64 * sizeof(float));
  cudaMemset(vWout_dev, 0, out_dim * 64 * sizeof(float));
  float* mW1_dev;
  cudaMalloc(&mW1_dev, 64 * 64 * sizeof(float));
  cudaMemset(mW1_dev, 0, 64 * 64 * sizeof(float));
  float* vW1_dev;
  cudaMalloc(&vW1_dev, 64 * 64 * sizeof(float));
  cudaMemset(vW1_dev, 0, 64 * 64 * sizeof(float));
  float* mW0_dev;
  cudaMalloc(&mW0_dev, 64 * feature_dim * sizeof(float));
  cudaMemset(mW0_dev, 0, 64 * feature_dim * sizeof(float));
  float* vW0_dev;
  cudaMalloc(&vW0_dev, 64 * feature_dim * sizeof(float));
  cudaMemset(vW0_dev, 0, 64 * feature_dim * sizeof(float));

  float* lanczos_host;
  initialize_lanczos_weights(2, 2, &lanczos_host);
  float* lanczos_dev;
  cudaMalloc(&lanczos_dev, 8 * sizeof(float));
  cudaMemcpy(lanczos_dev, lanczos_host, 8 * sizeof(float), cudaMemcpyHostToDevice);

  std::bernoulli_distribution dist_batch_type(0.05);
  std::uniform_real_distribution<float> dist_u(0.0f, 1.0f);
  std::uniform_int_distribution<int> dist_grid(0, 8);

  int *grid_draws_dev;
  cudaMalloc(&grid_draws_dev, sizeof(int) * 16);

  PerfTimer ptimer;

  int grid_batch_i[4] = {0, 0, 0, 0};

  int batch_count = 10000;
  // assert(PerfTimer::num_warmup + PerfTimer::len_window <= batch_count);
  int lock_i = 95 * batch_count / 100;
  assert(lock_i > 0);
  for (int batch_i = 0; batch_i < batch_count; batch_i++)
  {
    bool draw_uniform = dist_batch_type(gen);
    float U = dist_u(gen);
    int lod = int(floorf(0.5f * -log2f(U)));
    int lod_uniform = dist_grid(gen);
    if (draw_uniform)
      lod = lod_uniform;
    lod = std::clamp(lod, 0, 8);
    int feature_level = -1;
    if (lod <= 3)
      feature_level = 0;
    else if (lod <= 5)
      feature_level = 1;
    else if (lod <= 7)
      feature_level = 2;
    else
      feature_level = 3;
    int grid_draws[16];
    std::uniform_int_distribution<int> dist_grid(0, std::max(mip_dim[lod] - 256, 0));
    for (int i = 0; i < 8; i++)
    {
      grid_draws[i + i] = dist_grid(gen);
      grid_draws[i + i + 1] = dist_grid(gen);
    }
    cudaMemcpy(grid_draws_dev, grid_draws, sizeof(int) * 16, cudaMemcpyHostToDevice);
    int grid_dim_draw = std::min(mip_dim[lod], 256);
    if (batch_i < lock_i)
    {
      launch_generate_noise(g0_grid_dim[feature_level] * g0_grid_dim[feature_level] * g0_dim, g0_delta, rstate_dev, g0_noise_dev);
      launch_generate_noise(g1_grid_dim[feature_level] * g1_grid_dim[feature_level] * g1_dim, g1_delta, rstate_dev, g1_noise_dev);
    } 
    else if (batch_i == lock_i)
    {
      for (int level_i = 0; level_i < 4; level_i++)
      {
        launch_quantize_grid(
          g0_grid_dim[level_i] * g0_grid_dim[level_i] * g0_dim,
          g0_num_bytes,
          g0_delta,
          g0_dev[level_i]);
        launch_quantize_grid(
          g1_grid_dim[level_i] * g1_grid_dim[level_i] * g1_dim,
          g1_num_bytes,
          g1_delta,
          g1_dev[level_i]);
      }

      cudaMemset(g0_noise_dev, 0, sizeof(float) * g0_grid_dim[0] * g0_grid_dim[0] * g0_dim);
      cudaMemset(g1_noise_dev, 0, sizeof(float) * g1_grid_dim[0] * g1_grid_dim[0] * g1_dim);
    }

    ptimer.log(batch_i, 0, lod);
    launch_draw_features(
      8,
      grid_dim_draw,
      feature_dim,
      mip_dim[lod],
      g0_grid_dim[feature_level],
      g1_grid_dim[feature_level],
      g0_dim,
      g1_dim,
      lod / 8.0f,
      grid_draws_dev,
      g0_noise_dev,
      g1_noise_dev,
      g0_dev[feature_level],
      g1_dev[feature_level],
      x_dev);
    ptimer.log(batch_i, 1, lod);
    launch_draw_targets(8, grid_dim_draw, mip_dim[lod], out_dim, grid_draws_dev, mips.levels[lod].data_dev, lossdiff_dev);
    ptimer.log(batch_i, 2, lod);

    int batch_dim = 8 * grid_dim_draw * grid_dim_draw;

    // Forward pass
    launch_forward_pass(batch_dim, x_dev, W0_dev, W1_dev, Wout_dev, W0x_dev, W0xa_dev, W1x_dev, W1xa_dev, Woutx_dev);
    ptimer.log(batch_i, 3, lod);

    // Loss

    float cublas_neg1 = -1.0f;
    cublasSaxpy(handle, out_dim * batch_dim, &cublas_neg1, Woutx_dev, 1, lossdiff_dev, 1);
    float squarederr = 0.0f;
    cublasSdot(handle, out_dim * batch_dim, lossdiff_dev, 1, lossdiff_dev, 1, &squarederr);
    float mse = squarederr / ((float)out_dim * batch_dim);
    // std::printf("Batch %05d: %f\n", batch_i, mse);
    launch_scalar_product(out_dim * batch_dim, -2.0f / ((float)out_dim * batch_dim), lossdiff_dev, dLdPred_dev);
    ptimer.log(batch_i, 4, lod);

    // Backward pass

#if 0
    matmulABT(handle, out_dim, 64, batch_dim, dLdPred_dev, W1xa_dev, dLdWout_dev);
    matmulATB(handle, 64, batch_dim, out_dim, Wout_dev, dLdPred_dev, dLdW1xa_dev);
    launch_backward_hardgelu(64 * batch_dim, W1x_dev, dLdW1xa_dev, dLdW1x_dev);
    matmulABT(handle, 64, 64, batch_dim, dLdW1x_dev, W0xa_dev, dLdW1_dev);
    matmulATB(handle, 64, batch_dim, 64, W1_dev, dLdW1x_dev, dLdW0xa_dev);
    launch_backward_hardgelu(64 * batch_dim, W0x_dev, dLdW0xa_dev, dLdW0x_dev);
    matmulABT(handle, 64, feature_dim, batch_dim, dLdW0x_dev, x_dev, dLdW0_dev);
    matmulATB(handle, feature_dim, batch_dim, 64, W0_dev, dLdW0x_dev, dLdx_dev);
#endif
    launch_backward_pass(batch_dim, W0_dev, W1_dev, Wout_dev, dLdPred_dev, W0x_dev, W1x_dev, dLdx_dev, dLdW0x_dev, dLdW1x_dev);
    matmulABT(handle, out_dim, 64, batch_dim, dLdPred_dev, W1xa_dev, dLdWout_dev);
    matmulABT(handle, 64, 64, batch_dim, dLdW1x_dev, W0xa_dev, dLdW1_dev);
    matmulABT(handle, 64, feature_dim, batch_dim, dLdW0x_dev, x_dev, dLdW0_dev);
    ptimer.log(batch_i, 5, lod);

    cudaMemset(dLdG0_dev[feature_level], 0, g0_grid_dim[feature_level] * g0_grid_dim[feature_level] * g0_dim * sizeof(float));
    cudaMemset(dLdG1_dev[feature_level], 0, g1_grid_dim[feature_level] * g1_grid_dim[feature_level] * g1_dim * sizeof(float));
    launch_accumulate_grid_gradients(
      8,
      grid_dim_draw,
      feature_dim,
      mip_dim[lod],
      g0_grid_dim[feature_level],
      g1_grid_dim[feature_level],
      g0_dim,
      g1_dim,
      grid_draws_dev,
      dLdx_dev,
      dLdG0_dev[feature_level],
      dLdG1_dev[feature_level]);
    ptimer.log(batch_i, 6, lod);

    // Update ADAM parameters

    float beta_1 = 0.9f;
    float beta_2 = 0.999f;
    float bias_1 = 1.0f / (1.0f - powf(beta_1, batch_i + 1.0f));
    float bias_2 = 1.0f / (1.0f - powf(beta_2, batch_i + 1.0f));
    float grid_bias_1 = 1.0f / (1.0f - powf(beta_1, grid_batch_i[feature_level] + 1.0f));
    float grid_bias_2 = 1.0f / (1.0f - powf(beta_2, grid_batch_i[feature_level] + 1.0f));
    grid_batch_i[feature_level] += 1;

    float lr_grid = cosine_annealing(0.0f, 0.01f, batch_count, batch_i);
    float lr_decoder = cosine_annealing(0.0f, 0.005f, batch_count, batch_i);

    launch_update_adam(out_dim * 64, lr_decoder, beta_1, beta_2, bias_1, bias_2, dLdWout_dev, mWout_dev, vWout_dev, Wout_dev);
    launch_update_adam(64 * 64, lr_decoder, beta_1, beta_2, bias_1, bias_2, dLdW1_dev, mW1_dev, vW1_dev, W1_dev);
    launch_update_adam(64 * feature_dim, lr_decoder, beta_1, beta_2, bias_1, bias_2, dLdW0_dev, mW0_dev, vW0_dev, W0_dev);
    if (batch_i < lock_i)
    {
      launch_update_adam(
        g0_grid_dim[feature_level] * g0_grid_dim[feature_level] * g0_dim,
        lr_grid,
        beta_1,
        beta_2,
        grid_bias_1,
        grid_bias_2,
        dLdG0_dev[feature_level],
        mG0_dev[feature_level],
        vG0_dev[feature_level],
        g0_dev[feature_level]);
      launch_update_adam(
        g1_grid_dim[feature_level] * g1_grid_dim[feature_level] * g1_dim,
        lr_grid,
        beta_1,
        beta_2,
        grid_bias_1,
        grid_bias_2,
        dLdG1_dev[feature_level],
        mG1_dev[feature_level],
        vG1_dev[feature_level],
        g1_dev[feature_level]);

      launch_clamp_grid(
        g0_grid_dim[feature_level] * g0_grid_dim[feature_level] * g0_dim,
        g0_num_bytes,
        g0_delta,
        g0_dev[feature_level]);
      launch_clamp_grid(
        g1_grid_dim[feature_level] * g1_grid_dim[feature_level] * g1_dim,
        g1_num_bytes,
        g1_delta,
        g1_dev[feature_level]);
    }
    ptimer.log(batch_i, 7, lod);
  }

  ptimer.harvest();

  /*double mse_numer = 0;
  double mse_denom = 0;
  for (int mip_i = 0; mip_i <= 8; mip_i++)
  {
    int grids_per_dim = (mip_dim[mip_i] + 255) / 256;
    int num_grids = grids_per_dim * grids_per_dim;
    int grid_draws[16];
    int grid_dim_draw = std::min(256, mip_dim[mip_i]);
    for (int batch_start = 0; batch_start < num_grids; batch_start += 8)
    {
      int num_batches = std::min(8, num_grids - batch_start);
      int fill_i = 0;
      for (int g = batch_start; g < batch_start + num_batches; g++)
      {
        int gx = (g % grids_per_dim) * grid_dim_draw;
        int gy = (g / grids_per_dim) * grid_dim_draw;
        grid_draws[fill_i + fill_i] = gx;
        grid_draws[fill_i + fill_i + 1] = gy;
        fill_i++;
      }

      cudaMemcpy(grid_draws_dev, grid_draws, sizeof(int) * 2 * num_batches, cudaMemcpyHostToDevice);

      int feature_level = -1;
      if (mip_i <= 3)
        feature_level = 0;
      else if (mip_i <= 5)
        feature_level = 1;
      else if (mip_i <= 7)
        feature_level = 2;
      else
        feature_level = 3;

      int batch_dim = num_batches * grid_dim_draw * grid_dim_draw;

      launch_draw_features(
        num_batches,
        grid_dim_draw,
        feature_dim,
        mip_dim[mip_i],
        g0_grid_dim[feature_level],
        g1_grid_dim[feature_level],
        g0_dim,
        g1_dim,
        mip_i / 8.0f,
        grid_draws_dev,
        g0_noise_dev,
        g1_noise_dev,
        g0_dev[feature_level],
        g1_dev[feature_level],
        x_dev);
      launch_draw_targets(num_batches, grid_dim_draw, mip_dim[mip_i], out_dim, grid_draws_dev, mips.levels[mip_i].data_dev, lossdiff_dev);
      
      // Forward pass

      matmulAB(handle, 64, batch_dim, feature_dim, W0_dev, x_dev, W0x_dev);
      launch_forward_hardgelu(64 * batch_dim, W0x_dev, W0xa_dev);
      matmulAB(handle, 64, batch_dim, 64, W1_dev, W0xa_dev, W1x_dev);
      launch_forward_hardgelu(64 * batch_dim, W1x_dev, W1xa_dev);
      matmulAB(handle, out_dim, batch_dim, 64, Wout_dev, W1xa_dev, Woutx_dev);

      float cublas_neg1 = -1.0f;
      cublasSaxpy(handle, out_dim * batch_dim, &cublas_neg1, Woutx_dev, 1, lossdiff_dev, 1);
      float squarederr = 0.0f;
      cublasSdot(handle, out_dim * batch_dim, lossdiff_dev, 1, lossdiff_dev, 1, &squarederr);
      mse_numer += squarederr;
      mse_denom += batch_dim * out_dim;
    }
  }
  double mse = mse_numer / mse_denom;
  printf("PSNR: %f dB", -10.0 * log10(mse));*/

  cudaDeviceSynchronize();
  return 0;
}