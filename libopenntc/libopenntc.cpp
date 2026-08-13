#include <libopenntc/libopenntc.h>

#include <cassert>
#include <bit>
#include <fstream>

#include <curand_kernel.h>

#define STB_IMAGE_IMPLEMENTATION
#include <libopenntc/stb_image.h>
#include <libopenntc/ntc_kernel.cuh>
#include <libopenntc/json.hpp>

Tensor2d::Tensor2d() : initialized_(false), dev_(nullptr) {}

Tensor2d::~Tensor2d()
{
  Destroy();
}

OpenNTCResult Tensor2d::Init(int x, int y)
{
  shape_[0] = x;
  shape_[1] = y;
  cudaMalloc(&dev_, sizeof(float) * x * y);
  if (dev_ == nullptr)
    return OpenNTCResult::AllocationFailure;
  initialized_ = true;
  return OpenNTCResult::Success;
}

OpenNTCResult Tensor2d::InitLike(const Tensor2d& t0)
{
  return Init(t0.shape_[0], t0.shape_[1]);
}

void Tensor2d::FillZero()
{
  cudaMemset(dev_, 0, sizeof(float) * shape_[0] * shape_[1]);
}

void Tensor2d::FillKaiming(std::mt19937& gen)
{
  float* tmp = new float[shape_[0] * shape_[1]];

  // Fan in bounds +-1/sqrt(in_dim)
  float bounds = 1.0f / std::sqrtf((float)shape_[1]);
  std::uniform_real_distribution<float> dist(-bounds, bounds);

  for (int i = 0; i < shape_[0]; i++)
  {
    for (int j = 0; j < shape_[1]; j++)
    {
      tmp[i * shape_[1] + j] = dist(gen);
    }
  }

  cudaMemcpy(dev_, tmp, sizeof(float) * shape_[0] * shape_[1], cudaMemcpyHostToDevice);
  delete[] tmp;
}

void Tensor2d::Destroy()
{
  if (initialized_)
  {
    cudaFree(dev_);
    initialized_ = false;
  }
}

float* Tensor2d::DevicePtr()
{
  assert(initialized_);
  return dev_;
}

size_t Tensor2d::SizeBytes() const
{
  return sizeof(float) * shape_[0] * shape_[1];
}

size_t Tensor2d::NumElems() const
{
  return (size_t)shape_[0] * shape_[1];
}

Tensor3d::Tensor3d() : initialized_(false), dev_(nullptr) {}

Tensor3d::~Tensor3d()
{
  Destroy();
}

OpenNTCResult Tensor3d::Init(int x, int y, int z)
{
  shape_[0] = x;
  shape_[1] = y;
  shape_[2] = z;
  cudaMalloc(&dev_, sizeof(float) * x * y * z);
  if (dev_ == nullptr)
    return OpenNTCResult::AllocationFailure;
  initialized_ = true;
  return OpenNTCResult::Success;
}

OpenNTCResult Tensor3d::InitLike(const Tensor3d& t0)
{
  return Init(t0.shape_[0], t0.shape_[1], t0.shape_[2]);
}

void Tensor3d::FillZero()
{
  cudaMemset(dev_, 0, sizeof(float) * shape_[0] * shape_[1] * shape_[2]);
}

void Tensor3d::FillUniform(std::mt19937& gen, float lb, float ub)
{
  assert(lb < ub);
  float* tmp = new float[shape_[0] * shape_[1] * shape_[2]];

  std::uniform_real_distribution<float> dist(lb, ub);

  for (int i = 0; i < shape_[0]; i++)
  {
    for (int j = 0; j < shape_[1]; j++)
    {
      for (int k = 0; k < shape_[2]; k++)
      {
        tmp[i * shape_[1] * shape_[2] + j * shape_[2] + k] = dist(gen);
      }
    }
  }

  cudaMemcpy(dev_, tmp, sizeof(float) * shape_[0] * shape_[1] * shape_[2], cudaMemcpyHostToDevice);
  delete[] tmp;
}

void Tensor3d::Destroy()
{
  if (initialized_)
  {
    cudaFree(dev_);
    initialized_ = false;
  }
}

float* Tensor3d::DevicePtr()
{
  assert(initialized_);
  return dev_;
}

float** Tensor3d::DeviceDPtr()
{
  assert(!initialized_);
  initialized_ = true;
  return &dev_;
}

bool Tensor3d::IsInitialized() const
{
  return initialized_;
}

size_t Tensor3d::SizeBytes() const
{
  return sizeof(float) * shape_[0] * shape_[1] * shape_[2];
}

size_t Tensor3d::NumElems() const
{
  return (size_t)shape_[0] * shape_[1] * shape_[2];
}

IntTensor1d::IntTensor1d() : initialized_(false), dev_(nullptr) {}

IntTensor1d::~IntTensor1d()
{
  Destroy();
}

OpenNTCResult IntTensor1d::Init(int x)
{
  shape_[0] = x;
  cudaMalloc(&dev_, sizeof(int) * x);
  if (dev_ == nullptr)
    return OpenNTCResult::AllocationFailure;
  initialized_ = true;
  return OpenNTCResult::Success;
}

void IntTensor1d::Destroy()
{
  if (initialized_)
  {
    cudaFree(dev_);
    initialized_ = false;
  }
}

int* IntTensor1d::DevicePtr()
{
  assert(initialized_);
  return dev_;
}

OpenNTCContext::OpenNTCContext() : initialized_(false), gen_(123), rstate_(nullptr) {}
OpenNTCContext::~OpenNTCContext() { Destroy(); }

static int RoundUpToNearestK(int n, int k)
{
  assert(k > 0);
  return ((n + k - 1) / k) * k;
}

OpenNTCResult OpenNTCContext::Init(const OpenNTCContextInitInfo& init_info)
{
  assert(init_info.profile == OpenNTCProfile::Bpp_0_2);

  if (init_info.dim < OpenNTCContext::kMinDimension || init_info.dim > OpenNTCContext::kMaxDimension || std::popcount((unsigned int)init_info.dim) != 1)
  {
    return OpenNTCResult::InvalidDimension;
  }

  // Profile constants

  g0_bytes_per_channel_ = 2;
  g1_bytes_per_channel_ = 4;
  g0_delta_ = 2.0f / powf(2.0f, (float) g0_bytes_per_channel_);
  g1_delta_ = 2.0f / powf(2.0f, (float) g1_bytes_per_channel_);
  g0_channels_ = 8;
  g1_channels_ = 12;
  mip_dim_[0] = init_info.dim;
  mip_count_ = 0;
  while (mip_dim_[mip_count_] > 4)
  {
    mip_dim_[mip_count_ + 1] = mip_dim_[mip_count_] / 2;
    mip_count_++;
  }
  mip_count_++;
  g0_grid_dim_[0] = 256;
  g0_grid_dim_[1] = 64;
  g0_grid_dim_[2] = 16;
  g0_grid_dim_[3] = 4;
  g1_grid_dim_[0] = 128;
  g1_grid_dim_[1] = 32;
  g1_grid_dim_[2] = 8;
  g1_grid_dim_[3] = 2;
  feature_dim_ = RoundUpToNearestK(4 * g0_channels_ + g1_channels_ + 12 + 1, 4);
  feature_dim_padded_ = RoundUpToNearestK(feature_dim_, 16);
  out_dim_ = 9;
  out_dim_padded_ = RoundUpToNearestK(out_dim_, 4);
  max_batch_ = 8;
  max_batch_dim_ = max_batch_ * 256 * 256;
  hidden_dim_ = 64;

  for (int i = 0; i < 4; i++)
  {
    g0_[i].Init(g0_grid_dim_[i], g0_grid_dim_[i], g0_channels_);
    g1_[i].Init(g1_grid_dim_[i], g1_grid_dim_[i], g1_channels_);
  }

  g0_noise_.Init(g0_grid_dim_[0], g0_grid_dim_[0], g0_channels_);
  g1_noise_.Init(g1_grid_dim_[0], g1_grid_dim_[0], g1_channels_);

  W0_.Init(hidden_dim_, feature_dim_);
  W1_.Init(hidden_dim_, hidden_dim_);
  Wout_.Init(out_dim_, hidden_dim_);

  x_.Init(feature_dim_, max_batch_dim_);
  W0x_.Init(hidden_dim_, max_batch_dim_);
  W0xa_.Init(hidden_dim_, max_batch_dim_);
  W1x_.Init(hidden_dim_, max_batch_dim_);
  W1xa_.Init(hidden_dim_, max_batch_dim_);
  Woutx_.Init(out_dim_, max_batch_dim_);

  dLdWoutx_.InitLike(Woutx_);
  dLdWout_.InitLike(Wout_);
  dLdW1xa_.InitLike(W1xa_);
  dLdW1x_.InitLike(W1x_);
  dLdW1_.InitLike(W1_);
  dLdW0xa_.InitLike(W0xa_);
  dLdW0x_.InitLike(W0x_);
  dLdW0_.InitLike(W0_);
  dLdx_.InitLike(x_);
  for (int level_i = 0; level_i < 4; level_i++)
  {
    dLdG0_[level_i].InitLike(g0_[level_i]);
    dLdG1_[level_i].InitLike(g1_[level_i]);
  }

  mse_.Init(out_dim_, max_batch_dim_);

  for (int level_i = 0; level_i < 4; level_i++)
  {
    mG0_[level_i].InitLike(g0_[level_i]);
    vG0_[level_i].InitLike(g0_[level_i]);
    mG1_[level_i].InitLike(g1_[level_i]);
    vG1_[level_i].InitLike(g1_[level_i]);
  }
  mW0_.InitLike(W0_);
  vW0_.InitLike(W0_);
  mW1_.InitLike(W1_);
  vW1_.InitLike(W1_);
  mWout_.InitLike(Wout_);
  vWout_.InitLike(Wout_);

  grid_draws_.Init(2 * max_batch_);

  cublasCreate(&handle_);
  cublasSetMathMode(handle_, CUBLAS_TF32_TENSOR_OP_MATH);
  cudaMalloc(&rstate_, sizeof(curandState) * g0_grid_dim_[0] * g0_grid_dim_[0] * g0_channels_);
  launch_initialize_rand(g0_grid_dim_[0] * g0_grid_dim_[0] * g0_channels_, rstate_);

  for (int i = 0; i < 4; i++)
  {
    g0_host_[i] = new uint32_t[(g0_[i].NumElems() * g0_bytes_per_channel_) / 32];
    g1_host_[i] = new uint32_t[(g1_[i].NumElems() * g1_bytes_per_channel_) / 32];
  }
  W0_host_ = new uint32_t[(hidden_dim_ * feature_dim_padded_) / 4];
  W1_host_ = new uint32_t[W1_.NumElems() / 4];
  Wout_host_ = new uint32_t[(out_dim_padded_ * hidden_dim_) / 4];
  W0_scale_ = new float[hidden_dim_];
  W1_scale_ = new float[hidden_dim_];
  Wout_scale_ = new float[out_dim_padded_];

  for (int i = 0; i < kMaxSources; i++)
  {
    for (int j = 0; j < OpenNTCContext::kMaxMips; j++)
      mips_host_[i][j] = new float[mip_dim_[j] * mip_dim_[j] * 4];
  }

  for (int i = 0; i < kMaxSources; i++)
  {
    for (int j = 0; j < OpenNTCContext::kMaxMips; j++)
    {
      mips_[i][j].Init(mip_dim_[j], mip_dim_[j], 4);
    }
  }
  tex_prep_.Init(mip_dim_[0], mip_dim_[0], 4);
  tex_filter_.Init(mip_dim_[0], mip_dim_[0], 4);
  for (int i = 0; i < mip_count_; i++)
  {
    package_[i].Init(mip_dim_[i], mip_dim_[i], out_dim_);
  }

  return OpenNTCResult::Success;
}

// C = A * B, where A, B, C are row-major. C is n x m, A is n x k, B is k x m

static void matmulAB(cublasHandle_t handle, int n, int m, int k, float* A, float* B, float* C)
{
  float sgemm_alpha = 1.0f, sgemm_beta = 0.0f;
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &sgemm_alpha, B, m, A, k, &sgemm_beta, C, m);
}

// C = A^T * B, where A, B, C are row-major. C is n x m, A is k x n, B is k x m

static void matmulATB(cublasHandle_t handle, int n, int m, int k, float* A, float* B, float* C)
{
  float sgemm_alpha = 1.0f, sgemm_beta = 0.0f;
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &sgemm_alpha, B, m, A, n, &sgemm_beta, C, m);
}

// C = A * B^T, where A, B, C are row-major. C is n x m, A is n x k, B is m x k

static void matmulABT(cublasHandle_t handle, int n, int m, int k, float* A, float* B, float* C)
{
  float sgemm_alpha = 1.0f, sgemm_beta = 0.0f;
  cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &sgemm_alpha, B, k, A, k, &sgemm_beta, C, m);
}

static float cosine_annealing(float lr_min, float lr_max, int t_max, int t_cur)
{
  float lr = lr_min + 0.5f * (lr_max - lr_min) * (1.0f + cosf(t_cur * acosf(-1.0f) / t_max));
  return lr;
}

static void QuantizeWeights(float* Wf, int rows, int cols, float prescale, uint32_t* o_Wq, float* o_Ws)
{
  for (int i = 0; i < rows; i++)
  {
    float max_abs = 0.0f;
    for (int j = 0; j < cols; j++)
    {
      float val = Wf[i * cols + j];
      max_abs = std::max(std::abs(val), max_abs);
    }
    float scale = (max_abs == 0.0f) ? 1.0f : max_abs / 127.0f;
    o_Ws[i] = scale * prescale;
    for (int j = 0; j < cols; j += 4)
    {
      uint32_t pack = 0;
      for (int c = 0; c < 4; c++)
      {
        float val = Wf[i * cols + j + c] / scale;
        int clamp_val = std::clamp((int)std::lroundf(val), -128, 127);
        pack |= ((clamp_val & 0xFF) << (c * 8));
      }
      o_Wq[i * (cols / 4) + (j / 4)] = pack;
    }
  }
}

void OpenNTCContext::BeginTraining(const OpenNTCTrainInfo& train_info)
{
  assert(train_phase_ == OpenNTCTrainPhase::ManifestLoaded || train_phase_ == OpenNTCTrainPhase::TrainComplete);
  batch_count_ = train_info.batch_count_;
  lock_i_ = (95 * batch_count_) / 100;
  batch_i_ = 0;
  grids_per_batch_ = train_info.grids_per_batch_;
  for (int i = 0; i < 4; i++)
  {
    grid_batch_i_[i] = 0;
  }

  for (int level_i = 0; level_i < 4; level_i++)
  {
    g0_[level_i].FillUniform(gen_, -0.1f, 0.1f);
    g1_[level_i].FillUniform(gen_, -0.1f, 0.1f);
    mG0_[level_i].FillZero();
    vG0_[level_i].FillZero();
    mG1_[level_i].FillZero();
    vG1_[level_i].FillZero();
  }

  W0_.FillKaiming(gen_);
  mW0_.FillZero();
  vW0_.FillZero();
  W1_.FillKaiming(gen_);
  mW1_.FillZero();
  vW1_.FillZero();
  Wout_.FillKaiming(gen_);
  mWout_.FillZero();
  vWout_.FillZero();

  train_phase_ = OpenNTCTrainPhase::TrainInProgress;
}

OpenNTCTrainProgress OpenNTCContext::TrainUntilComplete()
{
  assert(train_phase_ == OpenNTCTrainPhase::TrainInProgress || train_phase_ == OpenNTCTrainPhase::TrainComplete);

  return Train(batch_count_);
}

OpenNTCTrainProgress OpenNTCContext::Train(int32_t batch_count)
{
  assert(batch_count > 0);
  int batches_remaining = std::max(0, batch_count_ - batch_i_);
  batch_count = std::min(batches_remaining, batch_count);
  int batch_target = batch_i_ + batch_count;
  if (batch_i_ == batch_target)
  {
    OpenNTCTrainProgress tprogress = {};
    tprogress.phase_ = OpenNTCTrainPhase::TrainComplete;
    tprogress.result_ = OpenNTCResult::Success;
    tprogress.batches_complete_ = batch_i_;
    tprogress.total_batches_ = batch_count_;
    return tprogress;
  }

  std::bernoulli_distribution dist_batch_type(0.05);
  std::uniform_real_distribution<float> dist_u(0.0f, 1.0f);
  std::uniform_int_distribution<int> dist_lod(0, 8);

  for (batch_i_; batch_i_ < batch_target; batch_i_++)
  {
    bool draw_uniform = dist_batch_type(gen_);
    float U = dist_u(gen_);
    int lod = int(std::floorf(0.5f * -log2f(U)));
    int lod_uniform = dist_lod(gen_);
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
    std::uniform_int_distribution<int> dist_grid(0, std::max(mip_dim_[lod] - 256, 0));
    for (int i = 0; i < grids_per_batch_; i++)
    {
      grid_draws[i + i] = dist_grid(gen_);
      grid_draws[i + i + 1] = dist_grid(gen_);
    }
    cudaMemcpy(grid_draws_.DevicePtr(), grid_draws, sizeof(int) * grids_per_batch_ * 2, cudaMemcpyHostToDevice);
    int grid_dim_draw = std::min(mip_dim_[lod], 256);
    if (batch_i_ < lock_i_)
    {
      launch_generate_noise(g0_grid_dim_[feature_level] * g0_grid_dim_[feature_level] * g0_channels_, g0_delta_, rstate_, g0_noise_.DevicePtr());
      launch_generate_noise(g1_grid_dim_[feature_level] * g1_grid_dim_[feature_level] * g1_channels_, g1_delta_, rstate_, g1_noise_.DevicePtr());
    }
    else if (batch_i_ == lock_i_)
    {
      for (int level_i = 0; level_i < 4; level_i++)
      {
        launch_quantize_grid(
          g0_grid_dim_[level_i] * g0_grid_dim_[level_i] * g0_channels_,
          g0_bytes_per_channel_,
          g0_delta_,
          g0_[level_i].DevicePtr());
        launch_quantize_grid(
          g1_grid_dim_[level_i] * g1_grid_dim_[level_i] * g1_channels_,
          g1_bytes_per_channel_,
          g1_delta_,
          g1_[level_i].DevicePtr());
      }

      g0_noise_.FillZero();
      g1_noise_.FillZero();
    }

    launch_draw_features(
      grids_per_batch_,
      grid_dim_draw,
      feature_dim_,
      mip_dim_[lod],
      g0_grid_dim_[feature_level],
      g1_grid_dim_[feature_level],
      g0_channels_,
      g1_channels_,
      lod / 8.0f,
      grid_draws_.DevicePtr(),
      g0_noise_.DevicePtr(),
      g1_noise_.DevicePtr(),
      g0_[feature_level].DevicePtr(),
      g1_[feature_level].DevicePtr(),
      x_.DevicePtr());

    launch_draw_targets(
      grids_per_batch_,
      grid_dim_draw,
      mip_dim_[lod],
      out_dim_,
      grid_draws_.DevicePtr(),
      package_[lod].DevicePtr(),
      mse_.DevicePtr());

    int batch_dim = grids_per_batch_ * grid_dim_draw * grid_dim_draw;

    // Forward pass

    matmulAB(handle_, hidden_dim_, batch_dim, feature_dim_, W0_.DevicePtr(), x_.DevicePtr(), W0x_.DevicePtr());
    launch_forward_hardgelu(hidden_dim_ * batch_dim, W0x_.DevicePtr(), W0xa_.DevicePtr());
    matmulAB(handle_, hidden_dim_, batch_dim, hidden_dim_, W1_.DevicePtr(), W0xa_.DevicePtr(), W1x_.DevicePtr());
    launch_forward_hardgelu(hidden_dim_ * batch_dim, W1x_.DevicePtr(), W1xa_.DevicePtr());
    matmulAB(handle_, out_dim_, batch_dim, hidden_dim_, Wout_.DevicePtr(), W1xa_.DevicePtr(), Woutx_.DevicePtr());

    // Loss

    float host_neg1 = -1.0f;
    float total_squared_error = 0.0f;
    cublasSaxpy(handle_, out_dim_ * batch_dim, &host_neg1, Woutx_.DevicePtr(), 1, mse_.DevicePtr(), 1);
    cublasSdot(handle_, out_dim_ * batch_dim, mse_.DevicePtr(), 1, mse_.DevicePtr(), 1, &total_squared_error);
    float mse = total_squared_error / ((float) out_dim_ * batch_dim);
    launch_scalar_product(out_dim_ * batch_dim, -2.0f / ((float)out_dim_ * batch_dim), mse_.DevicePtr(), dLdWoutx_.DevicePtr());

    // Backward pass

    matmulABT(handle_, out_dim_, hidden_dim_, batch_dim, dLdWoutx_.DevicePtr(), W1xa_.DevicePtr(), dLdWout_.DevicePtr());
    matmulATB(handle_, hidden_dim_, batch_dim, out_dim_, Wout_.DevicePtr(), dLdWoutx_.DevicePtr(), dLdW1xa_.DevicePtr());
    launch_backward_hardgelu(hidden_dim_ * batch_dim, W1x_.DevicePtr(), dLdW1xa_.DevicePtr(), dLdW1x_.DevicePtr());
    matmulABT(handle_, hidden_dim_, hidden_dim_, batch_dim, dLdW1x_.DevicePtr(), W0xa_.DevicePtr(), dLdW1_.DevicePtr());
    matmulATB(handle_, hidden_dim_, batch_dim, hidden_dim_, W1_.DevicePtr(), dLdW1x_.DevicePtr(), dLdW0xa_.DevicePtr());
    launch_backward_hardgelu(hidden_dim_ * batch_dim, W0x_.DevicePtr(), dLdW0xa_.DevicePtr(), dLdW0x_.DevicePtr());
    matmulABT(handle_, hidden_dim_, feature_dim_, batch_dim, dLdW0x_.DevicePtr(), x_.DevicePtr(), dLdW0_.DevicePtr());
    matmulATB(handle_, feature_dim_, batch_dim, hidden_dim_, W0_.DevicePtr(), dLdW0x_.DevicePtr(), dLdx_.DevicePtr());

    dLdG0_[feature_level].FillZero();
    dLdG1_[feature_level].FillZero();
    launch_accumulate_grid_gradients(
      grids_per_batch_,
      grid_dim_draw,
      feature_dim_,
      mip_dim_[lod],
      g0_grid_dim_[feature_level],
      g1_grid_dim_[feature_level],
      g0_channels_,
      g1_channels_,
      grid_draws_.DevicePtr(),
      dLdx_.DevicePtr(),
      dLdG0_[feature_level].DevicePtr(),
      dLdG1_[feature_level].DevicePtr());

    float beta_1 = 0.9f;
    float beta_2 = 0.999f;
    float bias_1 = 1.0f / (1.0f - powf(beta_1, batch_i_ + 1.0f));
    float bias_2 = 1.0f / (1.0f - powf(beta_2, batch_i_ + 1.0f));
    float grid_bias_1 = 1.0f / (1.0f - powf(beta_1, grid_batch_i_[feature_level] + 1.0f));
    float grid_bias_2 = 1.0f / (1.0f - powf(beta_2, grid_batch_i_[feature_level] + 1.0f));
    grid_batch_i_[feature_level] += 1;

    float lr_grid = cosine_annealing(0.0f, 0.01f, batch_count_, batch_i_);
    float lr_decoder = cosine_annealing(0.0f, 0.005f, batch_count_, batch_i_);

    launch_update_adam(out_dim_ * hidden_dim_, lr_decoder, beta_1, beta_2, bias_1, bias_2, dLdWout_.DevicePtr(), mWout_.DevicePtr(), vWout_.DevicePtr(), Wout_.DevicePtr());
    launch_update_adam(hidden_dim_ * hidden_dim_, lr_decoder, beta_1, beta_2, bias_1, bias_2, dLdW1_.DevicePtr(), mW1_.DevicePtr(), vW1_.DevicePtr(), W1_.DevicePtr());
    launch_update_adam(hidden_dim_ * feature_dim_, lr_decoder, beta_1, beta_2, bias_1, bias_2, dLdW0_.DevicePtr(), mW0_.DevicePtr(), vW0_.DevicePtr(), W0_.DevicePtr());
    if (batch_i_ < lock_i_)
    {
      launch_update_adam(
        g0_grid_dim_[feature_level] * g0_grid_dim_[feature_level] * g0_channels_,
        lr_grid,
        beta_1,
        beta_2,
        grid_bias_1,
        grid_bias_2,
        dLdG0_[feature_level].DevicePtr(),
        mG0_[feature_level].DevicePtr(),
        vG0_[feature_level].DevicePtr(),
        g0_[feature_level].DevicePtr());
      launch_update_adam(
        g1_grid_dim_[feature_level] * g1_grid_dim_[feature_level] * g1_channels_,
        lr_grid,
        beta_1,
        beta_2,
        grid_bias_1,
        grid_bias_2,
        dLdG1_[feature_level].DevicePtr(),
        mG1_[feature_level].DevicePtr(),
        vG1_[feature_level].DevicePtr(),
        g1_[feature_level].DevicePtr());

      launch_clamp_grid(
        g0_grid_dim_[feature_level] * g0_grid_dim_[feature_level] * g0_channels_,
        g0_bytes_per_channel_,
        g0_delta_,
        g0_[feature_level].DevicePtr());
      launch_clamp_grid(
        g1_grid_dim_[feature_level] * g1_grid_dim_[feature_level] * g1_channels_,
        g1_bytes_per_channel_,
        g1_delta_,
        g1_[feature_level].DevicePtr());
    }
  }

  OpenNTCTrainProgress tprogress = {};
  tprogress.phase_ = OpenNTCTrainPhase::TrainInProgress;
  tprogress.result_ = OpenNTCResult::Success;
  tprogress.batches_complete_ = batch_i_;
  tprogress.total_batches_ = batch_count_;

  if (batch_i_ == batch_count_)
  {
    uint32_t* g0pack = nullptr;
    uint32_t* g1pack = nullptr;

    cudaMalloc(&g0pack, sizeof(uint32_t) * (g0_[0].NumElems() * g0_bytes_per_channel_) / 32);
    cudaMalloc(&g1pack, sizeof(uint32_t) * (g1_[0].NumElems() * g1_bytes_per_channel_) / 32);

    for (int i = 0; i < 4; i++)
    {
      launch_quantize_pack(g0_[i].NumElems(), (g0_[i].NumElems() * g0_bytes_per_channel_) / 32, g0_bytes_per_channel_, g0_[i].DevicePtr(), g0pack);
      launch_quantize_pack(g1_[i].NumElems(), (g1_[i].NumElems() * g1_bytes_per_channel_) / 32, g1_bytes_per_channel_, g1_[i].DevicePtr(), g1pack);
      cudaMemcpy(g0_host_[i], g0pack, sizeof(uint32_t) * (g0_[i].NumElems() * g0_bytes_per_channel_) / 32, cudaMemcpyDeviceToHost);
      cudaMemcpy(g1_host_[i], g1pack, sizeof(uint32_t) * (g1_[i].NumElems() * g1_bytes_per_channel_) / 32, cudaMemcpyDeviceToHost);
    }

    cudaFree(g0pack);
    cudaFree(g1pack);

    caldata_ = Calibrate();

    float* W0_unpack_unpadded = new float[hidden_dim_ * feature_dim_];
    float* W0_unpack = new float[hidden_dim_ * feature_dim_padded_]();
    float* W1_unpack = new float[W1_.NumElems()];
    float* Wout_unpack = new float[out_dim_padded_ * hidden_dim_];

    cudaMemcpy(W0_unpack_unpadded, W0_.DevicePtr(), W0_.SizeBytes(), cudaMemcpyDeviceToHost);
    cudaMemcpy(W1_unpack, W1_.DevicePtr(), W1_.SizeBytes(), cudaMemcpyDeviceToHost);
    cudaMemcpy(Wout_unpack, Wout_.DevicePtr(), Wout_.SizeBytes(), cudaMemcpyDeviceToHost);

    for (int i = 0; i < hidden_dim_; i++)
    {
      memcpy(W0_unpack + i * feature_dim_padded_, W0_unpack_unpadded + i * feature_dim_, 57 * sizeof(float));
    }
    QuantizeWeights(W0_unpack, hidden_dim_, feature_dim_padded_, 1.0f / 128.0f, W0_host_, W0_scale_);
    QuantizeWeights(W1_unpack, hidden_dim_, hidden_dim_, caldata_.s_a1, W1_host_, W1_scale_);
    QuantizeWeights(Wout_unpack, out_dim_, hidden_dim_, caldata_.s_a2, Wout_host_, Wout_scale_);
    for (int i = out_dim_; i < out_dim_padded_; i++) Wout_scale_[i] = 0.0f;

    delete[] W0_unpack;
    delete[] W1_unpack;
    delete[] Wout_unpack;
    delete[] W0_unpack_unpadded;

    tprogress.phase_ = OpenNTCTrainPhase::TrainComplete;
    train_phase_ = OpenNTCTrainPhase::TrainComplete;
  }
  return tprogress;
}

OpenNTCCalibration OpenNTCContext::Calibrate(float headroom)
{
  // Exact per-layer max |activation| over every texel of every mip, using the
  // same tiling as Eval. Requires trained (post-freeze) weights and grids.
  float* dmax = nullptr;
  cudaMalloc(&dmax, sizeof(float) * 2);
  cudaMemset(dmax, 0, sizeof(float) * 2);

  for (int mip_i = 0; mip_i <= 8; mip_i++)
  {
    int grids_per_dim = (mip_dim_[mip_i] + 255) / 256;
    int num_grids = grids_per_dim * grids_per_dim;
    int grid_draws[16];
    int grid_dim_draw = std::min(256, mip_dim_[mip_i]);
    for (int batch_start = 0; batch_start < num_grids; batch_start += 8)
    {
      int num_batches = std::min(8, num_grids - batch_start);
      int fill_i = 0;
      for (int g = batch_start; g < batch_start + num_batches; g++)
      {
        grid_draws[fill_i + fill_i] = (g % grids_per_dim) * grid_dim_draw;
        grid_draws[fill_i + fill_i + 1] = (g / grids_per_dim) * grid_dim_draw;
        fill_i++;
      }
      cudaMemcpy(grid_draws_.DevicePtr(), grid_draws, sizeof(int) * 2 * num_batches, cudaMemcpyHostToDevice);

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
        feature_dim_,
        mip_dim_[mip_i],
        g0_grid_dim_[feature_level],
        g1_grid_dim_[feature_level],
        g0_channels_,
        g1_channels_,
        mip_i / 8.0f,
        grid_draws_.DevicePtr(),
        g0_noise_.DevicePtr(),
        g1_noise_.DevicePtr(),
        g0_[feature_level].DevicePtr(),
        g1_[feature_level].DevicePtr(),
        x_.DevicePtr());

      matmulAB(handle_, hidden_dim_, batch_dim, feature_dim_, W0_.DevicePtr(), x_.DevicePtr(), W0x_.DevicePtr());
      launch_forward_hardgelu(hidden_dim_ * batch_dim, W0x_.DevicePtr(), W0xa_.DevicePtr());
      launch_max_abs(hidden_dim_ * batch_dim, W0xa_.DevicePtr(), &dmax[0]);
      matmulAB(handle_, hidden_dim_, batch_dim, hidden_dim_, W1_.DevicePtr(), W0xa_.DevicePtr(), W1x_.DevicePtr());
      launch_forward_hardgelu(hidden_dim_ * batch_dim, W1x_.DevicePtr(), W1xa_.DevicePtr());
      launch_max_abs(hidden_dim_ * batch_dim, W1xa_.DevicePtr(), &dmax[1]);
    }
  }

  float hmax[2] = {};
  cudaMemcpy(hmax, dmax, sizeof(float) * 2, cudaMemcpyDeviceToHost);
  cudaFree(dmax);

  OpenNTCCalibration cal = {};
  cal.max_abs_a1 = hmax[0];
  cal.max_abs_a2 = hmax[1];
  cal.s_a1 = headroom * hmax[0] / 127.0f;
  cal.s_a2 = headroom * hmax[1] / 127.0f;
  return cal;
}

OpenNTCEvalResults OpenNTCContext::Eval()
{
  double mse_numer = 0;
  double mse_denom = 0;
  for (int mip_i = 0; mip_i <= 8; mip_i++)
  {
    int grids_per_dim = (mip_dim_[mip_i] + 255) / 256;
    int num_grids = grids_per_dim * grids_per_dim;
    int grid_draws[16];
    int grid_dim_draw = std::min(256, mip_dim_[mip_i]);
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

      cudaMemcpy(grid_draws_.DevicePtr(), grid_draws, sizeof(int) * 2 * num_batches, cudaMemcpyHostToDevice);

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
        feature_dim_,
        mip_dim_[mip_i],
        g0_grid_dim_[feature_level],
        g1_grid_dim_[feature_level],
        g0_channels_,
        g1_channels_,
        mip_i / 8.0f,
        grid_draws_.DevicePtr(),
        g0_noise_.DevicePtr(),
        g1_noise_.DevicePtr(),
        g0_[feature_level].DevicePtr(),
        g1_[feature_level].DevicePtr(),
        x_.DevicePtr());
      launch_draw_targets(num_batches, grid_dim_draw, mip_dim_[mip_i], out_dim_, grid_draws_.DevicePtr(), package_[mip_i].DevicePtr(), mse_.DevicePtr());
      
      // Forward pass

      matmulAB(handle_, hidden_dim_, batch_dim, feature_dim_, W0_.DevicePtr(), x_.DevicePtr(), W0x_.DevicePtr());
      launch_forward_hardgelu(hidden_dim_ * batch_dim, W0x_.DevicePtr(), W0xa_.DevicePtr());
      matmulAB(handle_, hidden_dim_, batch_dim, hidden_dim_, W1_.DevicePtr(), W0xa_.DevicePtr(), W1x_.DevicePtr());
      launch_forward_hardgelu(hidden_dim_ * batch_dim, W1x_.DevicePtr(), W1xa_.DevicePtr());
      matmulAB(handle_, out_dim_, batch_dim, hidden_dim_, Wout_.DevicePtr(), W1xa_.DevicePtr(), Woutx_.DevicePtr());

      float host_neg1 = -1.0f;
      float total_squared_error = 0.0f;
      cublasSaxpy(handle_, out_dim_ * batch_dim, &host_neg1, Woutx_.DevicePtr(), 1, mse_.DevicePtr(), 1);
      cublasSdot(handle_, out_dim_ * batch_dim, mse_.DevicePtr(), 1, mse_.DevicePtr(), 1, &total_squared_error);
      mse_numer += total_squared_error;
      mse_denom += batch_dim * out_dim_;
    }
  }

  OpenNTCEvalResults results = {};
  results.mse = mse_numer / mse_denom;
  results.psnr = -10.0 * std::log10(results.mse);
  return results;
}

OpenNTCCompressedData OpenNTCContext::GetCompressedData()
{
  OpenNTCCompressedData data = {};
  for (int i = 0; i < 4; i++)
  {
    data.g0_[i] = g0_host_[i];
    data.g0_size_[i] = (g0_[i].NumElems() * g0_bytes_per_channel_) / 8;
    data.g1_[i] = g1_host_[i];
    data.g1_size_[i] = (g1_[i].NumElems() * g1_bytes_per_channel_) / 8;
  }
  data.W0_ = W0_host_;
  data.W0_size_ = ((hidden_dim_ * feature_dim_padded_) / 4) * sizeof(uint32_t);
  data.W1_ = W1_host_;
  data.W1_size_ = W1_.SizeBytes() / 4;
  data.Wout_ = Wout_host_;
  data.Wout_size_ = ((out_dim_padded_ * hidden_dim_) / 4) * sizeof(uint32_t);

  data.W0_scale_ = W0_scale_;
  data.W0_scale_size_ = hidden_dim_ * sizeof(float);
  data.W1_scale_ = W1_scale_;
  data.W1_scale_size_ = hidden_dim_ * sizeof(float);
  data.Wout_scale_ = Wout_scale_;
  data.Wout_scale_size_ = out_dim_padded_ * sizeof(float);

  for (int i = 0; i < 4; i++)
  {
    data.g0_grid_dim_[i] = g0_grid_dim_[i];
    data.g1_grid_dim_[i] = g1_grid_dim_[i];
  }
  data.g0_bytes_per_channel_ = g0_bytes_per_channel_;
  data.g1_bytes_per_channel_ = g1_bytes_per_channel_;
  data.g0_channels_ = g0_channels_;
  data.g1_channels_ = g1_channels_;
  data.dim_ = mip_dim_[0];

  data.caldata_ = caldata_;

  return data;
}

static OpenNTCSemantic SemanticFromName(const std::string& s)
{
  static std::pair<std::string, OpenNTCSemantic> s_map_name_to_sem[] = {
    std::make_pair("Albedo", OpenNTCSemantic::Albedo),
    std::make_pair("Diffuse", OpenNTCSemantic::Albedo),
    std::make_pair("Alpha", OpenNTCSemantic::Alpha),
    std::make_pair("Mask", OpenNTCSemantic::Alpha),
    std::make_pair("AlphaMask", OpenNTCSemantic::Alpha),
    std::make_pair("Displ", OpenNTCSemantic::Displacement),
    std::make_pair("Displacement", OpenNTCSemantic::Displacement),
    std::make_pair("Emissive", OpenNTCSemantic::Emissive),
    std::make_pair("Emission", OpenNTCSemantic::Emissive),
    std::make_pair("Glossiness", OpenNTCSemantic::Gloss),
    std::make_pair("Gloss", OpenNTCSemantic::Gloss),
    std::make_pair("Metalness", OpenNTCSemantic::Metallic),
    std::make_pair("Metallic", OpenNTCSemantic::Metallic),
    std::make_pair("Normal", OpenNTCSemantic::Normal),
    std::make_pair("Occlusion", OpenNTCSemantic::AO),
    std::make_pair("AO", OpenNTCSemantic::AO),
    std::make_pair("AmbientOcclusion", OpenNTCSemantic::AO),
    std::make_pair("Roughness", OpenNTCSemantic::Roughness),
    std::make_pair("SpecularColor", OpenNTCSemantic::Specular),
    std::make_pair("Specular", OpenNTCSemantic::Specular),
    std::make_pair("Transmission", OpenNTCSemantic::Transmission),
  };

  for (int i = 0; i < _countof(s_map_name_to_sem); i++)
  {
    if (s_map_name_to_sem[i].first == s)
      return s_map_name_to_sem[i].second;
  }

  return OpenNTCSemantic::None;
}

static int GetChannelCountForSemantic(const OpenNTCSemantic sem)
{
  static const int32_t s_map_sem_to_channel_count[] = {
    -1,               // None (any channel count)
    3,                // Albedo
    1,                // Alpha
    1,                // Displacement
    1,                // Emissive
    1,                // Gloss
    1,                // Metallic
    3,                // Normal
    1,                // AO
    1,                // Roughness
    3,                // Specular
    1,                // Transmission
    -1,               // Count (invalid)
  };

  return s_map_sem_to_channel_count[static_cast<int32_t>(sem)];
}

static bool IsValidSemanticChannels(const OpenNTCSemantic sem, const std::string& channels)
{
  static const std::string rgba = "RGBA";
  if (channels.length() == 0 || channels.length() > 4)
    return false;
  int p = 0;
  for (int i = 0; i < channels.length(); i++)
  {
    while (p < 4 && rgba[p] != channels[i]) p++;
    if (p >= 4)
      return false;
    p++;
  }

  if (sem != OpenNTCSemantic::None && GetChannelCountForSemantic(sem) != channels.length())
    return false;

  return true;
}

// TODO: Better error codes for manifest parsing

OpenNTCResult OpenNTCContext::LoadManifest(const std::string& filepath)
{
  std::ifstream fil(filepath);
  auto jfil = nlohmann::json::parse(fil);
  if (!jfil.contains("textures"))
    return OpenNTCResult::InvalidManifest;
  int source_count = 0;
  int32_t dim = -1;
  int32_t width = jfil.value("width", -1);
  int32_t height = jfil.value("height", -1);
  if (width == -1 && height == -1)
    return OpenNTCResult::InvalidManifest;
  if (width != -1 && height != -1 && width != height)
    return OpenNTCResult::InvalidManifest;
  dim = (width == -1) ? height : width;
  if (dim < OpenNTCContext::kMinDimension || dim > OpenNTCContext::kMaxDimension)
    return OpenNTCResult::InvalidManifest;
  for (const auto& t : jfil["textures"])
  {
    if (!t.contains("semantics") || !t["semantics"].is_object())
      return OpenNTCResult::InvalidManifest;
    if (!t.contains("fileName") || !t["fileName"].is_string())
      return OpenNTCResult::InvalidManifest;
    for (const auto& [sem_name, sem_channels] : t["semantics"].items())
    {
      if (!sem_channels.is_string())
        return OpenNTCResult::InvalidManifest;
      OpenNTCSemantic sem = SemanticFromName(sem_name);
      std::string sem_str = sem_channels.get<std::string>();
      bool valid_sem_channels = IsValidSemanticChannels(sem, sem_str);
      if (!valid_sem_channels)
        return OpenNTCResult::InvalidManifest;
      manifest_.sources_[source_count].path_ = t.value("fileName", "");
      manifest_.sources_[source_count].name_ = t.value("name", "");
      manifest_.sources_[source_count].semantic_ = sem;
      manifest_.sources_[source_count].is_srgb_ = t.value("isSRGB", false);
      manifest_.sources_[source_count].vertical_flip_ = t.value("verticalFlip", false);
      manifest_.sources_[source_count].num_channels_ = GetChannelCountForSemantic(sem);
      for (int i = 0; i < 4; i++)
      {
        OpenNTCChannel ch = OpenNTCChannel::Invalid;
        if (i < manifest_.sources_[source_count].num_channels_)
        {
          if (sem_str[i] == 'R')
            ch = OpenNTCChannel::R;
          else if (sem_str[i] == 'G')
            ch = OpenNTCChannel::G;
          else if (sem_str[i] == 'B')
            ch = OpenNTCChannel::B;
          else
            ch = OpenNTCChannel::A;
        }
        manifest_.sources_[source_count].channel_mapping_[i] = ch;
      }
      source_count++;
    }
  }
  manifest_.source_count_ = source_count;
  manifest_.dim_ = dim;

  for (int i = 0; i < manifest_.source_count_; i++)
  {
    int w;
    int h;
    int c;
    int desired_channels = 4;
    stbi_ldr_to_hdr_gamma(1.0f);
    float* tex_data = stbi_loadf(manifest_.sources_[i].path_.c_str(), &w, &h, &c, desired_channels);
    if (tex_data == nullptr)
    {
      return OpenNTCResult::FileNotFound;
    }
    if (w != h || w != manifest_.dim_)
    {
      stbi_image_free(tex_data);
      return OpenNTCResult::InvalidManifest;
    }
    cudaMemcpy(tex_prep_.DevicePtr(), tex_data, w * h * desired_channels * sizeof(float), cudaMemcpyHostToDevice);
    stbi_image_free(tex_data);

    PrepareTexInput prepare_in = {};
    for (int j = 0; j < 4; j++) prepare_in.cmap[j] = static_cast<int32_t>(manifest_.sources_[i].channel_mapping_[j]);
    launch_prepare_tex(manifest_.dim_, desired_channels, prepare_in, tex_prep_.DevicePtr(), mips_[i][0].DevicePtr());
    for (int j = 1; j < mip_count_; j++)
    {
      launch_filter_lanczos(mip_dim_[j - 1], manifest_.sources_[i].num_channels_, 3, mips_[i][j - 1].DevicePtr(), tex_filter_.DevicePtr(), mips_[i][j].DevicePtr());
    }
    for (int j = 0; j < mip_count_; j++)
      cudaMemcpy(mips_host_[i][j], mips_[i][j].DevicePtr(), sizeof(float) * mip_dim_[j] * mip_dim_[j] * manifest_.sources_[i].num_channels_, cudaMemcpyDeviceToHost);
  }

  PackageTexInput package_in = {};
  int p = 0;
  for (int j = 0; j < manifest_.source_count_; j++)
  {
    package_in.source_channels[j] = manifest_.sources_[j].num_channels_;
    for (int c = 0; c < manifest_.sources_[j].num_channels_; c++)
    {
      package_in.map_feat_id_to_source_id[p] = j;
      package_in.map_feat_id_to_channel_id[p] = c;
      p++;
    }
  }

  if (p != out_dim_)
    return OpenNTCResult::InvalidManifest;

  for (int i = 0; i < mip_count_; i++)
  {
    for (int j = 0; j < manifest_.source_count_; j++)
      package_in.tex[j] = mips_[j][i].DevicePtr();
    launch_package_tex(mip_dim_[i], out_dim_, package_in, package_[i].DevicePtr());
  }

  manifest_loaded_ = true;
  return OpenNTCResult::Success;
}

OpenNTCTextureData OpenNTCContext::GetTextureData()
{
  OpenNTCTextureData tex_data = {};
  tex_data.tex_count_ = manifest_.source_count_;
  tex_data.mip_count_ = mip_count_;
  for (int i = 0; i < manifest_.source_count_; i++)
  {
    tex_data.semantics_[i] = manifest_.sources_[i].semantic_;
    tex_data.channels_[i] = manifest_.sources_[i].num_channels_;
    for (int j = 0; j < mip_count_; j++)
      tex_data.mips_[i][j] = mips_host_[i][j];
  }
  return tex_data;
}

void OpenNTCContext::Destroy()
{
  for (int i = 0; i < 4; i++)
  {
    g0_[i].Destroy();
    g1_[i].Destroy();
  }

  g0_noise_.Destroy();
  g1_noise_.Destroy();

  W0_.Destroy();
  W1_.Destroy();
  Wout_.Destroy();

  W0x_.Destroy();
  W0xa_.Destroy();
  W1x_.Destroy();
  W1xa_.Destroy();
  Woutx_.Destroy();

  dLdWoutx_.Destroy();
  dLdWout_.Destroy();
  dLdW1xa_.Destroy();
  dLdW1x_.Destroy();
  dLdW1_.Destroy();
  dLdW0xa_.Destroy();
  dLdW0x_.Destroy();
  dLdW0_.Destroy();
  dLdx_.Destroy();
  for (int level_i = 0; level_i < 4; level_i++)
  {
    dLdG0_[level_i].Destroy();
    dLdG1_[level_i].Destroy();
  }

  mse_.Destroy();

  for (int level_i = 0; level_i < 4; level_i++)
  {
    mG0_[level_i].Destroy();
    vG0_[level_i].Destroy();
    mG1_[level_i].Destroy();
    vG1_[level_i].Destroy();
  }
  mW0_.Destroy();
  vW0_.Destroy();
  mW1_.Destroy();
  vW1_.Destroy();
  mWout_.Destroy();
  vWout_.Destroy();

  grid_draws_.Destroy();
  x_.Destroy();

  for (int i = 0; i < kMaxSources; i++)
  {
    for (int j = 0; j < OpenNTCContext::kMaxMips; j++)
    {
      mips_[i][j].Destroy();
    }
  }
  tex_prep_.Destroy();
  tex_filter_.Destroy();
  for (int i = 0; i < mip_count_; i++)
  {
    package_[i].Destroy();
  }

  cudaFree(rstate_);

  for(int i = 0; i < 4; i++)
  {
    delete[] g0_host_[i];
    delete[] g1_host_[i];
  }
  delete[] W0_host_;
  delete[] W1_host_;
  delete[] Wout_host_;
  delete[] W0_scale_;
  delete[] W1_scale_;
  delete[] Wout_scale_;

  for (int i = 0; i < kMaxSources; i++)
  {
    for (int j = 0; j < OpenNTCContext::kMaxMips; j++)
      delete[] mips_host_[i][j];
  }
}