#include <libopenntc/libopenntc.h>

#include <algorithm>
#include <cassert>
#include <bit>
#include <fstream>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand_kernel.h>

#define STB_IMAGE_IMPLEMENTATION
#include <libopenntc/stb_image.h>
#include <libopenntc/ntc_kernel.cuh>
#include <libopenntc/json.hpp>

namespace openntc
{
Tensor2d::Tensor2d() : initialized_(false), dev_(nullptr) {}

Tensor2d::~Tensor2d()
{
  Destroy();
}

Result Tensor2d::Init(int x, int y)
{
  shape_[0] = x;
  shape_[1] = y;
  cudaMalloc(&dev_, sizeof(float) * x * y);
  if (dev_ == nullptr)
    return Result::AllocationFailure;
  initialized_ = true;
  return Result::Success;
}

Result Tensor2d::InitLike(const Tensor2d& t0)
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

Result Tensor3d::Init(int x, int y, int z)
{
  shape_[0] = x;
  shape_[1] = y;
  shape_[2] = z;
  cudaMalloc(&dev_, sizeof(float) * x * y * z);
  if (dev_ == nullptr)
    return Result::AllocationFailure;
  initialized_ = true;
  return Result::Success;
}

Result Tensor3d::InitLike(const Tensor3d& t0)
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

U8Tensor3d::U8Tensor3d() : initialized_(false), dev_(nullptr) {}

U8Tensor3d::~U8Tensor3d()
{
  Destroy();
}

Result U8Tensor3d::Init(int x, int y, int z)
{
  shape_[0] = x;
  shape_[1] = y;
  shape_[2] = z;
  cudaMalloc(&dev_, sizeof(uint8_t) * x * y * z);
  if (dev_ == nullptr)
    return Result::AllocationFailure;
  initialized_ = true;
  return Result::Success;
}

Result U8Tensor3d::InitLike(const U8Tensor3d& t0)
{
  return Init(t0.shape_[0], t0.shape_[1], t0.shape_[2]);
}

void U8Tensor3d::FillZero()
{
  cudaMemset(dev_, 0, sizeof(uint8_t) * shape_[0] * shape_[1] * shape_[2]);
}

void U8Tensor3d::Destroy()
{
  if (initialized_)
  {
    cudaFree(dev_);
    initialized_ = false;
  }
}

uint8_t* U8Tensor3d::DevicePtr()
{
  assert(initialized_);
  return dev_;
}

bool U8Tensor3d::IsInitialized() const
{
  return initialized_;
}

size_t U8Tensor3d::SizeBytes() const
{
  return sizeof(uint8_t) * shape_[0] * shape_[1] * shape_[2];
}

size_t U8Tensor3d::NumElems() const
{
  return (size_t)shape_[0] * shape_[1] * shape_[2];
}

IntTensor1d::IntTensor1d() : initialized_(false), dev_(nullptr) {}

IntTensor1d::~IntTensor1d()
{
  Destroy();
}

Result IntTensor1d::Init(int x)
{
  shape_[0] = x;
  cudaMalloc(&dev_, sizeof(int) * x);
  if (dev_ == nullptr)
    return Result::AllocationFailure;
  initialized_ = true;
  return Result::Success;
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

Context::Context() : initialized_(false), gen_(123), rstate_(nullptr), mip_count_(0), level_count_(0) {}
Context::~Context() { Destroy(); }

static int RoundUpToNearestK(int n, int k)
{
  assert(k > 0);
  return ((n + k - 1) / k) * k;
}

Result Context::Init(const ContextInitInfo& init_info)
{
  assert(!initialized_);
  if (initialized_)
    return Result::InvalidState;

  // Profile constants

  profile_ = init_info.profile_;
  switch (init_info.profile_)
  {
  case Profile::Bpp_0_2:
    g0_bits_per_channel_ = 2;
    g1_bits_per_channel_ = 4;
    g0_channels_ = 8;
    g1_channels_ = 12;
    break;
  case Profile::Bpp_0_5:
    g0_bits_per_channel_ = 4;
    g1_bits_per_channel_ = 4;
    g0_channels_ = 12;
    g1_channels_ = 20;
    break;
  case Profile::Bpp_1_0:
    g0_bits_per_channel_ = 2;
    g1_bits_per_channel_ = 4;
    g0_channels_ = 12;
    g1_channels_ = 10;
    break;
  case Profile::Bpp_2_25:
    g0_bits_per_channel_ = 4;
    g1_bits_per_channel_ = 4;
    g0_channels_ = 16;
    g1_channels_ = 12;
    break;
  default:
    assert(false);
    return Result::InvalidState;
  }
  switch (init_info.profile_)
  {
  case Profile::Bpp_1_0:
  case Profile::Bpp_2_25:
    g0_scale_ = 2;
    break;
  default:
    g0_scale_ = 4;
    break;
  }
  g0_delta_ = 2.0f / powf(2.0f, (float) g0_bits_per_channel_);
  g1_delta_ = 2.0f / powf(2.0f, (float) g1_bits_per_channel_);
  
  feature_dim_ = RoundUpToNearestK(4 * g0_channels_ + g1_channels_ + 12 + 1, 4);
  feature_dim_padded_ = RoundUpToNearestK(feature_dim_, 16);
  out_dim_ = 0;
  out_dim_padded_ = kMaxChannels;
  max_batch_ = 8;
  max_batch_dim_ = max_batch_ * 256 * 256;
  hidden_dim_ = 64;

  W0_.Init(hidden_dim_, feature_dim_);
  W1_.Init(hidden_dim_, hidden_dim_);
  Wout_.Init(out_dim_padded_, hidden_dim_);

  x_.Init(feature_dim_, max_batch_dim_);
  W0x_.Init(hidden_dim_, max_batch_dim_);
  W0xa_.Init(hidden_dim_, max_batch_dim_);
  W1x_.Init(hidden_dim_, max_batch_dim_);
  W1xa_.Init(hidden_dim_, max_batch_dim_);
  Woutx_.Init(out_dim_padded_, max_batch_dim_);

  dLdWoutx_.InitLike(Woutx_);
  dLdWout_.InitLike(Wout_);
  dLdW1xa_.InitLike(W1xa_);
  dLdW1x_.InitLike(W1x_);
  dLdW1_.InitLike(W1_);
  dLdW0xa_.InitLike(W0xa_);
  dLdW0x_.InitLike(W0x_);
  dLdW0_.InitLike(W0_);
  dLdx_.InitLike(x_);

  mse_.Init(out_dim_padded_, max_batch_dim_);

  mW0_.InitLike(W0_);
  vW0_.InitLike(W0_);
  mW1_.InitLike(W1_);
  vW1_.InitLike(W1_);
  mWout_.InitLike(Wout_);
  vWout_.InitLike(Wout_);

  grid_draws_.Init(2 * max_batch_);

  cublasHandle_t cublas_handle = nullptr;
  cublasCreate(&cublas_handle);
  cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH);
  handle_ = cublas_handle;

  rand_dim_ = 1024 * 1024;
  cudaMalloc(&rstate_, sizeof(curandState) * rand_dim_);
  launch_initialize_rand(rand_dim_, rstate_);

  size_t W0_size = hidden_dim_ * feature_dim_padded_;
  size_t W1_size = hidden_dim_ * hidden_dim_;
  size_t Wout_size = out_dim_padded_ * hidden_dim_;
  size_t W0_scale_size = hidden_dim_ * sizeof(float);
  size_t W1_scale_size = hidden_dim_ * sizeof(float);
  size_t Wout_scale_size = out_dim_padded_ * sizeof(float);
  decoder_size_ = W0_size + W1_size + Wout_size + W0_scale_size + W1_scale_size + Wout_scale_size;
  decoder_host_ = new uint8_t[decoder_size_];
  memset(decoder_host_, 0, decoder_size_);

  initialized_ = true;
  return Result::Success;
}

// C = A * B, where A, B, C are row-major. C is n x m, A is n x k, B is k x m

static void matmulAB(void* handle, int n, int m, int k, float* A, float* B, float* C)
{
  float sgemm_alpha = 1.0f, sgemm_beta = 0.0f;
  cublasSgemm((cublasHandle_t)handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &sgemm_alpha, B, m, A, k, &sgemm_beta, C, m);
}

// C = A^T * B, where A, B, C are row-major. C is n x m, A is k x n, B is k x m

static void matmulATB(void* handle, int n, int m, int k, float* A, float* B, float* C)
{
  float sgemm_alpha = 1.0f, sgemm_beta = 0.0f;
  cublasSgemm((cublasHandle_t)handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &sgemm_alpha, B, m, A, n, &sgemm_beta, C, m);
}

// C = A * B^T, where A, B, C are row-major. C is n x m, A is n x k, B is m x k

static void matmulABT(void* handle, int n, int m, int k, float* A, float* B, float* C)
{
  float sgemm_alpha = 1.0f, sgemm_beta = 0.0f;
  cublasSgemm((cublasHandle_t)handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &sgemm_alpha, B, k, A, k, &sgemm_beta, C, m);
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

void Context::BeginTraining(const TrainInfo& train_info)
{
  assert(train_phase_ == TrainPhase::ManifestLoaded || train_phase_ == TrainPhase::TrainComplete);
  batch_count_ = train_info.batch_count_;
  lock_i_ = (95 * batch_count_) / 100;
  batch_i_ = 0;
  grids_per_batch_ = train_info.grids_per_batch_;
  for (int i = 0; i < level_count_; i++)
  {
    grid_batch_i_[i] = 0;
  }

  for (int level_i = 0; level_i < level_count_; level_i++)
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

  train_phase_ = TrainPhase::TrainInProgress;
}

TrainProgress Context::TrainUntilComplete()
{
  assert(train_phase_ == TrainPhase::TrainInProgress || train_phase_ == TrainPhase::TrainComplete);

  return Train(batch_count_);
}

TrainProgress Context::Train(int32_t batch_count)
{
  if (!manifest_loaded_ ||
      (train_phase_ != TrainPhase::TrainInProgress &&
      train_phase_ != TrainPhase::TrainComplete))
  {
    TrainProgress tprogress = {};
    tprogress.phase_ = TrainPhase::TrainError;
    tprogress.result_ = Result::InvalidState;
    tprogress.batches_complete_ = batch_i_;
    tprogress.total_batches_ = batch_count_;
    return tprogress;
  }

  assert(batch_count > 0);
  int batches_remaining = std::max(0, batch_count_ - batch_i_);
  batch_count = std::min(batches_remaining, batch_count);
  int batch_target = batch_i_ + batch_count;
  if (batch_i_ == batch_target)
  {
    TrainProgress tprogress = {};
    tprogress.phase_ = TrainPhase::TrainComplete;
    tprogress.result_ = Result::Success;
    tprogress.batches_complete_ = batch_i_;
    tprogress.total_batches_ = batch_count_;
    return tprogress;
  }

  std::bernoulli_distribution dist_batch_type(0.05);
  std::uniform_real_distribution<float> dist_u(0.0f, 1.0f);
  std::uniform_int_distribution<int> dist_lod(0, mip_count_ - 1);

  for (batch_i_; batch_i_ < batch_target; batch_i_++)
  {
    bool draw_uniform = dist_batch_type(gen_);
    float U = dist_u(gen_);
    int lod = std::min(int(std::floorf(0.5f * -log2f(U))), mip_count_ - 1);
    int lod_uniform = dist_lod(gen_);
    if (draw_uniform)
      lod = lod_uniform;
    int feature_level = FeatureLevelForLod(lod);
    int grid_draws[16];
    std::uniform_int_distribution<int> dist_grid(0, mip_dim_[lod] - 1);
    for (int i = 0; i < grids_per_batch_; i++)
    {
      grid_draws[i + i] = dist_grid(gen_);
      grid_draws[i + i + 1] = dist_grid(gen_);
    }
    cudaMemcpy(grid_draws_.DevicePtr(), grid_draws, sizeof(int) * grids_per_batch_ * 2, cudaMemcpyHostToDevice);
    int grid_dim_draw = std::min(mip_dim_[lod], 256);
    if (batch_i_ < lock_i_)
    {
      launch_generate_noise(
        g0_grid_dim_[feature_level] * g0_grid_dim_[feature_level] * g0_channels_,
        rand_dim_,
        g0_delta_,
        rstate_,
        g0_noise_.DevicePtr());
      launch_generate_noise(
        g1_grid_dim_[feature_level] * g1_grid_dim_[feature_level] * g1_channels_,
        rand_dim_,
        g1_delta_,
        rstate_,
        g1_noise_.DevicePtr());
    }
    else if (batch_i_ == lock_i_)
    {
      for (int level_i = 0; level_i < level_count_; level_i++)
      {
        launch_quantize_grid(
          g0_grid_dim_[level_i] * g0_grid_dim_[level_i] * g0_channels_,
          g0_bits_per_channel_,
          g0_delta_,
          g0_[level_i].DevicePtr());
        launch_quantize_grid(
          g1_grid_dim_[level_i] * g1_grid_dim_[level_i] * g1_channels_,
          g1_bits_per_channel_,
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
      lod / static_cast<float>(mip_count_ - 1),
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
    cublasSaxpy((cublasHandle_t)handle_, out_dim_ * batch_dim, &host_neg1, Woutx_.DevicePtr(), 1, mse_.DevicePtr(), 1);
    cublasSdot((cublasHandle_t)handle_, out_dim_ * batch_dim, mse_.DevicePtr(), 1, mse_.DevicePtr(), 1, &total_squared_error);
    float mse = total_squared_error / ((float) out_dim_ * batch_dim);
    launch_scalar_product(
      out_dim_ * batch_dim,
      -2.0f / ((float)out_dim_ * batch_dim),
      mse_.DevicePtr(),
      dLdWoutx_.DevicePtr());

    // Backward pass

    matmulABT(
      handle_,
      out_dim_,
      hidden_dim_,
      batch_dim,
      dLdWoutx_.DevicePtr(),
      W1xa_.DevicePtr(),
      dLdWout_.DevicePtr());
    matmulATB(
      handle_,
      hidden_dim_,
      batch_dim,
      out_dim_,
      Wout_.DevicePtr(),
      dLdWoutx_.DevicePtr(),
      dLdW1xa_.DevicePtr());
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

    launch_update_adam(
      out_dim_ * hidden_dim_,
      lr_decoder,
      beta_1,
      beta_2,
      bias_1,
      bias_2,
      dLdWout_.DevicePtr(),
      mWout_.DevicePtr(),
      vWout_.DevicePtr(),
      Wout_.DevicePtr());
    launch_update_adam(
      hidden_dim_ * hidden_dim_,
      lr_decoder,
      beta_1,
      beta_2,
      bias_1,
      bias_2,
      dLdW1_.DevicePtr(),
      mW1_.DevicePtr(),
      vW1_.DevicePtr(),
      W1_.DevicePtr());
    launch_update_adam(
      hidden_dim_ * feature_dim_,
      lr_decoder,
      beta_1,
      beta_2,
      bias_1,
      bias_2,
      dLdW0_.DevicePtr(),
      mW0_.DevicePtr(),
      vW0_.DevicePtr(),
      W0_.DevicePtr());
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
        g0_bits_per_channel_,
        g0_delta_,
        g0_[feature_level].DevicePtr());
      launch_clamp_grid(
        g1_grid_dim_[feature_level] * g1_grid_dim_[feature_level] * g1_channels_,
        g1_bits_per_channel_,
        g1_delta_,
        g1_[feature_level].DevicePtr());
    }
  }

  TrainProgress tprogress = {};
  tprogress.phase_ = TrainPhase::TrainInProgress;
  tprogress.result_ = Result::Success;
  tprogress.batches_complete_ = batch_i_;
  tprogress.total_batches_ = batch_count_;

  if (batch_i_ == batch_count_)
  {
    tprogress.phase_ = TrainPhase::TrainComplete;
    train_phase_ = TrainPhase::TrainComplete;

    uint32_t* g0pack = nullptr;
    uint32_t* g1pack = nullptr;

    cudaMalloc(&g0pack, sizeof(uint32_t) * (g0_[0].NumElems() * g0_bits_per_channel_) / 32);
    cudaMalloc(&g1pack, sizeof(uint32_t) * (g1_[0].NumElems() * g1_bits_per_channel_) / 32);

    for (int i = 0; i < level_count_; i++)
    {
      launch_quantize_pack(
        g0_[i].NumElems(),
        (g0_[i].NumElems() * g0_bits_per_channel_) / 32,
        g0_bits_per_channel_,
        g0_[i].DevicePtr(),
        g0pack);
      launch_quantize_pack(
        g1_[i].NumElems(),
        (g1_[i].NumElems() * g1_bits_per_channel_) / 32,
        g1_bits_per_channel_,
        g1_[i].DevicePtr(),
        g1pack);
      cudaMemcpy(
        g0_host_[i],
        g0pack,
        sizeof(uint32_t) * (g0_[i].NumElems() * g0_bits_per_channel_) / 32,
        cudaMemcpyDeviceToHost);
      cudaMemcpy(
        g1_host_[i],
        g1pack,
        sizeof(uint32_t) * (g1_[i].NumElems() * g1_bits_per_channel_) / 32,
        cudaMemcpyDeviceToHost);
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
      memcpy(W0_unpack + i * feature_dim_padded_, W0_unpack_unpadded + i * feature_dim_, feature_dim_ * sizeof(float));
    }
    // Section views into the decoder blob; layout mirrors DEC_*_OFFSET in common.hlsli
    uint8_t* dec = decoder_host_;
    uint32_t* W0_out = (uint32_t*)(dec);
    dec += hidden_dim_ * feature_dim_padded_;
    uint32_t* W1_out = (uint32_t*)(dec);
    dec += hidden_dim_ * hidden_dim_;
    uint32_t* Wout_out = (uint32_t*)(dec);
    dec += out_dim_padded_ * hidden_dim_;
    float* W0_scale_out = (float*)(dec);
    dec += hidden_dim_ * sizeof(float);
    float* W1_scale_out = (float*)(dec);
    dec += hidden_dim_ * sizeof(float);
    float* Wout_scale_out = (float*)(dec);

    QuantizeWeights(W0_unpack, hidden_dim_, feature_dim_padded_, 1.0f / 128.0f, W0_out, W0_scale_out);
    QuantizeWeights(W1_unpack, hidden_dim_, hidden_dim_, caldata_.s_a1_, W1_out, W1_scale_out);
    QuantizeWeights(Wout_unpack, out_dim_, hidden_dim_, caldata_.s_a2_, Wout_out, Wout_scale_out);
    for (int i = out_dim_; i < out_dim_padded_; i++) Wout_scale_out[i] = 0.0f;

    delete[] W0_unpack;
    delete[] W1_unpack;
    delete[] Wout_unpack;
    delete[] W0_unpack_unpadded;
  }
  return tprogress;
}

int32_t Context::FeatureLevelForLod(int32_t lod) const
{
  // First few lods are handled by the highest-resolution feature level,
  //  then each successive level handles two each.
  
  int lods_for_first_level = (int)log2f(g0_scale_) + 2;
  if (lod < lods_for_first_level)
    return 0;
  int level = 1 + (lod - lods_for_first_level) / 2;
  return std::min(level, level_count_ - 1);
}

CalibrationData Context::Calibrate(float headroom)
{
  assert (manifest_loaded_ && train_phase_ == TrainPhase::TrainComplete);

  // TODO: Switch this out with allcation at Init-time.

  float* dmax = nullptr;
  cudaMalloc(&dmax, sizeof(float) * 2);
  cudaMemset(dmax, 0, sizeof(float) * 2);

  for (int mip_i = 0; mip_i < mip_count_; mip_i++)
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

      int32_t feature_level = FeatureLevelForLod(mip_i);

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
        mip_i / static_cast<float>(mip_count_ - 1),
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

  CalibrationData cal = {};
  cal.max_abs_a1_ = hmax[0];
  cal.max_abs_a2_ = hmax[1];
  cal.s_a1_ = headroom * hmax[0] / 127.0f;
  cal.s_a2_ = headroom * hmax[1] / 127.0f;
  return cal;
}

EvalResults Context::Eval()
{
  assert(manifest_loaded_ && (train_phase_ == TrainPhase::TrainInProgress || train_phase_==TrainPhase::TrainComplete));

  double mse_numer = 0;
  double mse_denom = 0;
  for (int mip_i = 0; mip_i < mip_count_; mip_i++)
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

      int feature_level = FeatureLevelForLod(mip_i);

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
        mip_i / static_cast<float>(mip_count_ - 1),
        grid_draws_.DevicePtr(),
        g0_noise_.DevicePtr(),
        g1_noise_.DevicePtr(),
        g0_[feature_level].DevicePtr(),
        g1_[feature_level].DevicePtr(),
        x_.DevicePtr());
      launch_draw_targets(
        num_batches,
        grid_dim_draw,
        mip_dim_[mip_i],
        out_dim_,
        grid_draws_.DevicePtr(),
        package_[mip_i].DevicePtr(),
        mse_.DevicePtr());
      
      // Forward pass

      matmulAB(handle_, hidden_dim_, batch_dim, feature_dim_, W0_.DevicePtr(), x_.DevicePtr(), W0x_.DevicePtr());
      launch_forward_hardgelu(hidden_dim_ * batch_dim, W0x_.DevicePtr(), W0xa_.DevicePtr());
      matmulAB(handle_, hidden_dim_, batch_dim, hidden_dim_, W1_.DevicePtr(), W0xa_.DevicePtr(), W1x_.DevicePtr());
      launch_forward_hardgelu(hidden_dim_ * batch_dim, W1x_.DevicePtr(), W1xa_.DevicePtr());
      matmulAB(handle_, out_dim_, batch_dim, hidden_dim_, Wout_.DevicePtr(), W1xa_.DevicePtr(), Woutx_.DevicePtr());

      float host_neg1 = -1.0f;
      float total_squared_error = 0.0f;
      cublasSaxpy((cublasHandle_t)handle_, out_dim_ * batch_dim, &host_neg1, Woutx_.DevicePtr(), 1, mse_.DevicePtr(), 1);
      cublasSdot((cublasHandle_t)handle_, out_dim_ * batch_dim, mse_.DevicePtr(), 1, mse_.DevicePtr(), 1, &total_squared_error);
      mse_numer += total_squared_error;
      mse_denom += batch_dim * out_dim_;
    }
  }

  EvalResults results = {};
  results.mse_ = mse_numer / mse_denom;
  results.psnr_ = -10.0 * std::log10(results.mse_);
  return results;
}

CompressedData Context::GetCompressedData()
{
  assert(manifest_loaded_ &&
    (train_phase_ == TrainPhase::TrainInProgress || train_phase_ == TrainPhase::TrainComplete));

  CompressedData data = {};
  for (int i = 0; i < level_count_; i++)
  {
    data.g0_[i] = g0_host_[i];
    data.g0_size_[i] = (g0_[i].NumElems() * g0_bits_per_channel_) / 8;
    data.g1_[i] = g1_host_[i];
    data.g1_size_[i] = (g1_[i].NumElems() * g1_bits_per_channel_) / 8;
  }
  for (int i = 1; i < level_count_; i++)
  {
    data.g0_offset_[i] = data.g0_offset_[i - 1] + data.g0_size_[i - 1];
    data.g1_offset_[i] = data.g1_offset_[i - 1] + data.g1_size_[i - 1];
  }
  data.decoder_ = decoder_host_;
  data.decoder_size_ = decoder_size_;

  for (int i = 0; i < level_count_; i++)
  {
    data.g0_grid_dim_[i] = g0_grid_dim_[i];
    data.g1_grid_dim_[i] = g1_grid_dim_[i];
  }
  data.profile_ = profile_;
  data.g0_bits_per_channel_ = g0_bits_per_channel_;
  data.g1_bits_per_channel_ = g1_bits_per_channel_;
  data.g0_channels_ = g0_channels_;
  data.g1_channels_ = g1_channels_;
  data.dim_ = mip_dim_[0];
  data.mip_count_ = mip_count_;
  data.level_count_ = level_count_;

  data.caldata_ = caldata_;
  data.channel_count_ = 0;
  for (int i = 0; i < manifest_.source_count_; i++)
  {
    for (int c = 0; c < manifest_.sources_[i].num_channels_; c++)
      data.channel_semantics_[data.channel_count_++] = manifest_.sources_[i].semantic_;
  }

  return data;
}

static const char* SemanticToString(Semantic sem)
{
  switch (sem)
  {
    case Semantic::Albedo:
      return "Albedo";
    case Semantic::Alpha:
      return "Alpha";
    case Semantic::Displacement:
      return "Displacement";
    case Semantic::Emissive:
      return "Emissive";
    case Semantic::Gloss:
      return "Gloss";
    case Semantic::Metallic:
      return "Metallic";
    case Semantic::Normal:
      return "Normal";
    case Semantic::AO:
      return "AO";
    case Semantic::Roughness:
      return "Roughness";
    case Semantic::Specular:
      return "Specular";
    case Semantic::Transmission:
      return "Transmission";
    default:
      return "None";
  }
}

static Semantic SemanticFromName(const std::string& s)
{
  static std::pair<std::string, Semantic> s_map_name_to_sem[] = {
    std::make_pair("Albedo", Semantic::Albedo),
    std::make_pair("Diffuse", Semantic::Albedo),
    std::make_pair("Alpha", Semantic::Alpha),
    std::make_pair("Mask", Semantic::Alpha),
    std::make_pair("AlphaMask", Semantic::Alpha),
    std::make_pair("Displ", Semantic::Displacement),
    std::make_pair("Displacement", Semantic::Displacement),
    std::make_pair("Emissive", Semantic::Emissive),
    std::make_pair("Emission", Semantic::Emissive),
    std::make_pair("Glossiness", Semantic::Gloss),
    std::make_pair("Gloss", Semantic::Gloss),
    std::make_pair("Metalness", Semantic::Metallic),
    std::make_pair("Metallic", Semantic::Metallic),
    std::make_pair("Normal", Semantic::Normal),
    std::make_pair("Occlusion", Semantic::AO),
    std::make_pair("AO", Semantic::AO),
    std::make_pair("AmbientOcclusion", Semantic::AO),
    std::make_pair("Roughness", Semantic::Roughness),
    std::make_pair("SpecularColor", Semantic::Specular),
    std::make_pair("Specular", Semantic::Specular),
    std::make_pair("Transmission", Semantic::Transmission),
  };

  for (int i = 0; i < _countof(s_map_name_to_sem); i++)
  {
    if (s_map_name_to_sem[i].first == s)
      return s_map_name_to_sem[i].second;
  }

  return Semantic::None;
}

static int GetChannelCountForSemantic(const Semantic sem)
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

static bool IsValidSemanticChannels(const Semantic sem, const std::string& channels)
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

  if (sem != Semantic::None && GetChannelCountForSemantic(sem) != channels.length())
    return false;

  return true;
}

// TODO: Better error codes for manifest parsing

Result Context::LoadManifest(const std::string& filepath)
{
  UnloadManifest();

  std::ifstream fil(filepath);
  if (!fil)
    return Result::FileNotFound;
  auto jfil = nlohmann::json::parse(fil, nullptr, false);
  if (jfil.is_discarded() || !jfil.contains("textures"))
    return Result::InvalidManifest;
  int source_count = 0;
  int32_t dim = -1;
  int32_t width = jfil.value("width", -1);
  int32_t height = jfil.value("height", -1);
  if (width == -1 && height == -1)
    return Result::InvalidManifest;
  if (width != -1 && height != -1 && width != height)
    return Result::InvalidManifest;
  dim = (width == -1) ? height : width;
  if (dim < Context::kMinDimension || dim > Context::kMaxDimension)
    return Result::InvalidManifest;
  for (const auto& t : jfil["textures"])
  {
    if (!t.contains("semantics") || !t["semantics"].is_object())
      return Result::InvalidManifest;
    if (!t.contains("fileName") || !t["fileName"].is_string())
      return Result::InvalidManifest;
    for (const auto& [sem_name, sem_channels] : t["semantics"].items())
    {
      if (!sem_channels.is_string())
        return Result::InvalidManifest;
      Semantic sem = SemanticFromName(sem_name);
      std::string sem_str = sem_channels.get<std::string>();
      bool valid_sem_channels = IsValidSemanticChannels(sem, sem_str);
      if (!valid_sem_channels)
        return Result::InvalidManifest;
      manifest_.sources_[source_count].path_ = t.value("fileName", "");
      manifest_.sources_[source_count].name_ = t.value("name", "");
      manifest_.sources_[source_count].semantic_ = sem;
      manifest_.sources_[source_count].is_srgb_ = t.value("isSRGB", false);
      manifest_.sources_[source_count].vertical_flip_ = t.value("verticalFlip", false);
      manifest_.sources_[source_count].num_channels_ = GetChannelCountForSemantic(sem);
      for (int i = 0; i < 4; i++)
      {
        Channel ch = Channel::Invalid;
        if (i < manifest_.sources_[source_count].num_channels_)
        {
          if (sem_str[i] == 'R')
            ch = Channel::R;
          else if (sem_str[i] == 'G')
            ch = Channel::G;
          else if (sem_str[i] == 'B')
            ch = Channel::B;
          else
            ch = Channel::A;
        }
        manifest_.sources_[source_count].channel_mapping_[i] = ch;
      }
      source_count++;
    }
  }
  manifest_.source_count_ = source_count;

  int total_channels = 0;
  for (int i = 0; i < source_count; i++)
    total_channels += manifest_.sources_[i].num_channels_;
  if (total_channels < 1 || total_channels > Context::kMaxChannels)
    return Result::InvalidManifest;
  out_dim_ = total_channels;

  // Order channels by semantic index
  auto semantic_cmp = [](const TextureSource& a, const TextureSource& b) {
    return (int32_t)a.semantic_ < (int32_t)b.semantic_;
  };
  std::stable_sort(manifest_.sources_, manifest_.sources_ + source_count, semantic_cmp);
  manifest_.dim_ = dim;

  if (manifest_.dim_ < Context::kMinDimension ||
      manifest_.dim_ > Context::kMaxDimension ||
      std::popcount((unsigned int)manifest_.dim_) != 1)
  {
    return Result::InvalidDimension;
  }

  mip_dim_[0] = manifest_.dim_;
  mip_count_ = 0;
  while (mip_dim_[mip_count_] > 4)
  {
    mip_dim_[mip_count_ + 1] = mip_dim_[mip_count_] / 2;
    mip_count_++;
  }
  mip_count_++;

  level_count_ = 0;
  g0_grid_dim_[0] = mip_dim_[0] / g0_scale_;
  g1_grid_dim_[0] = g0_grid_dim_[0] / 2;
  while (g0_grid_dim_[level_count_] / 4 >= 4)
  {
    g0_grid_dim_[level_count_ + 1] = g0_grid_dim_[level_count_] / 4;
    g1_grid_dim_[level_count_ + 1] = g1_grid_dim_[level_count_] / 4;
    level_count_++;
  }
  level_count_++;

  // TODO: No need to reallocate these if we are loading a manifest at the same resolution

  for (int i = 0; i < level_count_; i++)
  {
    g0_[i].Init(g0_grid_dim_[i], g0_grid_dim_[i], g0_channels_);
    g1_[i].Init(g1_grid_dim_[i], g1_grid_dim_[i], g1_channels_);
  }

  g0_noise_.Init(g0_grid_dim_[0], g0_grid_dim_[0], g0_channels_);
  g1_noise_.Init(g1_grid_dim_[0], g1_grid_dim_[0], g1_channels_);

  for (int level_i = 0; level_i < level_count_; level_i++)
  {
    dLdG0_[level_i].InitLike(g0_[level_i]);
    dLdG1_[level_i].InitLike(g1_[level_i]);
  }

  for (int level_i = 0; level_i < level_count_; level_i++)
  {
    mG0_[level_i].InitLike(g0_[level_i]);
    vG0_[level_i].InitLike(g0_[level_i]);
    mG1_[level_i].InitLike(g1_[level_i]);
    vG1_[level_i].InitLike(g1_[level_i]);
  }

  for (int i = 0; i < level_count_; i++)
  {
    g0_host_[i] = new uint32_t[(g0_[i].NumElems() * g0_bits_per_channel_) / 32];
    g1_host_[i] = new uint32_t[(g1_[i].NumElems() * g1_bits_per_channel_) / 32];
  }

  for (int i = 0; i < mip_count_; i++)
  {
    package_[i].Init(mip_dim_[i], mip_dim_[i], out_dim_);
  }

  // Build mips

  for (int i = 0; i < manifest_.source_count_; i++)
  {
    for (int j = 0; j < mip_count_; j++)
      mips_host_[i][j] = new uint8_t[mip_dim_[j] * mip_dim_[j] * 4];
  }
  for (int i = 0; i < manifest_.source_count_; i++)
  {
    for (int j = 0; j < mip_count_; j++)
    {
      mips_[i][j].Init(mip_dim_[j], mip_dim_[j], 4);
    }
  }
  tex_prep_.Init(mip_dim_[0], mip_dim_[0], 4);
  tex_filter_.Init(mip_dim_[0], mip_dim_[0], 4);
  manifest_loaded_ = true;

  for (int i = 0; i < manifest_.source_count_; i++)
  {
    int w;
    int h;
    int c;
    int desired_channels = 4;
    uint8_t* tex_data = stbi_load(manifest_.sources_[i].path_.c_str(), &w, &h, &c, desired_channels);
    if (tex_data == nullptr)
    {
      UnloadManifest();
      return Result::FileNotFound;
    }
    if (w != h || w != manifest_.dim_)
    {
      stbi_image_free(tex_data);
      UnloadManifest();
      return Result::InvalidManifest;
    }
    cudaMemcpy(tex_prep_.DevicePtr(), tex_data, w * h * desired_channels * sizeof(uint8_t), cudaMemcpyHostToDevice);
    stbi_image_free(tex_data);

    PrepareTexInput prepare_in = {};
    for (int j = 0; j < 4; j++) prepare_in.cmap[j] = static_cast<int32_t>(manifest_.sources_[i].channel_mapping_[j]);
    launch_prepare_tex(manifest_.dim_, desired_channels, prepare_in, tex_prep_.DevicePtr(), mips_[i][0].DevicePtr());
    for (int j = 1; j < mip_count_; j++)
    {
      launch_filter_lanczos(
        mip_dim_[j - 1],
        manifest_.sources_[i].num_channels_,
        3,
        mips_[i][j - 1].DevicePtr(),
        tex_filter_.DevicePtr(),
        mips_[i][j].DevicePtr());
    }
    for (int j = 0; j < mip_count_; j++)
      cudaMemcpy(
        mips_host_[i][j],
        mips_[i][j].DevicePtr(),
        sizeof(uint8_t) * mip_dim_[j] * mip_dim_[j] * manifest_.sources_[i].num_channels_,
        cudaMemcpyDeviceToHost);
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
  {
    UnloadManifest();
    return Result::InvalidManifest;
  }

  for (int i = 0; i < mip_count_; i++)
  {
    for (int j = 0; j < manifest_.source_count_; j++)
      package_in.tex[j] = mips_[j][i].DevicePtr();
    launch_package_tex(mip_dim_[i], out_dim_, package_in, package_[i].DevicePtr());
  }

  manifest_loaded_ = true;
  train_phase_ = TrainPhase::ManifestLoaded;
  return Result::Success;
}

void Context::UnloadManifest()
{
  if (!manifest_loaded_)
    return;

  // TODO: Implement fast path LoadNewManifest where these aren't destroyed if dim doesn't change

  for (int i = 0; i < level_count_; i++)
  {
    g0_[i].Destroy();
    g1_[i].Destroy();
  }

  g0_noise_.Destroy();
  g1_noise_.Destroy();

  for (int level_i = 0; level_i < level_count_; level_i++)
  {
    dLdG0_[level_i].Destroy();
    dLdG1_[level_i].Destroy();
  }
  
  for (int level_i = 0; level_i < level_count_; level_i++)
  {
    mG0_[level_i].Destroy();
    vG0_[level_i].Destroy();
    mG1_[level_i].Destroy();
    vG1_[level_i].Destroy();
  }

  for (int i = 0; i < mip_count_; i++)
  {
    package_[i].Destroy();
  }

  for(int i = 0; i < level_count_; i++)
  {
    delete[] g0_host_[i];
    delete[] g1_host_[i];
  }

  for (int i = 0; i < manifest_.source_count_; i++)
  {
    for (int j = 0; j < mip_count_; j++)
      delete[] mips_host_[i][j];
  }
  for (int i = 0; i < manifest_.source_count_; i++)
  {
    for (int j = 0; j < mip_count_; j++)
      mips_[i][j].Destroy();
  }
  tex_prep_.Destroy();
  tex_filter_.Destroy();

  manifest_loaded_ = false;
}

TextureData Context::GetTextureData()
{
  TextureData tex_data = {};
  if (!manifest_loaded_)
    return tex_data;

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

void Context::Destroy()
{
  if (!initialized_)
    return;

  if (manifest_loaded_)
  {
    UnloadManifest();
  }

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

  mse_.Destroy();

  mW0_.Destroy();
  vW0_.Destroy();
  mW1_.Destroy();
  vW1_.Destroy();
  mWout_.Destroy();
  vWout_.Destroy();

  grid_draws_.Destroy();
  x_.Destroy();
  
  cublasDestroy((cublasHandle_t)handle_);
  cudaFree(rstate_);
  rstate_ = nullptr;

  delete[] decoder_host_;
  decoder_host_ = nullptr;

  initialized_ = false;
}

int32_t Context::GetMipDim(int mip) const
{
  assert(initialized_ && manifest_loaded_);
  if (initialized_ && manifest_loaded_)
    return mip_dim_[mip];
  return 0;
}

struct Blob
{
  std::string name;
  uint64_t data_size;
  void* data;
};

static const char* ProfileToString(Profile p)
{
  switch (p)
  {
    case Profile::Bpp_0_2:
      return "bpp_0_2";
    case Profile::Bpp_0_5:
      return "bpp_0_5";
    case Profile::Bpp_1_0:
      return "bpp_1_0";
    case Profile::Bpp_2_25:
      return "bpp_2_25";
  }
  return "unknown";
}

static bool ProfileFromString(const std::string& s, Profile& o_profile)
{
  if (s == "bpp_0_2")
  {
    o_profile = Profile::Bpp_0_2;
    return true;
  }
  else if (s == "bpp_0_5")
  {
    o_profile = Profile::Bpp_0_5;
    return true;
  }
  else if (s == "bpp_1_0")
  {
    o_profile = Profile::Bpp_1_0;
    return true;
  }
  else if (s == "bpp_2_25")
  {
    o_profile = Profile::Bpp_2_25;
    return true;
  }
  return false;
}

void FillNTCConstants(const CompressedData& data, NTCConstants& o_constants)
{
  o_constants = {};
  for (int i = 0; i < data.level_count_; i++)
  {
    o_constants.g0_grid_dim_[i] = data.g0_grid_dim_[i];
    o_constants.g1_grid_dim_[i] = data.g1_grid_dim_[i];
    o_constants.g0_offset_[i] = static_cast<uint32_t>(data.g0_offset_[i]);
    o_constants.g1_offset_[i] = static_cast<uint32_t>(data.g1_offset_[i]);
  }
  o_constants.g0_bits_per_channel_ = data.g0_bits_per_channel_;
  o_constants.g1_bits_per_channel_ = data.g1_bits_per_channel_;
  o_constants.g0_channels_ = data.g0_channels_;
  o_constants.g1_channels_ = data.g1_channels_;
  o_constants.dim_ = data.dim_;
  o_constants.mip_count_ = data.mip_count_;
  o_constants.rcp_s_a1_ = 1.0f / data.caldata_.s_a1_;
  o_constants.rcp_s_a2_ = 1.0f / data.caldata_.s_a2_;
  o_constants.channel_count_ = data.channel_count_;
  for (int i = 0; i < data.channel_count_; i++)
    o_constants.channel_semantics_[i] = (uint32_t)data.channel_semantics_[i];
}

Result Context::Dump(const std::string& path, const CompressedData& data)
{
  std::vector<Blob> blobs;
  for (int i = 0; i < data.level_count_; i++)
    blobs.push_back({std::format("g0_{}", i), data.g0_size_[i], data.g0_[i]});
  for (int i = 0; i < data.level_count_; i++)
    blobs.push_back({std::format("g1_{}", i), data.g1_size_[i], data.g1_[i]});
  blobs.push_back({"decoder", data.decoder_size_, data.decoder_});

  nlohmann::json j;
  j["source"] = {{"generator", "openntc"}, {"version", 5}};
  j["profile"] = ProfileToString(data.profile_);
  {
    nlohmann::json jchannels = nlohmann::json::array();
    for (int i = 0; i < data.channel_count_; i++)
      jchannels.push_back(SemanticToString(data.channel_semantics_[i]));
    j["channel_semantics"] = jchannels;
  }
  j["dim"] = data.dim_;
  j["mip_count"] = data.mip_count_;
  j["level_count"] = data.level_count_;
  j["g0"] = {
    {"grid_dims",
      {data.g0_grid_dim_[0], data.g0_grid_dim_[1], data.g0_grid_dim_[2], data.g0_grid_dim_[3], data.g0_grid_dim_[4], data.g0_grid_dim_[5]}},
    {"bits", data.g0_bits_per_channel_},
    {"channels", data.g0_channels_}
  };
  j["g1"] = {
    {"grid_dims",
      {data.g1_grid_dim_[0], data.g1_grid_dim_[1], data.g1_grid_dim_[2], data.g1_grid_dim_[3], data.g1_grid_dim_[4], data.g1_grid_dim_[5]}},
    {"bits", data.g1_bits_per_channel_},
    {"channels", data.g1_channels_}
  };
  j["calibration"] = {
    {"max_abs_a1", data.caldata_.max_abs_a1_},
    {"max_abs_a2", data.caldata_.max_abs_a2_},
    {"s_a1", data.caldata_.s_a1_},
    {"s_a2", data.caldata_.s_a2_}
  };

  uint64_t off = 0;
  for (int i = 0; i < blobs.size(); i++)
  {
    off = RoundUpToNearestK(off, 256);
    j["blobs"].push_back({{"name", blobs[i].name}, {"size", blobs[i].data_size}, {"offset", off}});
    off += blobs[i].data_size;
  }

  std::string js = j.dump();
  // 4 byte magic word: ONTC = 0x43544E4F
  uint32_t header[4] = {0x43544E4F, 4, (uint32_t)js.size(), (uint32_t)off};

  std::ofstream f(path, std::ios::binary);
  if (!f) return Result::FileNotFound;
  f.write((const char*)header, sizeof(header));
  f.write(js.data(), js.length());
  off = 0;
  for (int i = 0; i < blobs.size(); i++)
  {
    uint64_t new_off = RoundUpToNearestK(off, 256);
    for (; off < new_off; off++)
      f.put(0);
    f.write((const char*)blobs[i].data, blobs[i].data_size);
    off += blobs[i].data_size;
  }

  Result res = Result::FileWriteFailure;
  if (f.good())
    res = Result::Success;
  f.close();

  return res;
}

template <typename T>
static bool TryGet(const nlohmann::json& j, const char* key, T& o_val)
{
  auto it = j.find(key);
  if (it == j.end())
    return false;
  
  if constexpr (std::is_integral_v<T>)
  {
    if (!it->is_number_integer())
      return false;
  }
  else if constexpr (std::is_floating_point_v<T>) {
    if (!it->is_number())
      return false;
  }
  else if constexpr (std::is_same_v<T, std::string>)
  {
    if (!it->is_string())
      return false;
  }
  it->get_to(o_val);
  return true;
}

template <typename T>
static bool TryGetArray(const nlohmann::json& j, const char* key, T* o_vals, int n)
{
  auto it = j.find(key);
  if (it == j.end() || !it->is_array() || (int)it->size() != n) return false;
  for (int i = 0; i < n; i++)
  {
    if (!(*it)[i].is_number_integer()) return false;
    o_vals[i] = (*it)[i];
  }
  return true;
}

static const nlohmann::json* TryGetObj(const nlohmann::json& j, const char* key)
{
  auto it = j.find(key);
  return (it != j.end() && it->is_object()) ? &*it : nullptr;
}

struct BlobSlot
{
  std::string name_;
  void** dst_;
  size_t* size_;
  bool found_;
};

Result Context::Load(const std::string& path, FileData& o_data)
{
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f)
    return Result::FileNotFound;
  size_t raw_size = f.tellg();
  o_data.raw_ = new uint8_t[raw_size];
  f.seekg(0);
  f.read((char*)o_data.raw_, raw_size);

  // Validate header
  if (raw_size < 16)
    return Result::InvalidFile;
  uint32_t* header = (uint32_t*)o_data.raw_;
  if (header[0] != 0x43544E4F || header[1] != 4)
    return Result::InvalidFile;

  // Validate json
  uint64_t header_size = sizeof(uint32_t) * 4;
  uint64_t json_size = header[2];
  uint64_t blob_size = header[3];
  uint64_t fil_size = header_size + json_size + blob_size;
  uint64_t blob_off = json_size + header_size;
  if (header_size + json_size > fil_size)
    return Result::InvalidFile;
  nlohmann::json j = nlohmann::json::parse(
    o_data.raw_ + header_size,
    o_data.raw_ + header_size + json_size,
    nullptr,
    false);
  if (j.is_discarded())
    return Result::InvalidFile;

  // Parse metadata from json
  const nlohmann::json* jg0 = TryGetObj(j, "g0");
  const nlohmann::json* jg1 = TryGetObj(j, "g1");
  const nlohmann::json* jcal = TryGetObj(j, "calibration");
  if (!jg0 || !jg1 || !jcal)
    return Result::InvalidFile;

  bool success = true;
  success &= TryGetArray(*jg0, "grid_dims", o_data.data_.g0_grid_dim_, Context::kMaxLevels);
  success &= TryGet(*jg0, "bits", o_data.data_.g0_bits_per_channel_);
  success &= TryGet(*jg0, "channels", o_data.data_.g0_channels_);
  success &= TryGetArray(*jg1, "grid_dims", o_data.data_.g1_grid_dim_, Context::kMaxLevels);
  success &= TryGet(*jg1, "bits", o_data.data_.g1_bits_per_channel_);
  success &= TryGet(*jg1, "channels", o_data.data_.g1_channels_);
  success &= TryGet(j, "dim", o_data.data_.dim_);
  success &= TryGet(j, "mip_count", o_data.data_.mip_count_);
  success &= TryGet(j, "level_count", o_data.data_.level_count_);
  success &= TryGet(*jcal, "max_abs_a1", o_data.data_.caldata_.max_abs_a1_);
  success &= TryGet(*jcal, "s_a1", o_data.data_.caldata_.s_a1_);
  success &= TryGet(*jcal, "max_abs_a2", o_data.data_.caldata_.max_abs_a2_);
  success &= TryGet(*jcal, "s_a2", o_data.data_.caldata_.s_a2_);
  if (!success)
    return Result::InvalidFile;

  std::string profile_name;
  if (!TryGet(j, "profile", profile_name) || !ProfileFromString(profile_name, o_data.data_.profile_))
    return Result::InvalidFile;

  {
    auto it = j.find("channel_semantics");
    if (it == j.end() || !it->is_array() || (int)it->size() > Context::kMaxChannels)
      return Result::InvalidFile;
    o_data.data_.channel_count_ = (int)it->size();
    for (int i = 0; i < o_data.data_.channel_count_; i++)
    {
      if (!(*it)[i].is_string())
        return Result::InvalidFile;
      Semantic sem = SemanticFromName((*it)[i].get<std::string>());
      if (sem == Semantic::None)
        return Result::InvalidFile;
      o_data.data_.channel_semantics_[i] = sem;
    }
  }

  if (o_data.data_.level_count_ < 1 ||
      o_data.data_.level_count_ > Context::kMaxLevels ||
      o_data.data_.mip_count_ < 1 ||
      o_data.data_.mip_count_ > Context::kMaxMips)
  {
    return Result::InvalidFile;
  }

  std::vector<BlobSlot> slots;
  for (int i = 0; i < o_data.data_.level_count_; i++)
    slots.push_back({std::format("g0_{}", i), (void**)&o_data.data_.g0_[i], &o_data.data_.g0_size_[i], false});
  for (int i = 0; i < o_data.data_.level_count_; i++)
    slots.push_back({std::format("g1_{}", i), (void**)&o_data.data_.g1_[i], &o_data.data_.g1_size_[i], false});
  slots.push_back({"decoder", (void**)&o_data.data_.decoder_, &o_data.data_.decoder_size_, false});
  
  const nlohmann::json* jblobs = nullptr;
  {
    auto it = j.find("blobs");
    if (it == j.end() || !it->is_array())
      return Result::InvalidFile;
    jblobs = &*it;
  }

  for (const auto& jb : *jblobs)
  {
    std::string name;
    uint64_t off = 0, size = 0;
    if (!TryGet(jb, "name", name) || !TryGet(jb, "offset", off) || !TryGet(jb, "size", size))
      return Result::InvalidFile;
    if (blob_off + off + size > fil_size)
      return Result::InvalidFile;
    for (int slot_i = 0; slot_i < slots.size(); slot_i++)
    {
      BlobSlot& s = slots[slot_i];
      if (name == s.name_)
      {
        *s.dst_ = o_data.raw_ + blob_off + off;
        *s.size_ = size;
        s.found_ = true;
      }
    }
  }

  for (int slot_i = 0; slot_i < slots.size(); slot_i++)
  {
    if (slots[slot_i].found_ == false)
      return Result::InvalidFile;
  }

  o_data.data_.g0_offset_[0] = 0;
  o_data.data_.g1_offset_[0] = 0;
  for (int level_i = 1; level_i < o_data.data_.level_count_; level_i++)
  {
    o_data.data_.g0_offset_[level_i] = o_data.data_.g0_size_[level_i - 1] + o_data.data_.g0_offset_[level_i - 1];
    o_data.data_.g1_offset_[level_i] = o_data.data_.g1_size_[level_i - 1] + o_data.data_.g1_offset_[level_i - 1];
  }

  return Result::Success;
}

FileData::FileData() : raw_(nullptr) {}
FileData::~FileData()
{
  delete[] raw_;
}
const CompressedData& FileData::Data()
{
  return data_;
}
} // namespace openntc