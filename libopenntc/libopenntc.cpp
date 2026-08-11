#include <libopenntc/libopenntc.h>

#include <cassert>
#include <bit>

#include <curand_kernel.h>

#include <libopenntc/ntc_kernel.cuh>

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
  W0_host_ = new uint32_t[W0_.NumElems() / 4];
  W1_host_ = new uint32_t[W1_.NumElems() / 4];
  Wout_host_ = new uint32_t[(out_dim_padded_ * hidden_dim_) / 4];
  W0_scale_ = new float[hidden_dim_];
  W1_scale_ = new float[hidden_dim_];
  Wout_scale_ = new float[out_dim_padded_];

  return OpenNTCResult::Success;
}

void OpenNTCContext::LoadPackage(void* handle, long long size, int mip)
{
  cudaExternalMemoryHandleDesc desc = {};
  desc.type = cudaExternalMemoryHandleTypeD3D12Resource;
  desc.handle.win32.handle = handle;
  desc.size = size;
  desc.flags = cudaExternalMemoryDedicated;
  cudaImportExternalMemory(&extmem_[mip], &desc);
  cudaExternalMemoryBufferDesc mdesc = {};
  mdesc.offset = 0;
  mdesc.flags = 0;
  mdesc.size = size;
  cudaExternalMemoryGetMappedBuffer((void**)package_[mip].DeviceDPtr(), extmem_[mip], &mdesc);
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

void OpenNTCContext::Train(std::atomic<OpenNTCTrainProgress>& progress)
{
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

  std::bernoulli_distribution dist_batch_type(0.05);
  std::uniform_real_distribution<float> dist_u(0.0f, 1.0f);
  std::uniform_int_distribution<int> dist_lod(0, 8);

  int batch_count = 10000;
  int lock_i = 95 * batch_count / 100;
  assert(lock_i > 0);
  int grids_per_batch = 1;
  int grid_batch_i[4] = { 0, 0, 0, 0 };

  for (int batch_i = 0; batch_i < batch_count; batch_i++)
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
    for (int i = 0; i < grids_per_batch; i++)
    {
      grid_draws[i + i] = dist_grid(gen_);
      grid_draws[i + i + 1] = dist_grid(gen_);
    }
    cudaMemcpy(grid_draws_.DevicePtr(), grid_draws, sizeof(int) * grids_per_batch * 2, cudaMemcpyHostToDevice);
    int grid_dim_draw = std::min(mip_dim_[lod], 256);
    if (batch_i < lock_i)
    {
      launch_generate_noise(g0_grid_dim_[feature_level] * g0_grid_dim_[feature_level] * g0_channels_, g0_delta_, rstate_, g0_noise_.DevicePtr());
      launch_generate_noise(g1_grid_dim_[feature_level] * g1_grid_dim_[feature_level] * g1_channels_, g1_delta_, rstate_, g1_noise_.DevicePtr());
    }
    else if (batch_i == lock_i)
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
      grids_per_batch,
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
      grids_per_batch,
      grid_dim_draw,
      mip_dim_[lod],
      out_dim_,
      grid_draws_.DevicePtr(),
      package_[lod].DevicePtr(),
      mse_.DevicePtr());

    int batch_dim = grids_per_batch * grid_dim_draw * grid_dim_draw;

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
      grids_per_batch,
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
    float bias_1 = 1.0f / (1.0f - powf(beta_1, batch_i + 1.0f));
    float bias_2 = 1.0f / (1.0f - powf(beta_2, batch_i + 1.0f));
    float grid_bias_1 = 1.0f / (1.0f - powf(beta_1, grid_batch_i[feature_level] + 1.0f));
    float grid_bias_2 = 1.0f / (1.0f - powf(beta_2, grid_batch_i[feature_level] + 1.0f));
    grid_batch_i[feature_level] += 1;

    float lr_grid = cosine_annealing(0.0f, 0.01f, batch_count, batch_i);
    float lr_decoder = cosine_annealing(0.0f, 0.005f, batch_count, batch_i);

    launch_update_adam(out_dim_ * hidden_dim_, lr_decoder, beta_1, beta_2, bias_1, bias_2, dLdWout_.DevicePtr(), mWout_.DevicePtr(), vWout_.DevicePtr(), Wout_.DevicePtr());
    launch_update_adam(hidden_dim_ * hidden_dim_, lr_decoder, beta_1, beta_2, bias_1, bias_2, dLdW1_.DevicePtr(), mW1_.DevicePtr(), vW1_.DevicePtr(), W1_.DevicePtr());
    launch_update_adam(hidden_dim_ * feature_dim_, lr_decoder, beta_1, beta_2, bias_1, bias_2, dLdW0_.DevicePtr(), mW0_.DevicePtr(), vW0_.DevicePtr(), W0_.DevicePtr());
    if (batch_i < lock_i)
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

    OpenNTCTrainProgress tprogress;
    tprogress.phase = 2;
    tprogress.step = batch_i;
    tprogress.total_steps = batch_count;
    progress.store(tprogress, std::memory_order_relaxed);
  }

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

  float* W0_unpack = new float[W0_.NumElems()];
  float* W1_unpack = new float[W1_.NumElems()];
  float* Wout_unpack = new float[out_dim_padded_ * hidden_dim_];

  cudaMemcpy(W0_unpack, W0_.DevicePtr(), W0_.SizeBytes(), cudaMemcpyDeviceToHost);
  cudaMemcpy(W1_unpack, W1_.DevicePtr(), W1_.SizeBytes(), cudaMemcpyDeviceToHost);
  cudaMemcpy(Wout_unpack, Wout_.DevicePtr(), Wout_.SizeBytes(), cudaMemcpyDeviceToHost);

  for (int i = 0; i < hidden_dim_; i++)
  {
    for (int j = 57; j < feature_dim_; j++)
    {
      W0_unpack[i * feature_dim_ + j] = 0.0f;
    }
  }
  QuantizeWeights(W0_unpack, hidden_dim_, feature_dim_, 1.0f / 128.0f, W0_host_, W0_scale_);
  QuantizeWeights(W1_unpack, hidden_dim_, hidden_dim_, caldata_.s_a1, W1_host_, W1_scale_);
  QuantizeWeights(Wout_unpack, out_dim_, hidden_dim_, caldata_.s_a2, Wout_host_, Wout_scale_);
  for (int i = out_dim_; i < out_dim_padded_; i++) Wout_scale_[i] = 0.0f;

  delete[] W0_unpack;
  delete[] W1_unpack;
  delete[] Wout_unpack;
  
  OpenNTCTrainProgress tprogress;
  tprogress.phase = 3;
  progress.store(tprogress, std::memory_order_relaxed);
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
  data.W0_size_ = W0_.SizeBytes() / 4;
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

  for (int i = 0; i < kMaxMips; i++)
  {
    if (package_[i].IsInitialized())
    {
      cudaFree(package_[i].DevicePtr());
      cudaDestroyExternalMemory(extmem_[i]);
    }
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
}