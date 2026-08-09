#include <random>
#include <atomic>

#include <cublas_v2.h>
#include <curand_kernel.h>

enum class OpenNTCProfile
{
  Bpp_0_2,
};

struct OpenNTCContextInitInfo
{
  OpenNTCProfile profile;
  int dim;
};

enum class OpenNTCResult
{
  Success,

  InvalidDimension,
  AllocationFailure,
};

class Tensor2d
{
public:
  Tensor2d();
  ~Tensor2d();

  OpenNTCResult Init(int x, int y);
  OpenNTCResult InitLike(const Tensor2d& t0);
  void FillZero();
  void FillKaiming(std::mt19937& gen);
  void Destroy();
  float* DevicePtr();
  size_t SizeBytes() const;
  size_t NumElems() const;

  private:
    bool initialized_;
    int shape_[2];
    float* dev_;
};

class Tensor3d
{
public:
  Tensor3d();
  ~Tensor3d();

  OpenNTCResult Init(int x, int y, int z);
  OpenNTCResult InitLike(const Tensor3d& t0);
  void FillZero();
  void FillUniform(std::mt19937& gen, float lb, float ub);
  void Destroy();
  float* DevicePtr();
  float** DeviceDPtr();
  bool IsInitialized() const;
  size_t SizeBytes() const;
  size_t NumElems() const;

  private:
    bool initialized_;
    int shape_[3];
    float* dev_;
};

class IntTensor1d
{
public:
  IntTensor1d();
  ~IntTensor1d();

  OpenNTCResult Init(int x);
  void Destroy();
  int* DevicePtr();

  private:
    bool initialized_;
    int shape_[1];
    int* dev_;
};

struct OpenNTCEvalResults
{
  double mse;
  double psnr;
};

class OpenNTCTrainProgress
{
public:
  int phase; // 0: initial, 1: received package, 2: training, 3: finished training
  int step;
  int total_steps;
};

struct OpenNTCCompressedData
{
  uint16_t* g0_[4];
  float* g1_[4];
  float* W0_;
  float* W1_;
  float* Wout_;

  size_t g0_size_[4];
  size_t g1_size_[4];
  size_t W0_size_;
  size_t W1_size_;
  size_t Wout_size_;
  
  int g0_grid_dim_[4];
  int g1_grid_dim_[4];
  int g0_bytes_per_channel_;
  int g1_bytes_per_channel_;
  int g0_channels_;
  int g1_channels_;
  int dim_;
};

class OpenNTCContext
{
public:
  OpenNTCContext();
  ~OpenNTCContext();

  OpenNTCContext(const OpenNTCContext&) = delete;
  OpenNTCContext& operator=(const OpenNTCContext&) = delete;

  OpenNTCResult Init(const OpenNTCContextInitInfo& init_info);
  void Destroy();
  void LoadPackage(void* handle, long long size, int mip);
  void Train(std::atomic<OpenNTCTrainProgress>& progress);
  OpenNTCEvalResults Eval();
  OpenNTCCompressedData GetCompressedData();

  // Host-side parameters after training

  float* g0_host_[4];
  uint16_t* g0_host_16_[4];
  float* g1_host_[4];
  float* W0_host_;
  float* W1_host_;
  float* Wout_host_;

private:
  static const int kMinDimension = 1024;
  static const int kMaxDimension = 1024;
  static const int kMaxMips = 9;

  bool initialized_;
  int g0_bytes_per_channel_;
  int g1_bytes_per_channel_;
  float g0_delta_;
  float g1_delta_;
  int g0_channels_;
  int g1_channels_;
  int mip_count_;
  int mip_dim_[OpenNTCContext::kMaxMips];
  int g0_grid_dim_[4];
  int g1_grid_dim_[4];
  int feature_dim_;
  int out_dim_;
  int max_batch_;
  int max_batch_dim_;
  int hidden_dim_;

  std::mt19937 gen_;
  cublasHandle_t handle_;
  curandState* rstate_;

  // Feature grids

  Tensor3d g0_[4];
  Tensor3d g1_[4];

  // Quantization Noise

  Tensor3d g0_noise_;
  Tensor3d g1_noise_;

  // Decoder

  Tensor2d W0_;
  Tensor2d W1_;
  Tensor2d Wout_;

  // Forward pass

  Tensor2d W0x_;
  Tensor2d W0xa_;
  Tensor2d W1x_;
  Tensor2d W1xa_;
  Tensor2d Woutx_;

  // Backward pass

  Tensor2d dLdWoutx_;
  Tensor2d dLdWout_;
  Tensor2d dLdW1xa_;
  Tensor2d dLdW1x_;
  Tensor2d dLdW1_;
  Tensor2d dLdW0xa_;
  Tensor2d dLdW0x_;
  Tensor2d dLdW0_;
  Tensor2d dLdx_;
  Tensor3d dLdG0_[4];
  Tensor3d dLdG1_[4];

  // Loss

  Tensor2d mse_;

  // Adam parameters

  Tensor3d mG0_[4];
  Tensor3d vG0_[4];
  Tensor3d mG1_[4];
  Tensor3d vG1_[4];
  Tensor2d mW0_;
  Tensor2d vW0_;
  Tensor2d mW1_;
  Tensor2d vW1_;
  Tensor2d mWout_;
  Tensor2d vWout_;

  // Training

  IntTensor1d grid_draws_;
  Tensor2d x_;
  Tensor3d package_[OpenNTCContext::kMaxMips];
  cudaExternalMemory_t extmem_[OpenNTCContext::kMaxMips];
};