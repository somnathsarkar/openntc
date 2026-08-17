#include <random>
#include <atomic>

#include <cublas_v2.h>
#include <curand_kernel.h>

namespace openntc
{
enum class Profile
{
  Bpp_0_2,
};

struct ContextInitInfo
{
  Profile profile;
};

enum class Result
{
  Success,

  InvalidDimension,
  AllocationFailure,
  InvalidManifest,
  FileNotFound,
  InvalidFile,
  FileWriteFailure,
  InvalidState,
};

class Tensor2d
{
public:
  Tensor2d();
  ~Tensor2d();

  Result Init(int x, int y);
  Result InitLike(const Tensor2d& t0);
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

  Result Init(int x, int y, int z);
  Result InitLike(const Tensor3d& t0);
  void FillZero();
  void FillUniform(std::mt19937& gen, float lb, float ub);
  void Destroy();
  float* DevicePtr();
  bool IsInitialized() const;
  size_t SizeBytes() const;
  size_t NumElems() const;

  private:
    bool initialized_;
    int shape_[3];
    float* dev_;
};

class U8Tensor3d
{
public:
  U8Tensor3d();
  ~U8Tensor3d();

  Result Init(int x, int y, int z);
  Result InitLike(const U8Tensor3d& t0);
  void FillZero();
  void Destroy();
  uint8_t* DevicePtr();
  bool IsInitialized() const;
  size_t SizeBytes() const;
  size_t NumElems() const;

  private:
    bool initialized_;
    int shape_[3];
    uint8_t* dev_;
};

class IntTensor1d
{
public:
  IntTensor1d();
  ~IntTensor1d();

  Result Init(int x);
  void Destroy();
  int* DevicePtr();

  private:
    bool initialized_;
    int shape_[1];
    int* dev_;
};

struct EvalResults
{
  double mse;
  double psnr;
};

struct Calibration
{
  float max_abs_a1;
  float max_abs_a2;
  float s_a1;
  float s_a2;
};

enum TrainPhase : int32_t
{
  ManifestLoaded = 0,
  TrainInProgress = 1,
  TrainComplete = 2,

  TrainError = -1,
};

struct TrainProgress
{
  TrainPhase phase_;
  Result result_;
  int batches_complete_;
  int total_batches_;
};

enum class Semantic : int32_t
{
  None = 0,

  Albedo = 1,
  Alpha = 2,
  Displacement = 3,
  Emissive = 4,
  Gloss = 5,
  Metallic = 6,
  Normal = 7,
  AO = 8,
  Roughness = 9,
  Specular = 10,
  Transmission = 11,

  Count = 12,
};

enum class Channel : int32_t
{
  R = 0,
  G = 1,
  B = 2,
  A = 3,

  Count,
  Invalid = -1
};

struct TextureSource
{
  std::string path_;
  std::string name_;
  bool is_srgb_;
  Semantic semantic_;
  bool vertical_flip_;
  int32_t num_channels_;
  Channel channel_mapping_[4];
};

constexpr int32_t kMaxSources = 16;

struct Manifest
{
  TextureSource sources_[kMaxSources];
  int32_t source_count_;
  int32_t dim_;
};

struct TrainInfo
{
  int batch_count_;
  int grids_per_batch_;
};

class CompressedData;
class FileData;
struct TextureData;

class Context
{
public:
  static const int kMinDimension = 1024;
  static const int kMaxDimension = 8192;
  static const int kMaxMips = 12;
  static const int kMaxChannels = 16;
  static const int kMaxLevels = 5;

  Context();
  ~Context();

  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;

  Result Init(const ContextInitInfo& init_info);
  void Destroy();
  void BeginTraining(const TrainInfo& train_info);
  TrainProgress Train(int32_t num_batches);
  TrainProgress TrainUntilComplete();
  EvalResults Eval();
  Calibration Calibrate(float headroom = 1.1f);
  CompressedData GetCompressedData();
  Result LoadManifest(const std::string& filepath);
  void UnloadManifest();
  TextureData GetTextureData();
  int32_t GetMipDim(int mip) const;

  static Result Dump(const std::string& path, const CompressedData& data);
  static Result Load(const std::string& path, FileData& data);

  // Host-side parameters after training

  uint32_t* g0_host_[kMaxLevels];
  uint32_t* g1_host_[kMaxLevels];
  uint32_t* W0_host_;
  uint32_t* W1_host_;
  uint32_t* Wout_host_;
  float* W0_scale_;
  float* W1_scale_;
  float* Wout_scale_;
  Calibration caldata_;

  uint8_t* mips_host_[kMaxSources][kMaxMips];

private:

  bool initialized_;
  bool manifest_loaded_;
  int g0_bytes_per_channel_;
  int g1_bytes_per_channel_;
  float g0_delta_;
  float g1_delta_;
  int g0_channels_;
  int g1_channels_;
  int mip_count_;
  int mip_dim_[Context::kMaxMips];
  int level_count_;
  int g0_grid_dim_[Context::kMaxLevels];
  int g1_grid_dim_[Context::kMaxLevels];
  int feature_dim_;
  int feature_dim_padded_;
  int out_dim_;
  int out_dim_padded_;
  int max_batch_;
  int max_batch_dim_;
  int hidden_dim_;
  int rand_dim_;

  // Per training run

  int batch_count_;
  int batch_i_;
  int lock_i_;
  int grids_per_batch_;
  int grid_batch_i_[Context::kMaxLevels];
  TrainPhase train_phase_;

  std::mt19937 gen_;
  cublasHandle_t handle_;
  curandState* rstate_;

  // Feature grids

  Tensor3d g0_[Context::kMaxLevels];
  Tensor3d g1_[Context::kMaxLevels];

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
  Tensor3d dLdG0_[Context::kMaxLevels];
  Tensor3d dLdG1_[Context::kMaxLevels];

  // Loss

  Tensor2d mse_;

  // Adam parameters

  Tensor3d mG0_[Context::kMaxLevels];
  Tensor3d vG0_[Context::kMaxLevels];
  Tensor3d mG1_[Context::kMaxLevels];
  Tensor3d vG1_[Context::kMaxLevels];
  Tensor2d mW0_;
  Tensor2d vW0_;
  Tensor2d mW1_;
  Tensor2d vW1_;
  Tensor2d mWout_;
  Tensor2d vWout_;

  // Training

  IntTensor1d grid_draws_;
  Tensor2d x_;
  U8Tensor3d package_[Context::kMaxMips];

  // File management

  Manifest manifest_;
  U8Tensor3d mips_[kMaxSources][Context::kMaxMips];
  U8Tensor3d tex_prep_;
  Tensor3d tex_filter_;
};

struct TextureData
{
  int32_t tex_count_;
  int32_t mip_count_;
  int32_t channels_[kMaxSources];
  Semantic semantics_[kMaxSources];
  uint8_t* mips_[kMaxSources][Context::kMaxMips];
};

struct CompressedData
{
  uint32_t* g0_[Context::kMaxLevels];
  uint32_t* g1_[Context::kMaxLevels];
  uint32_t* W0_;
  uint32_t* W1_;
  uint32_t* Wout_;

  float* W0_scale_;
  float* W1_scale_;
  float* Wout_scale_;

  size_t g0_size_[Context::kMaxLevels];
  size_t g1_size_[Context::kMaxLevels];
  size_t g0_offset_[Context::kMaxLevels];
  size_t g1_offset_[Context::kMaxLevels];
  size_t W0_size_;
  size_t W1_size_;
  size_t Wout_size_;
  size_t W0_scale_size_;
  size_t W1_scale_size_;
  size_t Wout_scale_size_;
  
  int g0_grid_dim_[Context::kMaxLevels];
  int g1_grid_dim_[Context::kMaxLevels];
  int g0_bytes_per_channel_;
  int g1_bytes_per_channel_;
  int g0_channels_;
  int g1_channels_;
  int dim_;
  int mip_count_;
  int level_count_;

  Calibration caldata_;
};

class FileData
{
  friend Result Context::Load(const std::string& path, FileData& o_data);

public:
  FileData();
  ~FileData();

  // Uncopyable
  FileData(const FileData& cd) = delete;
  FileData& operator=(const FileData& cd) = delete;

  const CompressedData& Data();

private:
  CompressedData data_;
  uint8_t* raw_;
};
} // namespace openntc