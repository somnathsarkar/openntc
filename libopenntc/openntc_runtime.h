#pragma once

#include <cstdint>
#include <string>

namespace openntc
{
enum class Profile
{
  Bpp_0_2,
  Bpp_0_5,
  Bpp_1_0,
  Bpp_2_25,

  Count
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

constexpr int32_t kMaxMips = 12;
constexpr int32_t kMaxChannels = 16;
constexpr int32_t kMaxLevels = 6;

const char* ProfileToString(Profile p);
bool ProfileFromString(const std::string& s, Profile& o_profile);
const char* SemanticToString(Semantic sem);
Semantic SemanticFromName(const std::string& s);
int GetChannelCountForSemantic(const Semantic sem);

struct CalibrationData
{
  float max_abs_a1_;
  float max_abs_a2_;
  float s_a1_;
  float s_a2_;
};

struct NTCConstants
{
  int32_t g0_grid_dim_[8];
  int32_t g1_grid_dim_[8];
  uint32_t g0_offset_[8];
  uint32_t g1_offset_[8];
  int32_t g0_bits_per_channel_;
  int32_t g1_bits_per_channel_;
  int32_t g0_channels_;
  int32_t g1_channels_;
  int32_t dim_;
  int32_t mip_count_;
  float rcp_s_a1_;
  float rcp_s_a2_;
  int32_t channel_count_;
  int32_t pad0_[3];
  uint32_t channel_semantics_[16];
};

// Update reminder: Change matching struct in ntc_decode.hlsli
static_assert(sizeof(NTCConstants) == 240);

struct CompressedData
{
  uint32_t* g0_[kMaxLevels];
  uint32_t* g1_[kMaxLevels];
  void* decoder_;

  size_t g0_size_[kMaxLevels];
  size_t g1_size_[kMaxLevels];
  size_t g0_offset_[kMaxLevels];
  size_t g1_offset_[kMaxLevels];
  size_t decoder_size_;

  int g0_grid_dim_[kMaxLevels];
  int g1_grid_dim_[kMaxLevels];
  int g0_bits_per_channel_;
  int g1_bits_per_channel_;
  int g0_channels_;
  int g1_channels_;
  int dim_;
  int mip_count_;
  int level_count_;
  int channel_count_;
  Semantic channel_semantics_[kMaxChannels];

  // Compression profile this data was produced with
  Profile profile_;

  CalibrationData caldata_;
};

void FillNTCConstants(const CompressedData& data, NTCConstants& o_constants);

class FileData;

// ONTC container serialization
Result Dump(const std::string& path, const CompressedData& data);
Result Load(const std::string& path, FileData& o_data);

class FileData
{
  friend Result Load(const std::string& path, FileData& o_data);

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
