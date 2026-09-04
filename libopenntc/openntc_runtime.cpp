// ONTC container serialization, decoder constants and semantic tables: the
//  CUDA-free runtime half of libopenntc. See openntc_runtime.h.

#include <libopenntc/openntc_runtime.h>

#include <cstring>
#include <format>
#include <fstream>
#include <vector>

#include <libopenntc/json.hpp>

namespace openntc
{
static uint64_t RoundUpToNearestK(uint64_t n, uint64_t k)
{
  return ((n + k - 1) / k) * k;
}

const char* SemanticToString(Semantic sem)
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

Semantic SemanticFromName(const std::string& s)
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

int GetChannelCountForSemantic(const Semantic sem)
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


struct Blob
{
  std::string name;
  uint64_t data_size;
  void* data;
};

const char* ProfileToString(Profile p)
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

bool ProfileFromString(const std::string& s, Profile& o_profile)
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

Result Dump(const std::string& path, const CompressedData& data)
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

Result Load(const std::string& path, FileData& o_data)
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
  success &= TryGetArray(*jg0, "grid_dims", o_data.data_.g0_grid_dim_, kMaxLevels);
  success &= TryGet(*jg0, "bits", o_data.data_.g0_bits_per_channel_);
  success &= TryGet(*jg0, "channels", o_data.data_.g0_channels_);
  success &= TryGetArray(*jg1, "grid_dims", o_data.data_.g1_grid_dim_, kMaxLevels);
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
    if (it == j.end() || !it->is_array() || (int)it->size() > kMaxChannels)
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
      o_data.data_.level_count_ > kMaxLevels ||
      o_data.data_.mip_count_ < 1 ||
      o_data.data_.mip_count_ > kMaxMips)
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
