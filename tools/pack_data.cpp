// Pack all binary assets required for UI in a single .bin file, similar to our .ntc binaries.
// Current contents:
//  1) all shader .csos, 
//  2) two images corresponding to the split-sum specular IBL and diffuse sh data from cmgen
//  3) Public domain Mitsuba Knob model data

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

struct PackEntry
{
  char name_[64];
  uint64_t offset_;
  uint64_t size_;
};

static uint64_t RoundUpTo(uint64_t n, uint64_t k)
{
  return ((n + k - 1) / k) * k;
}

int main(int argc, char** argv)
{
  if (argc < 3)
  {
    return 1;
  }

  std::vector<PackEntry> entries;
  std::vector<std::string> paths;
  for (int i = 2; i < argc; i++)
  {
    const char* sep = strchr(argv[i], '=');
    if (!sep || sep == argv[i] || strlen(argv[i]) == 0)
      return 1;
    PackEntry e = {};
    size_t name_len = sep - argv[i];
    if (name_len >= sizeof(e.name_))
      return 1;
    memcpy(e.name_, argv[i], name_len);
    entries.push_back(e);
    paths.push_back(sep + 1);
  }

  // Sizes and offsets
  uint64_t header_size = 16 + sizeof(PackEntry) * entries.size();
  uint64_t off = RoundUpTo(header_size, 256);
  for (size_t i = 0; i < entries.size(); i++)
  {
    std::ifstream f(paths[i], std::ios::binary | std::ios::ate);
    if (!f)
    {
      return 1;
    }
    entries[i].size_ = f.tellg();
    entries[i].offset_ = off;
    off = RoundUpTo(off + entries[i].size_, 256);
  }

  std::ofstream out(argv[1], std::ios::binary);
  if (!out)
    return 1;
  
  // Magic word for this pack: OPAK
  uint32_t header[4] = {0x4B41504F, 1, (uint32_t)entries.size(), 0};
  out.write((const char*)header, sizeof(header));
  out.write((const char*)entries.data(), sizeof(PackEntry) * entries.size());

  uint64_t written = header_size;
  for (size_t i = 0; i < entries.size(); i++)
  {
    for (; written < entries[i].offset_; written++)
      out.put(0);
    std::ifstream f(paths[i], std::ios::binary);
    std::vector<char> buf((size_t)entries[i].size_);
    f.read(buf.data(), buf.size());
    if (!f)
      return 1;
    out.write(buf.data(), buf.size());
    written += entries[i].size_;
  }

  if (!out.good())
    return 1;
  return 0;
}
