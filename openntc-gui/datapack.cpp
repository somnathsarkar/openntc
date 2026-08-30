#include <openntc-gui/datapack.h>

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

#include <cassert>
#include <cstring>
#include <fstream>

bool DataPack::Load(const std::wstring& path)
{
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f)
    return false;
  size_t size = f.tellg();
  raw_ = new uint8_t[size];
  f.seekg(0);
  f.read((char*)raw_, size);
  if (!f || size < 16)
    return false;

  const uint32_t* header = (const uint32_t*)raw_;
  if (header[0] != 0x4B41504F || header[1] != 1)
    return false;
  count_ = header[2];
  if (16 + sizeof(Entry) * count_ > size)
    return false;
  entries_ = (const Entry*)(raw_ + 16);
  for (uint32_t i = 0; i < count_; i++)
  {
    if (entries_[i].offset_ + entries_[i].size_ > size)
      return false;
  }
  return true;
}

DataPack::Blob DataPack::Get(const char* name) const
{
  for (uint32_t i = 0; i < count_; i++)
  {
    if (strncmp(entries_[i].name_, name, sizeof(entries_[i].name_)) == 0)
      return {raw_ + entries_[i].offset_, entries_[i].size_};
  }
  assert(false && "Blob not found in data.bin");
  return {nullptr, 0};
}

std::wstring GetExeRelativePath(const wchar_t* filename)
{
  wchar_t buf[MAX_PATH] = {};
  GetModuleFileNameW(nullptr, buf, MAX_PATH);
  std::wstring path(buf);
  size_t slash = path.find_last_of(L"\\/");
  return path.substr(0, slash + 1) + filename;
}
