#pragma once

#include <cstdint>
#include <string>
#include <vector>

class DataPack
{
public:
  struct Blob
  {
    const void* data_;
    size_t size_;
  };

  bool Load(const std::wstring& path);
  Blob Get(const char* name) const;

private:
  struct Entry
  {
    char name_[64];
    uint64_t offset_;
    uint64_t size_;
  };

  uint8_t* raw_ = nullptr;
  const Entry* entries_ = nullptr;
  uint32_t count_ = 0;
};

std::wstring GetExeRelativePath(const wchar_t* filename);
