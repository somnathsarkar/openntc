#include <openntc/dataloader.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

void cuda_check(cudaError_t err, const char* what)
{
  if (err != cudaSuccess)
  {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
  }
}

// IEEE 754 half -> float, handling subnormals, inf and nan.
float half_to_float(uint16_t h)
{
  uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1Fu;
  uint32_t mant = h & 0x3FFu;
  uint32_t bits;
  if (exp == 0)
  {
    if (mant == 0)
    {
      bits = sign;  // +/- zero
    }
    else
    {
      // subnormal: renormalize
      int e = -1;
      do
      {
        mant <<= 1;
        e++;
      } while (!(mant & 0x400u));
      mant &= 0x3FFu;
      bits = sign | ((uint32_t)(127 - 15 - e) << 23) | (mant << 13);
    }
  }
  else if (exp == 31)
  {
    bits = sign | 0x7F800000u | (mant << 13);  // inf / nan
  }
  else
  {
    bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
  }
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

struct NpyArray
{
  std::vector<int> shape;
  std::vector<float> data;
};

// Minimal npy reader for C-order arrays of dtype <f2 or <f4.
NpyArray load_npy(const std::string& path)
{
  std::ifstream f(path, std::ios::binary);
  if (!f)
  {
    throw std::runtime_error("cannot open " + path);
  }

  char magic[8];
  f.read(magic, 8);
  if (!f || std::memcmp(magic, "\x93NUMPY", 6) != 0)
  {
    throw std::runtime_error("not an npy file: " + path);
  }
  uint8_t ver_major = (uint8_t)magic[6];

  uint32_t header_len = 0;
  if (ver_major == 1)
  {
    uint8_t b[2];
    f.read((char*)b, 2);
    header_len = b[0] | ((uint32_t)b[1] << 8);
  }
  else
  {
    uint8_t b[4];
    f.read((char*)b, 4);
    header_len = b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16)
                 | ((uint32_t)b[3] << 24);
  }

  std::string header(header_len, '\0');
  f.read(&header[0], header_len);
  if (!f)
  {
    throw std::runtime_error("truncated npy header: " + path);
  }

  size_t dpos = header.find("'descr'");
  size_t q0 = header.find('\'', dpos + 7);
  size_t q1 = header.find('\'', q0 + 1);
  std::string descr = header.substr(q0 + 1, q1 - q0 - 1);
  size_t elem_size;
  if (descr == "<f2")
  {
    elem_size = 2;
  }
  else if (descr == "<f4")
  {
    elem_size = 4;
  }
  else
  {
    throw std::runtime_error("unsupported dtype '" + descr + "' in " + path);
  }

  if (header.find("'fortran_order': False") == std::string::npos)
  {
    throw std::runtime_error("expected C-order array in " + path);
  }

  size_t p0 = header.find('(', header.find("'shape'"));
  size_t p1 = header.find(')', p0);
  std::string shape_str = header.substr(p0 + 1, p1 - p0 - 1);
  std::vector<int> shape;
  size_t pos = 0;
  while (pos < shape_str.size())
  {
    while (pos < shape_str.size() && !isdigit((unsigned char)shape_str[pos]))
    {
      pos++;
    }
    if (pos >= shape_str.size())
    {
      break;
    }
    shape.push_back(std::stoi(shape_str.substr(pos)));
    while (pos < shape_str.size() && isdigit((unsigned char)shape_str[pos]))
    {
      pos++;
    }
  }
  if (shape.empty())
  {
    throw std::runtime_error("cannot parse shape in " + path);
  }

  size_t elems = 1;
  for (int d : shape)
  {
    elems *= (size_t)d;
  }

  NpyArray arr;
  arr.shape = shape;
  arr.data.resize(elems);
  if (elem_size == 4)
  {
    f.read((char*)arr.data.data(), elems * 4);
    if (!f)
    {
      throw std::runtime_error("truncated npy data: " + path);
    }
  }
  else
  {
    std::vector<uint16_t> raw(elems);
    f.read((char*)raw.data(), elems * 2);
    if (!f)
    {
      throw std::runtime_error("truncated npy data: " + path);
    }
    for (size_t i = 0; i < elems; i++)
    {
      arr.data[i] = half_to_float(raw[i]);
    }
  }
  return arr;
}

bool file_exists(const std::string& path)
{
  std::ifstream f(path, std::ios::binary);
  return (bool)f;
}

}  // namespace

MipChain load_mip_chain(const std::string& dir)
{
  MipChain chain;
  chain.channels = 0;

  for (int level = 0;; level++)
  {
    std::string path = dir + "/mip" + std::to_string(level) + ".npy";
    if (!file_exists(path))
    {
      break;
    }
    NpyArray arr = load_npy(path);
    if (arr.shape.size() != 3)
    {
      throw std::runtime_error("expected (H, W, C) array: " + path);
    }

    MipLevel lvl;
    lvl.height = arr.shape[0];
    lvl.width = arr.shape[1];
    lvl.channels = arr.shape[2];
    if (chain.channels == 0)
    {
      chain.channels = lvl.channels;
    }
    else if (lvl.channels != chain.channels)
    {
      throw std::runtime_error("channel count mismatch at " + path);
    }

    size_t bytes = arr.data.size() * sizeof(float);
    cuda_check(cudaMalloc(&lvl.data_dev, bytes), "cudaMalloc mip level");
    cuda_check(cudaMemcpy(lvl.data_dev, arr.data.data(), bytes,
                          cudaMemcpyHostToDevice),
               "cudaMemcpy mip level");
    chain.levels.push_back(lvl);
  }

  if (chain.levels.empty())
  {
    throw std::runtime_error("no mip0.npy found in " + dir);
  }
  return chain;
}

void free_mip_chain(MipChain& chain)
{
  for (MipLevel& lvl : chain.levels)
  {
    cudaFree(lvl.data_dev);
    lvl.data_dev = nullptr;
  }
  chain.levels.clear();
  chain.channels = 0;
}
