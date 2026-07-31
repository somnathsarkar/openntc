#pragma once

#include <string>
#include <vector>

// Loads a material's mip chain (data/mips/<material>/mip{k}.npy) into CUDA
// memory. Files are numpy arrays of shape (H, W, C), C-order, dtype <f2
// (float16) or <f4; values are stored-space in [0,1], channel-last texel
// order (texel (y,x) starts at (y*W + x) * C). Shapes and level count are
// derived from the npy headers themselves; channel *semantics* live in the
// directory's meta.json and are not needed for training math.

struct MipLevel
{
  int height;
  int width;
  int channels;
  float* data_dev;  // device pointer, H*W*C floats, channel-last
};

struct MipChain
{
  std::vector<MipLevel> levels;
  int channels;  // same for every level
};

// Throws std::runtime_error on missing/malformed files. Chain must contain
// at least mip0.npy; levels are read consecutively until the first gap.
MipChain load_mip_chain(const std::string& dir);

void free_mip_chain(MipChain& chain);
