#include <openntc/dataloader.h>

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv)
{
  const char* dir = argc > 1 ? argv[1] : "data/mips/Bricks101_1K-JPG";
  MipChain chain = load_mip_chain(dir);

  std::printf("%s: %zu levels, %d channels\n", dir, chain.levels.size(),
              chain.channels);
  for (size_t i = 0; i < chain.levels.size(); i++)
  {
    const MipLevel& l = chain.levels[i];
    float first[4] = {0};
    int n = l.channels < 4 ? l.channels : 4;
    cudaMemcpy(first, l.data_dev, n * sizeof(float), cudaMemcpyDeviceToHost);
    std::printf("  mip%zu: %4d x %4d  texel(0,0)[0..%d] =", i, l.height,
                l.width, n - 1);
    for (int c = 0; c < n; c++)
    {
      std::printf(" %.6f", first[c]);
    }
    std::printf("\n");
  }

  free_mip_chain(chain);
  return 0;
}
