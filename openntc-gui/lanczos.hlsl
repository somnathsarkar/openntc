#define PI 3.14159265359

struct Lanczos
{
  int mip_src;
  int w_src;
  int h_src;
  int a;
  int is_y;
};

ConstantBuffer<Lanczos> LanczosCB : register(b0, space0);
Texture2D<float4> tex_in : register(t1, space0);
RWTexture2D<float4> tex_out : register(u2, space0);

float sinc(float x)
{
  if (abs(x) < 1e-8)
    return 1.0f;
  return sin(x) / x;
}

float lanczos(float x, float a)
{
  return sinc(PI * x) * sinc(PI * x / a);
}

[numthreads(8, 8, 1)]
void cs_main(uint3 tid : SV_DispatchThreadID)
{
  int2 axis = LanczosCB.is_y ? int2(0, 1) : int2(1, 0);
  int2 out_dim = LanczosCB.is_y ?
                  int2(LanczosCB.w_src, LanczosCB.h_src / 2) :
                  int2(LanczosCB.w_src / 2, LanczosCB.h_src);
  if (any(tid.xy >= out_dim)) return;
  int3 texel_center = LanczosCB.is_y ?
                      int3(tid.x, tid.y * 2 + 1, LanczosCB.mip_src) :
                      int3(tid.x * 2 + 1, tid.y, LanczosCB.mip_src);
  int lim = LanczosCB.is_y ? LanczosCB.h_src : LanczosCB.w_src;
  // Samples
  float4 total = float4(0.0, 0.0, 0.0, 0.0);
  float total_weight = 0.0;
  for (int i = -2 * LanczosCB.a; i < 2 * LanczosCB.a; i++)
  {
    int3 sample_loc = texel_center + i * int3(axis, 0);
    int sample_loc_dim = LanczosCB.is_y ? sample_loc.y : sample_loc.x;
    if (sample_loc_dim < 0 || sample_loc_dim >= lim)
      continue;
    float weight = lanczos((0.5f + i) * 0.5f, LanczosCB.a);
    float4 samp = tex_in.Load(sample_loc);
    total += samp * weight;
    total_weight += weight;
  }
  tex_out[tid.xy] = total / total_weight;
}