// Package textures into an efficient representation for training

struct PackageTex
{
  int mip;
  int dim;
};

ConstantBuffer<PackageTex> PackageTexCB : register(b0, space0);
Texture2D<float4> tex_albedo : register(t1, space0);
Texture2D<float4> tex_ao : register(t2, space0);
Texture2D<float4> tex_displacement : register(t3, space0);
Texture2D<float4> tex_normal : register(t4, space0);
Texture2D<float4> tex_roughness : register(t5, space0);
RWByteAddressBuffer buffer_package : register(u6, space0);

[numthreads(8, 8, 1)]
void cs_main(uint3 tid : SV_DispatchThreadID)
{
  if (any(tid.xy >= PackageTexCB.dim.xx)) return;

  int3 loc = int3(tid.xy, PackageTexCB.mip);

  float3 albedo = tex_albedo.Load(loc).rgb;
  float ao = tex_ao.Load(loc).r;
  float displacement = tex_displacement.Load(loc).r;
  float3 normal = tex_normal.Load(loc).rgb;
  float roughness = tex_roughness.Load(loc).r;

  int head = (tid.y * PackageTexCB.dim + tid.x) * 9 * 4;
  buffer_package.Store3(head + 0, asuint(albedo));
  buffer_package.Store<float>(head + 3 * 4, ao);
  buffer_package.Store<float>(head + 4 * 4, displacement);
  buffer_package.Store3(head + 5 * 4, asuint(normal));
  buffer_package.Store<float>(head + 8 * 4, roughness);
}