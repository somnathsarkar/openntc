#include "common.hlsli"

struct Transforms
{
  matrix proj_to_view;
  matrix view_to_world;
};

ConstantBuffer<Transforms> TransformsCB : register(b0, space0);

struct VertexShaderOutput
{
  float4 pos : SV_Position0;
  float2 ndc : TEXCOORD0;
};

static const float2 g_map_vid_to_pos[6] = {
  float2(-1, -1),
  float2(-1, 1),
  float2(1, -1),
  float2(-1, 1),
  float2(1, 1),
  float2(1, -1)
};

VertexShaderOutput vs_main(uint vid : SV_VertexID)
{
  VertexShaderOutput v_out;

  v_out.pos = float4(g_map_vid_to_pos[vid], 1.0f, 1.0f);
  v_out.ndc = g_map_vid_to_pos[vid];

  return v_out;
}

ConstantBuffer<LightingParams> LightingParamsCBV : register(b1, space0);

TextureCube<float4> tex_cubemap : register(t0, space0);
SamplerState sampler_trilinear : register(s0);

struct PixelShaderInput
{
  float4 pos : SV_Position0;
  float2 ndc : TEXCOORD0;
};

struct PixelShaderOutput
{
  float4 color : SV_TARGET;
};

PixelShaderOutput ps_main(PixelShaderInput p_in)
{
  PixelShaderOutput p_out;

  float4 view_pos = mul(TransformsCB.proj_to_view, float4(p_in.ndc, 1.0, 1.0));
  float4 world_pos = mul(TransformsCB.view_to_world, view_pos);

  float3 world_dir = normalize(world_pos.xyz / world_pos.w);

  float3 radiance = tex_cubemap.Sample(sampler_trilinear, world_dir).rgb;
  
  // Tonemapping

  radiance *= LightingParamsCBV.exposure;
  radiance = saturate(mul(ACESOutput, RRTAndODTFit(mul(ACESInput, radiance))));
  p_out.color = float4(pow(radiance, 1.0f / 2.2f), 1.0f);

  return p_out;
}