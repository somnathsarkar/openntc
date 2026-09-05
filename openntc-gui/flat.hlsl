#include "common.hlsli"

struct FlatParams
{
  float2 pane_dim_;
  int semantic_;
  int channels_;
};

ConstantBuffer<FlatParams> FlatCBV : register(b0, space0);

Texture2D<float4> tex_color : register(t0, space0);
SamplerState sampler_trilinear : register(s0);

struct VertexShaderOutput
{
  float4 pos_ : SV_Position;
  float2 uv_ : TEXCOORD;
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
  float2 ndc = g_map_vid_to_pos[vid];
  v_out.pos_ = float4(ndc, 0.0, 1.0);
  float aspect = FlatCBV.pane_dim_.x / FlatCBV.pane_dim_.y;
  float2 square = (aspect >= 1.0) ? float2(ndc.x * aspect, ndc.y) : float2(ndc.x, ndc.y / aspect);
  v_out.uv_ = float2(square.x * 0.5 + 0.5, 0.5 - square.y * 0.5);
  return v_out;
}

float4 ps_main(VertexShaderOutput p_in) : SV_TARGET
{
  if (any(p_in.uv_ < 0.0) || any(p_in.uv_ > 1.0))
    discard;
  float3 samp = tex_color.Sample(sampler_trilinear, p_in.uv_).rgb;
  if (FlatCBV.channels_ == 1)
    samp = samp.rrr;
  return float4(samp, 1.0);
}
