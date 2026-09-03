#include "common.hlsli"

struct ModelViewProjection
{
  matrix model_to_world_;
  matrix world_to_view_;
  matrix view_to_proj_;
};

ConstantBuffer<ModelViewProjection> ModelViewProjectionCB : register(b0, space0);

struct VertexShaderInput
{
  float3 pos_ : SV_Position;
  float3 normal_ : NORMAL;
  float3 tangent_ : TANGENT;
  float2 uv_ : TEXCOORD;
};

struct VertexShaderOutput
{
  float4 pos_ : SV_Position0;
	float2 uv_ : TEXCOORD;
  float4 pos_view_ : POSITION0;
  float4 normal_view_ : NORMAL0;
  float4 tangent_view_ : TANGENT0;
  float4 pos_world_ : POSITION1;
};

ConstantBuffer<LightingParams> LightingParamsCBV : register(b1, space0);

Texture2D<float4> tex_ao : register(t0, space0);
Texture2D<float4> tex_albedo : register(t1, space0);
Texture2D<float4> tex_displacement : register(t2, space0);
Texture2D<float4> tex_normal : register(t3, space0);
Texture2D<float4> tex_roughness : register(t4, space0);
TextureCube<float4> tex_specular_ibl : register(t5, space0);
Texture2D<float4> tex_dfg : register(t6, space0);
SamplerState sampler_trilinear : register(s0);

VertexShaderOutput vs_main(VertexShaderInput v_in)
{
  VertexShaderOutput v_out;

  float displacement = (tex_displacement.SampleLevel(
    sampler_trilinear,
    v_in.uv_,
    0.0).r - 0.5) * LightingParamsCBV.displacement_scale_;
  float4 model_pos = float4(v_in.pos_ + v_in.normal_ * displacement, 1.0f);

  float4 world_pos = mul(ModelViewProjectionCB.model_to_world_, model_pos);
  float4 view_pos = mul(ModelViewProjectionCB.world_to_view_, world_pos);
  float4 proj_pos = mul(ModelViewProjectionCB.view_to_proj_, view_pos);

  float4 world_normal = mul(ModelViewProjectionCB.model_to_world_, float4(v_in.normal_, 0.0f));
  float4 view_normal = mul(ModelViewProjectionCB.world_to_view_, world_normal);

  float4 world_tangent = mul(ModelViewProjectionCB.model_to_world_, float4(v_in.tangent_, 0.0f));
  float4 view_tangent = mul(ModelViewProjectionCB.world_to_view_, world_tangent);

  v_out.pos_ = proj_pos;
  v_out.uv_ = v_in.uv_;
  v_out.pos_view_ = view_pos;
  v_out.normal_view_ = view_normal;
  v_out.tangent_view_ = view_tangent;
  v_out.pos_world_ = world_pos;

  return v_out;
}

struct PixelShaderInput
{
  float4 pos_ : SV_Position0;
  float2 uv_ : TEXCOORD;
  float4 pos_view_ : POSITION0;
  float4 normal_view_ : NORMAL0;
  float4 tangent_view_ : TANGENT0;
  float4 pos_world_ : POSITION1;
};

struct PixelShaderOutput
{
  float4 color_ : SV_TARGET;
};

PixelShaderOutput ps_main(PixelShaderInput p_in)
{
  PixelShaderOutput p_out;

  // Samples

  float2 uv = UnjitterUv(p_in.uv_, LightingParamsCBV.jitter_px_);
  NTCMaterialParams mat = NTCDefaultMaterialParams();
  mat.albedo_ = tex_albedo.Sample(sampler_trilinear, uv).rgb;
  mat.normal_ = tex_normal.Sample(sampler_trilinear, uv).rgb;
  mat.ao_ = tex_ao.Sample(sampler_trilinear, uv).r;
  mat.roughness_ = tex_roughness.Sample(sampler_trilinear, uv).r;
  mat.present_ =
    (1u << NTC_SEM_ALBEDO) | (1u << NTC_SEM_NORMAL) | (1u << NTC_SEM_AO) | (1u << NTC_SEM_ROUGHNESS);

  p_out.color_ = ShadeMaterial(
    mat,
    p_in.pos_view_,
    p_in.normal_view_,
    p_in.tangent_view_,
    ModelViewProjectionCB.world_to_view_,
    LightingParamsCBV,
    tex_specular_ibl,
    tex_dfg,
    sampler_trilinear);

  return p_out;
}