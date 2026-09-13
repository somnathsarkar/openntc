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

ConstantBuffer<NTC> NTCCBV : register(b1, space0);
ConstantBuffer<LightingParams> LightingParamsCBV : register(b2, space0);

Buffer<uint> g0 : register(t0, space0);
Buffer<uint> g1 : register(t1, space0);
ByteAddressBuffer decoder : register(t2, space0);
TextureCube<float4> tex_specular_ibl : register(t3, space0);
Texture2D<float4> tex_dfg : register(t4, space0);

VertexShaderOutput vs_main(VertexShaderInput v_in)
{
  VertexShaderOutput v_out;

  // NOTE: This heuristic LOD is exclusively for the models shipped with the UI
  //  No stochastic filtering for displacement.
  float displacement = 0.0f;
  float lod = clamp(max(0.0f, log2(NTCCBV.dim_ / 200.0f)), 0.0, float(NTCCBV.mip_count_ - 1));
  if (LightingParamsCBV.displacement_scale_ > 0.0f)
  {
    MaterialParams vs_mat = SampleMaterial(g0, g1, decoder, NTCCBV, v_in.uv_, int(round(lod)));
    displacement = (vs_mat.displacement_ - 0.5) * LightingParamsCBV.displacement_scale_;
  }
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

SamplerState sampler_trilinear : register(s0);

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
  float2 footprint;
  float lodab = ComputeLod(NTCCBV, uv, footprint);
  // Incorporate temporal noise
  float2 pos_noise = p_in.pos_.xy + 5.588238f * LightingParamsCBV.noise_frame_;
  int lod = StochasticFilterLod(lodab, pos_noise);
  MaterialParams mat =
    SampleMaterial(g0, g1, decoder, NTCCBV, StochasticFilterUv(uv, footprint, pos_noise), lod);

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