#include "common.hlsli"

struct FlatParams
{
  float2 pane_dim_;
  int semantic_;
  int channels_;
};

ConstantBuffer<FlatParams> FlatCBV : register(b0, space0);
ConstantBuffer<NTC> NTCCBV : register(b1, space0);
ConstantBuffer<LightingParams> LightingParamsCBV : register(b2, space0);

Buffer<uint> g0 : register(t0, space0);
Buffer<uint> g1 : register(t1, space0);
ByteAddressBuffer decoder : register(t2, space0);

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

float3 SemanticToRgb(MaterialParams mat, Semantic sem)
{
  switch (sem)
  {
    case Semantic::Albedo:
      return mat.albedo_;
    case Semantic::Alpha:
      return mat.alpha_.xxx;
    case Semantic::Displacement:
      return mat.displacement_.xxx;
    case Semantic::Emissive:
      return mat.emissive_.xxx;
    case Semantic::Gloss:
      return mat.gloss_.xxx;
    case Semantic::Metallic:
      return mat.metallic_.xxx;
    case Semantic::Normal:
      return mat.normal_;
    case Semantic::AO:
      return mat.ao_.xxx;
    case Semantic::Roughness:
      return mat.roughness_.xxx;
    case Semantic::Specular:
      return mat.specular_;
    case Semantic::Transmission:
      return mat.transmission_.xxx;
    default: return 0.0.xxx;
  }
}

float4 ps_main(VertexShaderOutput p_in) : SV_TARGET
{
  if (any(p_in.uv_ < 0.0) || any(p_in.uv_ > 1.0))
    discard;

  float2 uv = p_in.uv_;
  float2 footprint;
  float lodab = ComputeLod(NTCCBV, uv, footprint);
  float2 pos_noise = p_in.pos_.xy + 5.588238f * LightingParamsCBV.noise_frame_;
  int lod = StochasticFilterLod(lodab, pos_noise);
  MaterialParams mat =
    SampleMaterial(g0, g1, decoder, NTCCBV, StochasticFilterUv(uv, footprint, pos_noise), lod);

  return float4(SemanticToRgb(mat, (Semantic)FlatCBV.semantic_), 1.0);
}
