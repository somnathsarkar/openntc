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

  // Constants

  float3 f0 = float3(0.04, 0.04, 0.04);
  float3 f90 = float3(1.0, 1.0, 1.0);

  // Samples

  float4 bitangent_view = float4(cross(p_in.normal_view_.xyz, p_in.tangent_view_.xyz), 0.0);
  matrix tbn_view = transpose(matrix(p_in.tangent_view_, bitangent_view, p_in.normal_view_, float4(0.0, 0.0, 0.0, 1.0)));
  float2 uv = UnjitterUv(p_in.uv_, LightingParamsCBV.jitter_px_);
  float4 surface_normal = float4(tex_normal.Sample(sampler_trilinear, uv).rgb * 2.0 - 1.0, 0.0);
  float4 view_normal = mul(tbn_view, surface_normal);
  float3 normal_scaled = normalize(float3(view_normal.xy * LightingParamsCBV.normal_scale_, view_normal.z));
  float3 normal = normal_scaled;
  float perceptual_roughness = tex_roughness.Sample(sampler_trilinear, uv).r;
  float3 albedo = pow(tex_albedo.Sample(sampler_trilinear, uv).rgb, 2.2);
  float ao = tex_ao.Sample(sampler_trilinear, uv).r;

  float3 view_dir = -normalize(p_in.pos_view_.xyz / p_in.pos_view_.w);
  float3 reflect_dir = reflect(-view_dir, normal);

  float NoV = abs(dot(normal, view_dir));

  float roughness = perceptual_roughness * perceptual_roughness;

  float3 reflect_world = mul(float4(reflect_dir, 0.0), ModelViewProjectionCB.world_to_view_).xyz;
  float3 normal_world = mul(float4(normal, 0.0), ModelViewProjectionCB.world_to_view_).xyz;
  float lod_ibl = perceptual_roughness * 4.0;
  float3 specular_ibl = tex_specular_ibl.SampleLevel(sampler_trilinear, reflect_world, lod_ibl).rgb;
  float2 specular_dfg = tex_dfg.Sample(sampler_trilinear, float2(NoV, 1.0 - perceptual_roughness)).rg;
  float3 specular_color = f0 * specular_dfg.x + f90 * specular_dfg.y;
  float3 diffuse_ibl = max(IrradianceSh(LightingParamsCBV, normal_world), 0.0);
  float so = SpecularOcclusion(NoV, ao, perceptual_roughness);

  float3 radiance = diffuse_ibl * albedo * ao + specular_color * specular_ibl * so;
  
  // Tonemapping

  radiance *= LightingParamsCBV.exposure_;
  radiance = saturate(mul(ACESOutput, RRTAndODTFit(mul(ACESInput, radiance))));

  p_out.color_ = float4(pow(radiance, 1.0f / 2.2f), 1.0);

  return p_out;
}