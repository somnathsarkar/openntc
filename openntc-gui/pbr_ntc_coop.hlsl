#define COOP_SUPPORT
#include "common.hlsli"

struct ModelViewProjection
{
  matrix model_to_world;
  matrix world_to_view;
  matrix view_to_proj;
};

ConstantBuffer<ModelViewProjection> ModelViewProjectionCB : register(b0, space0);

ConstantBuffer<NTC> NTCCBV : register(b1, space0);
ConstantBuffer<LightingParams> LightingParamsCBV : register(b2, space0);

Buffer<uint> g0 : register(t0, space0);
Buffer<uint> g1 : register(t1, space0);
ByteAddressBuffer W0 : register(t2, space0);
ByteAddressBuffer W1 : register(t3, space0);
ByteAddressBuffer Wout : register(t4, space0);
ByteAddressBuffer W0_scale : register(t5, space0);
ByteAddressBuffer W1_scale : register(t6, space0);
ByteAddressBuffer Wout_scale : register(t7, space0);
TextureCube<float4> tex_specular_ibl : register(t8, space0);
Texture2D<float4> tex_dfg : register(t9, space0);
SamplerState sampler_trilinear : register(s0);

struct VertexShaderInput
{
  float3 pos : SV_Position;
  float3 normal : NORMAL;
  float3 tangent : TANGENT;
  float2 uv : TEXCOORD;
};

struct VertexShaderOutput
{
  float4 pos : SV_Position0;
	float2 uv : TEXCOORD;
  float4 pos_view : POSITION0;
  float4 normal_view : NORMAL0;
  float4 tangent_view : TANGENT0;
  float4 pos_world : POSITION1;
};

VertexShaderOutput vs_main(VertexShaderInput v_in)
{
  VertexShaderOutput v_out;

  vector<uint, 64 / 4> feat;
  vector<float, 12> Woutx;
  GetFeaturesPacked(g0, g1, NTCCBV, v_in.uv, v_in.pos.xy, feat);
  PerformNTCInference(W0, W1, Wout, W0_scale, W1_scale, Wout_scale, NTCCBV, feat, Woutx);
  float displacement = (Woutx[4] - 0.5) * LightingParamsCBV.displacement_scale;
  float4 model_pos = float4(v_in.pos + v_in.normal * displacement, 1.0f);

  float4 world_pos = mul(ModelViewProjectionCB.model_to_world, model_pos);
  float4 view_pos = mul(ModelViewProjectionCB.world_to_view, world_pos);
  float4 proj_pos = mul(ModelViewProjectionCB.view_to_proj, view_pos);

  float4 world_normal = mul(ModelViewProjectionCB.model_to_world, float4(v_in.normal, 0.0f));
  float4 view_normal = mul(ModelViewProjectionCB.world_to_view, world_normal);

  float4 world_tangent = mul(ModelViewProjectionCB.model_to_world, float4(v_in.tangent, 0.0f));
  float4 view_tangent = mul(ModelViewProjectionCB.world_to_view, world_tangent);

  v_out.pos = proj_pos;
  v_out.uv = v_in.uv;
  v_out.pos_view = view_pos;
  v_out.normal_view = view_normal;
  v_out.tangent_view = view_tangent;
  v_out.pos_world = world_pos;

  return v_out;
}

struct PixelShaderInput
{
  float4 pos : SV_Position0;
  float2 uv : TEXCOORD;
  float4 pos_view : POSITION0;
  float4 normal_view : NORMAL0;
  float4 tangent_view : TANGENT0;
  float4 pos_world : POSITION1;
};

struct PixelShaderOutput
{
  float4 color : SV_TARGET;
};

PixelShaderOutput ps_main(PixelShaderInput p_in)
{
  PixelShaderOutput p_out;

  // Constants

  float3 directional_light = float3(0.0, 0.0, 1.0);
  float3 light_color = float3(0.0, 0.0, 0.0);
  float3 f0 = float3(0.04, 0.04, 0.04);
  float3 f90 = float3(1.0, 1.0, 1.0);

  // Samples

  vector<uint, 64 / 4> feat;
  vector<float, 12> Woutx;
  GetFeaturesPacked(g0, g1, NTCCBV, UnjitterUv(p_in.uv, LightingParamsCBV.jitter_px), p_in.pos.xy, feat);
  PerformNTCInference(W0, W1, Wout, W0_scale, W1_scale, Wout_scale, NTCCBV, feat, Woutx);

  float ntc_ao = Woutx[0];
  float3 ntc_albedo = float3(Woutx[1], Woutx[2], Woutx[3]);
  float ntc_displacement = Woutx[4];
  float3 ntc_normal = float3(Woutx[5], Woutx[6], Woutx[7]);
  float ntc_roughness = Woutx[8];

  float4 bitangent_view = float4(cross(p_in.normal_view.xyz, p_in.tangent_view.xyz), 0.0);
  matrix tbn_view = transpose(matrix(p_in.tangent_view, bitangent_view, p_in.normal_view, float4(0.0, 0.0, 0.0, 1.0)));
  float4 surface_normal = float4(ntc_normal * 2.0 - 1.0, 0.0);
  float4 view_normal = mul(tbn_view, surface_normal);
  float3 normal = view_normal.rgb;
  float perceptual_roughness = ntc_roughness;
  float3 albedo = pow(ntc_albedo, 2.2f);

  float3 view_dir = -normalize(p_in.pos_view.xyz / p_in.pos_view.w);
  float3 light_dir = -normalize(directional_light);
  float3 half_dir = normalize((view_dir + light_dir) / 2.0);
  float3 reflect_dir = reflect(-view_dir, normal);

  float NoV = abs(dot(normal, view_dir));
  float NoL = clamp(dot(normal, light_dir), 0.0, 1.0);
  float NoH = clamp(dot(normal, half_dir), 0.0, 1.0);
  float LoH = clamp(dot(light_dir, half_dir), 0.0, 1.0);

  float roughness = perceptual_roughness * perceptual_roughness;

  float D = D_GGX(NoH, roughness);
  float3 F = F_Schlick(LoH, f0);
  float V = V_SmithGGXCorrelated(NoV, NoL, roughness);

  float3 Fr = (D * V) * F;

  float3 Fd = albedo * Fd_Lambert();

  float3 radiance = (Fd + Fr) * light_color * NoL;

  float3 reflect_world = mul(float4(reflect_dir, 0.0), ModelViewProjectionCB.world_to_view).xyz;
  float3 normal_world = mul(float4(normal, 0.0), ModelViewProjectionCB.world_to_view).xyz;
  float lod_ibl = perceptual_roughness * 4.0;
  float3 specular_ibl = tex_specular_ibl.SampleLevel(sampler_trilinear, reflect_world, lod_ibl).rgb;
  float2 specular_dfg = tex_dfg.Sample(sampler_trilinear, float2(NoV, perceptual_roughness)).rg;
  float3 specular_color = f0 * specular_dfg.x + f90 * specular_dfg.y;
  float3 diffuse_ibl = max(IrradianceSh(LightingParamsCBV, normal_world), 0.0);

  radiance += diffuse_ibl * albedo * ntc_ao + specular_color * specular_ibl;
  
  // Tonemapping

  radiance *= LightingParamsCBV.exposure;
  radiance = saturate(mul(ACESOutput, RRTAndODTFit(mul(ACESInput, radiance))));

  p_out.color = float4(pow(radiance, 1.0f / 2.2f), 1.0);

  return p_out;
}