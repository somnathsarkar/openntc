#define PI 3.14159265359

struct ModelViewProjection
{
  matrix model_to_world;
  matrix world_to_view;
  matrix view_to_proj;
};

ConstantBuffer<ModelViewProjection> ModelViewProjectionCB : register(b0, space0);

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
};

VertexShaderOutput vs_main(VertexShaderInput v_in)
{
  VertexShaderOutput v_out;

  float4 world_pos = mul(ModelViewProjectionCB.model_to_world, float4(v_in.pos, 1.0f));
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

  return v_out;
}

Texture2D<float4> tex_ao : register(t1, space0);
Texture2D<float4> tex_albedo : register(t2, space0);
Texture2D<float4> tex_displacement : register(t3, space0);
Texture2D<float4> tex_normal : register(t4, space0);
Texture2D<float4> tex_roughness : register(t5, space0);
SamplerState sampler_bilinear_clamp : register(s0);

struct PixelShaderInput
{
  float4 pos : SV_Position0;
  float2 uv : TEXCOORD;
  float4 pos_view : POSITION0;
  float4 normal_view : NORMAL0;
  float4 tangent_view : TANGENT0;
};

struct PixelShaderOutput
{
  float4 color : SV_TARGET;
};

float D_GGX(float NoH, float a)
{
  float a2 = a * a;
  float f = (NoH * a2 - NoH) * NoH + 1.0;
  return a2 / (PI * f * f);
}

float3 F_Schlick(float u, float3 f0)
{
  return f0 + (float3(1.0, 1.0, 1.0) - f0) * pow(1.0 - u, 5.0);
}

float V_SmithGGXCorrelated(float NoV, float NoL, float a)
{
  float a2 = a * a;
  float GGXL = NoV * sqrt((-NoL * a2 + NoL) * NoL + a2);
  float GGXV = NoL * sqrt((-NoV * a2 + NoV) * NoV + a2);
  return 0.5 / (GGXV + GGXL);
}

float Fd_Lambert()
{
  return 1.0 / PI;
}

PixelShaderOutput ps_main(PixelShaderInput p_in)
{
  PixelShaderOutput p_out;

  // Constants

  float3 directional_light = float3(0.0, 0.0, 1.0);
  float3 light_color = float3(1.0, 1.0, 1.0);
  float3 f0 = float3(0.04, 0.04, 0.04);

  // Samples

  float4 bitangent_view = float4(cross(p_in.normal_view.xyz, p_in.tangent_view.xyz), 0.0);
  matrix tbn_view = transpose(matrix(p_in.tangent_view, bitangent_view, p_in.normal_view, float4(0.0, 0.0, 0.0, 1.0)));
  float4 surface_normal = float4(tex_normal.Sample(sampler_bilinear_clamp, p_in.uv).rgb * 2.0 - 1.0, 0.0);
  float4 view_normal = mul(tbn_view, surface_normal);
  float3 normal = view_normal.rgb;
  float perceptual_roughness = tex_roughness.Sample(sampler_bilinear_clamp, p_in.uv).r;
  float3 albedo = tex_albedo.Sample(sampler_bilinear_clamp, p_in.uv).rgb;

  float3 view_dir = -normalize(p_in.pos_view.xyz / p_in.pos_view.w);
  float3 light_dir = -normalize(directional_light);
  float3 half_dir = normalize((view_dir + light_dir) / 2.0);

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

  p_out.color = float4(radiance, 1.0);

  return p_out;
}