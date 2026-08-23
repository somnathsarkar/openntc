#include <dx/linalg.h>

#define PI 3.14159265359

#define G0_BITS 2
#define G1_BITS 4
#define G0_CHANNELS 8
#define G1_CHANNELS 12
#define MAX_LEVELS 5

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
  float4 pos_world : POSITION1;
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
  v_out.pos_world = world_pos;

  return v_out;
}

struct NTC
{
  int4 g0_grid_dim[2];
  int4 g1_grid_dim[2];
  uint4 g0_offset[2];
  uint4 g1_offset[2];
  int g0_bytes_per_channel;
  int g1_bytes_per_channel;
  int g0_channels;
  int g1_channels;
  int dim;
  int mip_count;
  float rcp_s_a1;
  float rcp_s_a2;
};

struct LightingParams
{
  float exposure;
  float displacement_scale;
  float normal_scale;
  float pad0;
  float3 diffuse_sh[9];
};

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


int FeatureLevelForLod(int lod)
{
  if (lod <= 3)
    return 0;
  else if (lod <= 5)
    return 1;
  else if (lod <= 7)
    return 2;
  else if (lod <= 9)
    return 3;
  return 4;
}

// 4 x 2-bit fields in the low byte of b -> top 2 bits of 4 bytes (k << 6 per lane)
uint Spread2(uint b)
{
  return ((b & 0x03u) <<  6) | ((b & 0x0Cu) << 12) |
         ((b & 0x30u) << 18) | ((b & 0xC0u) << 24);
}

void GetFeaturesPacked(float2 uv, float2 pos_screen, out vector<uint, 64 / 4> o_feat)
{
  float2 dUvdX = ddx(uv) * NTCCBV.dim;
  float2 dUvdY = ddy(uv) * NTCCBV.dim;
  float gUvdX = dot(dUvdX, dUvdX);
  float gUvdY = dot(dUvdY, dUvdY);
  float2 texels_along_major_axis = (gUvdX > gUvdY) ? dUvdX : dUvdY;
  float minor_axis = sqrt(min(gUvdX, gUvdY));
  float major_axis = sqrt(max(gUvdX, gUvdY));
  float max_aniso = 8.0f;
  minor_axis = max(minor_axis, major_axis / max_aniso);

  float lodab = log2(minor_axis);
  float lodab_clamped = clamp(lodab, 0.0, float(NTCCBV.mip_count - 1));
  // Interleaved Gradient Noise - "Next Generation Post-Processing in Call of Duty Advanced Warfare"
  float ign0 = frac(52.9829189 * frac(0.06711056 * pos_screen.x + 0.00583715 * pos_screen.y));
  int lod = int(lodab_clamped) + (ign0 < frac(lodab_clamped) ? 1 : 0);
  int feature_level = FeatureLevelForLod(lod);
  int fli = (feature_level / 4);
  int flj = (feature_level % 4);
  int g0_dim = NTCCBV.g0_grid_dim[fli][flj];
  int g1_dim = NTCCBV.g1_grid_dim[fli][flj];

  float ign1 = frac(52.9829189 * frac(0.06711056 * (pos_screen.x + 61.0) + 0.00583715 * (pos_screen.y + 37.0)));
  float2 uv_jittered = uv + (texels_along_major_axis / NTCCBV.dim) * (ign1 - 0.5);

  // G0: 8ch x 2b = 16b per cell, half-word aligned; exact integer decode.
  int2 g0_xy = int2(floor(uv_jittered * g0_dim - 0.5));
  int g0_x[2];
  g0_x[0] = max(g0_xy.x, 0);
  g0_x[1] = min(g0_xy.x + 1, g0_dim - 1);
  int g0_y[2];
  g0_y[0] = max(g0_xy.y, 0);
  g0_y[1] = min(g0_xy.y + 1, g0_dim - 1);

  uint g0_off = NTCCBV.g0_offset[fli][flj] / 4;
  uint g1_off = NTCCBV.g1_offset[fli][flj] / 4;

  [unroll]
  for (int i = 0; i < 2; i++)
  {
    [unroll]
    for (int j = 0; j < 2; j++)
    {
      int ij = i * 2 + j;
      uint cell = (uint)(g0_y[i] * g0_dim + g0_x[j]);
      uint word = g0.Load(g0_off + (cell >> 1));
      uint c16 = (word >> ((cell & 1u) * 16u)) & 0xFFFFu;
      // (k << 6) - 128 == (k << 6) ^ 0x80 per byte: lane = f * 128 exactly
      o_feat[ij * 2 + 0] = Spread2(c16 & 0xFFu) ^ 0x80808080u;  // ch 0-3
      o_feat[ij * 2 + 1] = Spread2(c16 >>   8u) ^ 0x80808080u;  // ch 4-7
    }
  }

  // G1: 12ch x 4b = 48b per cell
  int2 g1_xy = int2(floor(uv_jittered * g1_dim - 0.5));
  int g1_x[2];
  g1_x[0] = max(g1_xy.x, 0);
  g1_x[1] = min(g1_xy.x + 1, g1_dim - 1);
  int g1_y[2];
  g1_y[0] = max(g1_xy.y, 0);
  g1_y[1] = min(g1_xy.y + 1, g1_dim - 1);
  float2 fr = frac(uv_jittered * g1_dim - 0.5);
  float mult[4] = {(1 - fr.x) * (1 - fr.y), fr.x * (1 - fr.y), (1 - fr.x) * fr.y, fr.x * fr.y};

  float g1_blend[G1_CHANNELS];
  [unroll]
  for (int k = 0; k < G1_CHANNELS; k++)
    g1_blend[k] = 0.0f;

  [unroll]
  for (int i = 0; i < 2; i++)
  {
    [unroll]
    for (int j = 0; j < 2; j++)
    {
      float m = mult[i * 2 + j];
      uint cell = (uint)(g1_y[i] * g1_dim + g1_x[j]);
      uint bitpos = cell * 48u;
      uint w0 = g1.Load(g1_off + (bitpos >> 5));
      uint w1 = g1.Load(g1_off + (bitpos >> 5) + 1u);
      uint shift = bitpos & 31u;   // 0 or 16
      [unroll]
      for (int k = 0; k < G1_CHANNELS; k++)
      {
        uint bit = shift + (uint)k * 4u;
        uint nib = ((bit < 32u ? w0 : w1) >> (bit & 31u)) & 0xFu;
        g1_blend[k] += m * (((float)nib - 8.0f) / 8.0f);
      }
    }
  }

  [unroll]
  for (int q = 0; q < 3; q++)
  {
    o_feat[8 + q] = pack_clamp_s8(int4(round(float4(
      g1_blend[q * 4 + 0], g1_blend[q * 4 + 1],
      g1_blend[q * 4 + 2], g1_blend[q * 4 + 3]) * 128.0f)));
  }

  // 12 triangular waves
  float2 cpos = uv_jittered * float(NTCCBV.dim >> lod);
  float pe[12];
  int periods[3] = {8, 4, 2};
  [unroll]
  for (int p = 0; p < 3; p++)
  {
    [unroll]
    for (int a = 0; a < 2; a++)
    {
      [unroll]
      for (int h = 0; h < 2; h++)
      {
        float P = (float)periods[p];
        float c = (a == 0) ? floor(cpos.x) : floor(cpos.y);
        float phase = (h == 0) ? 0.0f : 0.25f * P;
        float t = (c + 0.5f + phase) / P;
        pe[p * 4 + a * 2 + h] = 1.0f - 4.0f * abs(frac(t) - 0.5f);
      }
    }
  }
  [unroll]
  for (int q = 0; q < 3; q++)
  {
    o_feat[11 + q] = pack_clamp_s8(int4(round(float4(
      pe[q * 4 + 0], pe[q * 4 + 1], pe[q * 4 + 2], pe[q * 4 + 3]) * 128.0f)));
  }

  // Lane 56: lod scalar; lanes 57-59: zero pad (must match W0's zero pad rows).
  o_feat[14] = pack_clamp_s8(int4(int(round(lod / float(NTCCBV.mip_count - 1) * 128.0f)), 0, 0, 0));
  o_feat[15] = 0u;
}

vector<float, 64> hardgelu_coop(vector<float, 64> x)
{
  return select(x < -1.5f, 0.0, select(x < 1.5f, (x / 3.0f) * (x + 1.5f), x));
}

float3 IrradianceSh(float3 n)
{
  return LightingParamsCBV.diffuse_sh[0] +
          LightingParamsCBV.diffuse_sh[1] * n.y +
          LightingParamsCBV.diffuse_sh[2] * n.z +
          LightingParamsCBV.diffuse_sh[3] * n.x +
          LightingParamsCBV.diffuse_sh[4] * (n.y * n.x) +
          LightingParamsCBV.diffuse_sh[5] * (n.y * n.z) +
          LightingParamsCBV.diffuse_sh[6] * (3.0 * n.z * n.z - 1.0) +
          LightingParamsCBV.diffuse_sh[7] * (n.z * n.x) +
          LightingParamsCBV.diffuse_sh[8] * (n.x * n.x - n.y * n.y);
}

PixelShaderOutput ps_main(PixelShaderInput p_in)
{
  using namespace dx::linalg;

  PixelShaderOutput p_out;

  // Constants

  float3 directional_light = float3(0.0, 0.0, 1.0);
  float3 light_color = float3(1.0, 1.0, 1.0);
  float3 f0 = float3(0.04, 0.04, 0.04);
  float3 f90 = float3(1.0, 1.0, 1.0);

  // Samples

  vector<uint, 64 / 4> feat;
  vector<uint, 64 / 4> W0x;
  vector<uint, 64 / 4> W1x;
  vector<float, 12> Woutx;
  GetFeaturesPacked(p_in.uv, p_in.pos.xy, feat);

  const vector<int32_t, 64> zero64 = (vector<int32_t, 64>)0;
  const vector<int32_t, 12> zero12 = (vector<int32_t, 12>)0;

  // Coop multiplication
  typedef Matrix<ComponentType::I8, 64, 64, MatrixUse::A, MatrixScope::Thread> W0_t;
  W0_t W0_coop = W0_t::Load<MatrixLayout::RowMajor>(W0, 0, 64);
  InterpretedVector<uint, 64 / 4, ComponentType::I8> feat_coop = MakeInterpretedVector<ComponentType::I8>(feat);
  vector<int32_t, 64> W0x_acc = MultiplyAdd<int32_t>(W0_coop, feat_coop, zero64);
  vector<float, 64> W0_scale_coop = W0_scale.Load< vector<float, 64> >(0);
  vector<float, 64> W0x_facc = vector<float, 64>(W0x_acc) * W0_scale_coop;
  W0x_facc = hardgelu_coop(W0x_facc) * NTCCBV.rcp_s_a1;
  vector<int32_t, 64> W0x_unpacked = vector<int32_t, 64>(round(W0x_facc));
  [unroll]
  for (int i = 0 ; i < 64; i+=4)
    W0x[i / 4] = pack_clamp_s8(int4(W0x_unpacked[i], W0x_unpacked[i + 1], W0x_unpacked[i + 2], W0x_unpacked[i + 3]));

  typedef Matrix<ComponentType::I8, 64, 64, MatrixUse::A, MatrixScope::Thread> W1_t;
  W1_t W1_coop = W1_t::Load<MatrixLayout::RowMajor>(W1, 0, 64);
  InterpretedVector<uint, 64 / 4, ComponentType::I8> W0x_coop = MakeInterpretedVector<ComponentType::I8>(W0x);
  vector<int32_t, 64> W1x_acc = MultiplyAdd<int32_t>(W1_coop, W0x_coop, zero64);
  vector<float, 64> W1_scale_coop = W1_scale.Load< vector<float, 64> >(0);
  vector<float, 64> W1x_facc = vector<float, 64>(W1x_acc) * W1_scale_coop;
  W1x_facc = hardgelu_coop(W1x_facc) * NTCCBV.rcp_s_a2;
  vector<int32_t, 64> W1x_unpacked = vector<int32_t, 64>(round(W1x_facc));
  [unroll]
  for (int i = 0 ; i < 64; i+=4)
    W1x[i / 4] = pack_clamp_s8(int4(W1x_unpacked[i], W1x_unpacked[i + 1], W1x_unpacked[i + 2], W1x_unpacked[i + 3]));

  typedef Matrix<ComponentType::I8, 12, 64, MatrixUse::A, MatrixScope::Thread> Wout_t;
  Wout_t Wout_coop = Wout_t::Load<MatrixLayout::RowMajor>(Wout, 0, 64);
  InterpretedVector<uint, 64 / 4, ComponentType::I8> W1x_coop = MakeInterpretedVector<ComponentType::I8>(W1x);
  vector<int32_t, 12> Woutx_acc = MultiplyAdd<int32_t>(Wout_coop, W1x_coop, zero12);
  vector<float, 12> Wout_scale_coop = Wout_scale.Load< vector<float, 12> >(0);
  Woutx = vector<float, 12>(Woutx_acc) * Wout_scale_coop;

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
  float3 diffuse_ibl = max(IrradianceSh(normal_world), 0.0);

  radiance += diffuse_ibl * albedo * ntc_ao + specular_color * specular_ibl;
  // Exposure correction
  radiance *= LightingParamsCBV.exposure;

  p_out.color = float4(pow(radiance, 1.0f / 2.2f), 1.0);

  return p_out;
}