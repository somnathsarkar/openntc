#define PI 3.14159265359

#define G0_BITS 2
#define G1_BITS 4
#define G0_CHANNELS 8
#define G1_CHANNELS 12

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

struct NTC
{
  int4 g0_grid_dim;
  int4 g1_grid_dim;
  int g0_bytes_per_channel;
  int g1_bytes_per_channel;
  int g0_channels;
  int g1_channels;
  int dim;
  float rcp_s_a1;
  float rcp_s_a2;
};

ConstantBuffer<NTC> NTCCBV : register(b1, space0);

Buffer<uint> g0[4] : register(t0, space0);
Buffer<uint> g1[4] : register(t4, space0);
Buffer<uint4> W0 : register(t8, space0);
Buffer<uint4> W1 : register(t9, space0);
Buffer<uint4> Wout : register(t10, space0);
Buffer<float4> W0_scale : register(t11, space0);
Buffer<float4> W1_scale : register(t12, space0);
Buffer<float4> Wout_scale : register(t13, space0);

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


int FeatureLevelForLod(int lod)
{
  if (lod <= 3)
    return 0;
  else if (lod <= 5)
    return 1;
  else if (lod <= 7)
    return 2;
  return 3;
}

float GetG0(int level, int x, int y, int c)
{
  int vals_per_pack = 32 / G0_BITS;
  int g0_idx = y * NTCCBV.g0_grid_dim[level] * G0_CHANNELS + x * G0_CHANNELS + c;
  int pack_idx = g0_idx / vals_per_pack;
  int pack_subidx = g0_idx % vals_per_pack;
  uint val = g0[level].Load(pack_idx);
  val >>= (pack_subidx * G0_BITS);
  val &= ((1u << G0_BITS) - 1);
  float ans = val;
  ans -= (1u << (G0_BITS - 1));
  ans /= (1u << (G0_BITS - 1));
  return ans;
}

float GetG1(int level, int x, int y, int c)
{
  int vals_per_pack = 32 / G1_BITS;
  int g1_idx = y * NTCCBV.g1_grid_dim[level] * G1_CHANNELS + x * G1_CHANNELS + c;
  int pack_idx = g1_idx / vals_per_pack;
  int pack_subidx = g1_idx % vals_per_pack;
  uint val = g1[level].Load(pack_idx);
  val >>= (pack_subidx * G1_BITS);
  val &= ((1u << G1_BITS) - 1);
  float ans = val;
  ans -= (1u << (G1_BITS - 1));
  ans /= (1u << (G1_BITS - 1));
  return ans;
}

void GetFeatures(float2 uv, out float o_feat[64])
{
  float2 dUvdX = ddx(uv) * NTCCBV.dim;
  float2 dUvdY = ddy(uv) * NTCCBV.dim;
  float d = max(dot(dUvdX, dUvdX), dot(dUvdY, dUvdY));
  float lodab = 0.5 * log2(d);
  int lod = clamp(int(lodab), 0, 8);
  int feature_level = FeatureLevelForLod(lod);

  int2 g0_xy = int2(floor(uv * NTCCBV.g0_grid_dim[feature_level] - 0.5));
  int g0_x[2];
  g0_x[0] = max(g0_xy.x, 0.0);
  g0_x[1] = min(g0_xy.x + 1, NTCCBV.g0_grid_dim[feature_level] - 1);
  int g0_y[2];
  g0_y[0] = max(g0_xy.y, 0.0);
  g0_y[1] = min(g0_xy.y + 1, NTCCBV.g0_grid_dim[feature_level] - 1);

  for (int i = 0; i < 2; i++)
  {
    for (int j = 0; j < 2; j++)
    {
      int ij = i * 2 + j;
      for (int k = 0; k < G0_CHANNELS; k++)
      {
        o_feat[ij * G0_CHANNELS + k] = GetG0(feature_level, g0_x[j], g0_y[i], k);
      }
    }
  }

  int2 g1_xy = int2(floor((uv * NTCCBV.g1_grid_dim[feature_level] - 0.5)));
  int g1_x[2];
  g1_x[0] = max(g1_xy.x, 0.0);
  g1_x[1] = min(g1_xy.x + 1, NTCCBV.g1_grid_dim[feature_level] - 1);
  int g1_y[2];
  g1_y[0] = max(g1_xy.y, 0.0);
  g1_y[1] = min(g1_xy.y + 1, NTCCBV.g1_grid_dim[feature_level] - 1);
  float2 xy_frac = frac(uv * NTCCBV.g1_grid_dim[feature_level] - 0.5);
  float mult[4] = {(1 - xy_frac.x) * (1 - xy_frac.y), xy_frac.x * (1 - xy_frac.y), (1 - xy_frac.x) * xy_frac.y, xy_frac.x * xy_frac.y};
  float g1_contrib[12];
  for (int i = 0; i < 12; i++)
    g1_contrib[i] = 0;

  for (int i = 0; i < 2; i++)
  {
    for (int j = 0; j < 2; j++)
    {
      int ij = i * 2 + j;
      float m = mult[ij];
      for (int k = 0; k < G1_CHANNELS; k++)
      {
        g1_contrib[k] += m * GetG1(feature_level, g1_x[j], g1_y[i], k);
      }
    }
  }

  for (int i = 0; i < G1_CHANNELS; i++)
    o_feat[4 * G0_CHANNELS + i] = g1_contrib[i];

  int periods[3] = {8, 4, 2};
  int pos_off = 4 * G0_CHANNELS + G1_CHANNELS;
  float2 cpos = uv * float(NTCCBV.dim >> lod);
  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 2; j++)
    {
      for (int k = 0; k < 2; k++)
      {
        int P = periods[i];
        int c = (j == 0) ? cpos.x : cpos.y;
        float phase = (k == 0) ? 0.0f : (0.25f * P);
        float t = (c + 0.5f + phase) / P;
        float s = t - floor(t);
        float o = 1 - 4.0f * abs(s - 0.5f);
        o_feat[pos_off] = o;
        pos_off++;
      }
    }
  }

  o_feat[57 - 1] = lod / 8.0f;

  for (int i = 57; i < 64; i++)
    o_feat[i] = 0.0f;
}

// 4 x 2-bit fields in the low byte of b -> top 2 bits of 4 bytes (k << 6 per lane)
uint Spread2(uint b)
{
  return ((b & 0x03u) <<  6) | ((b & 0x0Cu) << 12) |
         ((b & 0x30u) << 18) | ((b & 0xC0u) << 24);
}

// clamp to [-128,127] and pack 4 int8 lanes
// This relies on SM 6.6 with minimal perf benefit, could hardcode it instead.

uint PackS8(int4 v)
{
  return pack_clamp_s8(v);
}

void GetFeaturesPacked(float2 uv, out uint o_feat[16])
{
  float2 dUvdX = ddx(uv) * NTCCBV.dim;
  float2 dUvdY = ddy(uv) * NTCCBV.dim;
  float d = max(dot(dUvdX, dUvdX), dot(dUvdY, dUvdY));
  float lodab = 0.5 * log2(d);
  int lod = clamp(int(lodab), 0, 8);
  int feature_level = FeatureLevelForLod(lod);
  int g0_dim = NTCCBV.g0_grid_dim[feature_level];
  int g1_dim = NTCCBV.g1_grid_dim[feature_level];

  // G0: 8ch x 2b = 16b per cell, half-word aligned; exact integer decode.
  int2 g0_xy = int2(floor(uv * g0_dim - 0.5));
  int g0_x[2];
  g0_x[0] = max(g0_xy.x, 0);
  g0_x[1] = min(g0_xy.x + 1, g0_dim - 1);
  int g0_y[2];
  g0_y[0] = max(g0_xy.y, 0);
  g0_y[1] = min(g0_xy.y + 1, g0_dim - 1);

  [unroll]
  for (int i = 0; i < 2; i++)
  {
    [unroll]
    for (int j = 0; j < 2; j++)
    {
      int ij = i * 2 + j;
      uint cell = (uint)(g0_y[i] * g0_dim + g0_x[j]);
      uint word = g0[feature_level].Load(cell >> 1);
      uint c16 = (word >> ((cell & 1u) * 16u)) & 0xFFFFu;
      // (k << 6) - 128 == (k << 6) ^ 0x80 per byte: lane = f * 128 exactly
      o_feat[ij * 2 + 0] = Spread2(c16 & 0xFFu) ^ 0x80808080u;  // ch 0-3
      o_feat[ij * 2 + 1] = Spread2(c16 >>   8u) ^ 0x80808080u;  // ch 4-7
    }
  }

  // G1: 12ch x 4b = 48b per cell
  int2 g1_xy = int2(floor(uv * g1_dim - 0.5));
  int g1_x[2];
  g1_x[0] = max(g1_xy.x, 0);
  g1_x[1] = min(g1_xy.x + 1, g1_dim - 1);
  int g1_y[2];
  g1_y[0] = max(g1_xy.y, 0);
  g1_y[1] = min(g1_xy.y + 1, g1_dim - 1);
  float2 fr = frac(uv * g1_dim - 0.5);
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
      uint w0 = g1[feature_level].Load(bitpos >> 5);
      uint w1 = g1[feature_level].Load((bitpos >> 5) + 1u);
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
    o_feat[8 + q] = PackS8(int4(round(float4(
      g1_blend[q * 4 + 0], g1_blend[q * 4 + 1],
      g1_blend[q * 4 + 2], g1_blend[q * 4 + 3]) * 128.0f)));
  }

  // 12 triangular waves
  float2 cpos = uv * float(NTCCBV.dim >> lod);
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
    o_feat[11 + q] = PackS8(int4(round(float4(
      pe[q * 4 + 0], pe[q * 4 + 1], pe[q * 4 + 2], pe[q * 4 + 3]) * 128.0f)));
  }

  // Lane 56: lod scalar; lanes 57-59: zero pad (must match W0's zero pad rows).
  o_feat[14] = PackS8(int4(int(round(lod / 8.0f * 128.0f)), 0, 0, 0));
  o_feat[15] = 0u;
}

float hardgelu(float x)
{
  if (x < -1.5f) return 0.0f;
  else if (x < 1.5f) return (x / 3.0f) * (x + 1.5f);
  return x;
}

float4 hardgelu4(float4 x)
{
  return select(x < -1.5f, 0.0.xxxx, select(x < 1.5f, (x / 3.0f) * (x + 1.5f), x));
}

PixelShaderOutput ps_main(PixelShaderInput p_in)
{
  PixelShaderOutput p_out;

  // Constants

  float3 directional_light = float3(0.0, 0.0, 1.0);
  float3 light_color = float3(1.0, 1.0, 1.0);
  float3 f0 = float3(0.04, 0.04, 0.04);

  // Samples

  uint feat[64 / 4];
  uint W0x[64 / 4];
  uint W1x[64 / 4];
  float Woutx[12];
  GetFeaturesPacked(p_in.uv, feat);

  [loop]
  for (int i = 0; i < 64; i += 4)
  {
    int4 acc = int4(0, 0, 0, 0);
    [unroll]
    for (int j = 0; j < 64 / 4; j += 4)
    {
      uint4 W0vx = W0.Load((i + 0) * (64 / 16) + (j / 4));
      uint4 W0vy = W0.Load((i + 1) * (64 / 16) + (j / 4));
      uint4 W0vz = W0.Load((i + 2) * (64 / 16) + (j / 4));
      uint4 W0vw = W0.Load((i + 3) * (64 / 16) + (j / 4));
      [unroll]
      for (int c = 0; c < 4; c++)
      {
        acc.x = dot4add_i8packed(W0vx[c], feat[j + c], acc.x);
        acc.y = dot4add_i8packed(W0vy[c], feat[j + c], acc.y);
        acc.z = dot4add_i8packed(W0vz[c], feat[j + c], acc.z);
        acc.w = dot4add_i8packed(W0vw[c], feat[j + c], acc.w);
      }
    }
    float4 facc = float4(acc) * W0_scale.Load(i / 4);
    facc = hardgelu4(facc) * NTCCBV.rcp_s_a1;
    int4 unpacked = int4(round(facc));
    W0x[i / 4] = PackS8(unpacked);
  }
  [loop]
  for (int i = 0; i < 64; i += 4)
  {
    int4 acc = int4(0, 0, 0, 0);
    [unroll]
    for (int j = 0; j < 64 / 4; j += 4)
    {
      uint4 W1vx = W1.Load((i + 0) * (64 / 16) + (j / 4));
      uint4 W1vy = W1.Load((i + 1) * (64 / 16) + (j / 4));
      uint4 W1vz = W1.Load((i + 2) * (64 / 16) + (j / 4));
      uint4 W1vw = W1.Load((i + 3) * (64 / 16) + (j / 4));
      [unroll]
      for (int c = 0; c < 4; c++)
      {
        acc.x = dot4add_i8packed(W1vx[c], W0x[j + c], acc.x);
        acc.y = dot4add_i8packed(W1vy[c], W0x[j + c], acc.y);
        acc.z = dot4add_i8packed(W1vz[c], W0x[j + c], acc.z);
        acc.w = dot4add_i8packed(W1vw[c], W0x[j + c], acc.w);
      }
    }
    float4 facc = float4(acc) * W1_scale.Load(i / 4);
    facc = hardgelu4(facc) * NTCCBV.rcp_s_a2;
    int4 unpacked = int4(round(facc));
    W1x[i / 4] = PackS8(unpacked);
  }
  [loop]
  for (int i = 0; i < 9; i += 4)
  {
    int4 acc = int4(0, 0, 0, 0);
    [unroll]
    for (int j = 0; j < 64 / 4; j += 4)
    {
      uint4 Woutvx = Wout.Load((i + 0) * (64 / 16) + (j / 4));
      uint4 Woutvy = Wout.Load((i + 1) * (64 / 16) + (j / 4));
      uint4 Woutvz = Wout.Load((i + 2) * (64 / 16) + (j / 4));
      uint4 Woutvw = Wout.Load((i + 3) * (64 / 16) + (j / 4));
      [unroll]
      for (int c = 0; c < 4; c++)
      {
        acc.x = dot4add_i8packed(Woutvx[c], W1x[j + c], acc.x);
        acc.y = dot4add_i8packed(Woutvy[c], W1x[j + c], acc.y);
        acc.z = dot4add_i8packed(Woutvz[c], W1x[j + c], acc.z);
        acc.w = dot4add_i8packed(Woutvw[c], W1x[j + c], acc.w);
      }
    }
    float4 facc = float4(acc) * Wout_scale.Load(i / 4);
    Woutx[i + 0] = facc.x;
    Woutx[i + 1] = facc.y;
    Woutx[i + 2] = facc.z;
    Woutx[i + 3] = facc.w;
  }

  float3 ntc_albedo = float3(Woutx[0], Woutx[1], Woutx[2]);
  float ntc_ao = Woutx[3];
  float ntc_displacement = Woutx[4];
  float3 ntc_normal = float3(Woutx[5], Woutx[6], Woutx[7]);
  float ntc_roughness = Woutx[8];

  float4 bitangent_view = float4(cross(p_in.normal_view.xyz, p_in.tangent_view.xyz), 0.0);
  matrix tbn_view = transpose(matrix(p_in.tangent_view, bitangent_view, p_in.normal_view, float4(0.0, 0.0, 0.0, 1.0)));
  float4 surface_normal = float4(ntc_normal * 2.0 - 1.0, 0.0);
  float4 view_normal = mul(tbn_view, surface_normal);
  float3 normal = view_normal.rgb;
  float perceptual_roughness = ntc_roughness;
  float3 albedo = ntc_albedo;

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