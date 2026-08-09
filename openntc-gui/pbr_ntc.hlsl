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

struct NTC
{
  int4 g0_grid_dim;
  int4 g1_grid_dim;
  int g0_bytes_per_channel;
  int g1_bytes_per_channel;
  int g0_channels;
  int g1_channels;
  int dim;
};

ConstantBuffer<NTC> NTCCBV : register(b1, space0);

Buffer<float16_t> g0[4] : register(t0, space0);
Buffer<float> g1[4] : register(t4, space0);
Buffer<float4> W0 : register(t8, space0);
Buffer<float4> W1 : register(t9, space0);
Buffer<float4> Wout : register(t10, space0);

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

void GetFeatures(float2 uv, out float o_feat[60])
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
      for (int k = 0; k < NTCCBV.g0_channels; k++)
      {
        int g0_idx = g0_y[i] * NTCCBV.g0_grid_dim[feature_level] * NTCCBV.g0_channels + g0_x[j] * NTCCBV.g0_channels + k;
        o_feat[ij * NTCCBV.g0_channels + k] = float(g0[feature_level].Load(g0_idx));
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
      for (int k = 0; k < NTCCBV.g1_channels; k++)
      {
        int g1_idx = g1_y[i] * NTCCBV.g1_grid_dim[feature_level] * NTCCBV.g1_channels + g1_x[j] * NTCCBV.g1_channels + k;
        g1_contrib[k] += m * float(g1[feature_level].Load(g1_idx));
      }
    }
  }

  for (int i = 0; i < NTCCBV.g1_channels; i++)
    o_feat[4 * NTCCBV.g0_channels + i] = g1_contrib[i];

  int periods[3] = {8, 4, 2};
  int pos_off = 4 * NTCCBV.g0_channels + NTCCBV.g1_channels;
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

  for (int i = 57; i < 60; i++)
    o_feat[i] = 0.0f;
}

float hardgelu(float x)
{
  if (x < -1.5f) return 0.0f;
  else if (x < 1.5f) return (x / 3.0f) * (x + 1.5f);
  return x;
}

PixelShaderOutput ps_main(PixelShaderInput p_in)
{
  PixelShaderOutput p_out;

  // Constants

  float3 directional_light = float3(0.0, 0.0, 1.0);
  float3 light_color = float3(1.0, 1.0, 1.0);
  float3 f0 = float3(0.04, 0.04, 0.04);

  // Samples

  float feat[60];
  float W0x[64];
  float W1x[64];
  float Woutx[9];
  GetFeatures(p_in.uv, feat);

  [unroll]
  for (int i = 0; i < 64; i++)
  {
    W0x[i] = 0.0f;
    W1x[i] = 0.0f;
  }
  [unroll]
  for (int i = 0; i < 9; i++)
    Woutx[i] = 0.0f;
  for (int i = 0; i < 64; i++)
  {
    [unroll]
    for (int j = 0; j < 60 / 4; j++)
    {
      float4 f4 = float4(feat[j * 4], feat[j * 4 + 1], feat[j * 4 + 2], feat[j * 4 + 3]); 
      W0x[i] += dot(W0.Load(i * (60 / 4) + j), f4);
    }
    W0x[i] = hardgelu(W0x[i]);
  }
  for (int i = 0; i < 64; i++)
  {
    [unroll]
    for (int j = 0; j < 64 / 4; j++)
    {
      float4 f4 = float4(W0x[j * 4], W0x[j * 4 + 1], W0x[j * 4 + 2], W0x[j * 4 + 3]);
      W1x[i] += dot(W1.Load(i * (64 / 4) + j), f4);
    }
    W1x[i] = hardgelu(W1x[i]);
  }
  for (int i = 0; i < 9; i++)
  {
    [unroll]
    for (int j = 0; j < 64 / 4; j++)
    {
      float4 f4 = float4(W1x[j * 4], W1x[j * 4 + 1], W1x[j * 4 + 2], W1x[j * 4 + 3]);
      Woutx[i] += dot(Wout.Load(i * (64 / 4) + j), f4);
    }
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