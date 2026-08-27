#ifdef COOP_SUPPORT
#include <dx/linalg.h>
#endif

struct LightingParams
{
  float exposure;
  float displacement_scale;
  float normal_scale;
  float pad0;
  float2 jitter_px;
  float2 pad1;
  float3 diffuse_sh[9];
};


// Unjitter UVs from TAA to prevent texture blurring
//  Details: https://www.elopezr.com/temporal-aa-and-the-quest-for-the-holy-trail/

float2 UnjitterUv(float2 uv, float2 jitter_px)
{
  return uv + ddx(uv) * jitter_px.x + ddy(uv) * jitter_px.y;
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

#define G0_BITS 2
#define G1_BITS 4
#define G0_CHANNELS 8
#define G1_CHANNELS 12
#define MAX_LEVELS 5
#define PI 3.14159265359

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

float SpecularOcclusion(float NoV, float ao, float perceptual_roughness)
{
  return saturate(pow(NoV + ao, exp2(-16.0 * perceptual_roughness - 1.0)) - 1.0 + ao);
}

float3 IrradianceSh(LightingParams LightingParamsCBV, float3 n)
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

// ACES tonemapping code based on Stephen Hill's (@self_shadow) snippet in BakingLab

static const float3x3 ACESInput = { 0.59719, 0.35458, 0.04823, 0.07600, 0.90834, 0.01566, 0.02840, 0.13383, 0.83777 };
static const float3x3 ACESOutput = { 1.60475, -0.53108, -0.07367, -0.10208,  1.10813, -0.00605, -0.00327, -0.07276, 1.07602 };

float3 RRTAndODTFit(float3 v)
{
  return (v * (v + 0.0245786) - 0.000090537) / (v * (0.983729 * v + 0.4329510) + 0.238081);
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

// Spread packed 4x2 bits (in the bottom 8 bits of input uint b) to 4 8-bit slots in a uint
//  They occupy the top 2 bits of each 8-bit slot.
uint Spread2(uint b)
{
  return ((b & 0x03u) <<  6) | ((b & 0x0Cu) << 12) |
         ((b & 0x30u) << 18) | ((b & 0xC0u) << 24);
}

#ifdef COOP_SUPPORT
void GetFeaturesPacked(Buffer<uint> g0, Buffer<uint> g1, NTC NTCCBV, float2 uv, float2 pos_screen, out vector<uint, 64 / 4> o_feat)
#else
void GetFeaturesPacked(Buffer<uint> g0, Buffer<uint> g1, NTC NTCCBV, float2 uv, float2 pos_screen, out uint o_feat[64 / 4])
#endif
{
#if __SHADER_TARGET_STAGE == __SHADER_STAGE_PIXEL
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
#else
  // No stochastic filtering in vertex shader

  float2 texels_along_major_axis = 0.0.xx;
  float texels_per_vertex_step = NTCCBV.dim / 200.0f;
  float lodab = max(0.0f, log2(texels_per_vertex_step));
  float lodab_clamped = clamp(lodab, 0.0, float(NTCCBV.mip_count - 1));
  int lod = int(round(lodab_clamped));
#endif
  int feature_level = FeatureLevelForLod(lod);

  int fli = (feature_level / 4);
  int flj = (feature_level % 4);
  int g0_dim = NTCCBV.g0_grid_dim[fli][flj];
  int g1_dim = NTCCBV.g1_grid_dim[fli][flj];

  float ign1 = frac(52.9829189 * frac(0.06711056 * (pos_screen.x + 61.0) + 0.00583715 * (pos_screen.y + 37.0)));
  float2 uv_jittered = uv + (texels_along_major_axis / NTCCBV.dim) * (ign1 - 0.5);

  // G0: 8ch x 2b = 16b per cell
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

      // Cell is the index of the 16-bit segment (inside the grid level)
      //  containing all data for one feature grid point
      uint cell = (uint)(g0_y[i] * g0_dim + g0_x[j]);

      // Word is the contents of the 32-bit segment containing the cell in the combined 
      //  feature grid levels. 1 word contains 2 cells
      uint word = g0.Load(g0_off + (cell / 2u));

      // c16 is the contents of the cell above
      uint c16 = (word >> ((cell & 1u) * 16u)) & 0xFFFFu;
      
      // Unpack 2-bit integers to 8-bit by placing each in the upper 2-bits of a 8-bit slot
      //  and xoring each 8-bit slot with 0x80. (Corresponds to (k << 6) - 128)
      o_feat[ij * 2 + 0] = Spread2(c16 & 0xFFu) ^ 0x80808080u;
      o_feat[ij * 2 + 1] = Spread2(c16 >> 8u) ^ 0x80808080u;
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

      // Cell is the index of the 48-bit segment (inside the grid level)
      //  containing all data for one feature grid point
      uint cell = (uint)(g1_y[i] * g1_dim + g1_x[j]);

      // Bitpos is the index of the first bit inside the grid level for this segment
      uint bitpos = cell * 48u;

      // Word0/1 are the two 32-bit values that contain the 48-bit segment 
      uint word0 = g1.Load(g1_off + (bitpos / 32u));
      uint word1 = g1.Load(g1_off + (bitpos / 32u) + 1u);

      // Each segment starts at 0 or 16, alternating
      uint shift = bitpos & 31u;
      [unroll]
      for (int k = 0; k < G1_CHANNELS; k++)
      {
        uint bit = shift + (uint)k * 4u;
        uint nib = ((bit < 32u ? word0 : word1) >> (bit & 31u)) & 0xFu;

        // Remap 4 bit integers to [-1, 1) floats
        g1_blend[k] += m * (((float)nib - 8.0f) / 8.0f);
      }
    }
  }

  [unroll]
  for (int q = 0; q < 3; q++)
  {
    // Remap [-1, 1) floats to 8-bit signed integers
    o_feat[8 + q] = pack_clamp_s8(int4(round(float4(
      g1_blend[q * 4 + 0], g1_blend[q * 4 + 1],
      g1_blend[q * 4 + 2], g1_blend[q * 4 + 3]) * 128.0f)));
  }

  // 12 triangular waves
  // Compare against training code, this should probably be part of a shared header
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

  // Zero pad slots 57 - 63
  o_feat[14] = pack_clamp_s8(int4(int(round(lod / float(NTCCBV.mip_count - 1) * 128.0f)), 0, 0, 0));
  o_feat[15] = 0u;
}

#ifdef COOP_SUPPORT
vector<float, 64> hardgelu_coop(vector<float, 64> x)
{
  return select(x < -1.5f, 0.0, select(x < 1.5f, (x / 3.0f) * (x + 1.5f), x));
}

void PerformNTCInference(
  ByteAddressBuffer W0,
  ByteAddressBuffer W1,
  ByteAddressBuffer Wout,
  ByteAddressBuffer W0_scale,
  ByteAddressBuffer W1_scale,
  ByteAddressBuffer Wout_scale,
  NTC NTCCBV,
  vector<uint, 64 / 4> feat,
  out vector<float, 12> o_Woutx)
{
  using namespace dx::linalg;

  vector<uint, 64 / 4> W0x;
  vector<uint, 64 / 4> W1x;
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
  o_Woutx = vector<float, 12>(Woutx_acc) * Wout_scale_coop;
}

#else

float4 hardgelu4(float4 x)
{
  return select(x < -1.5f, 0.0.xxxx, select(x < 1.5f, (x / 3.0f) * (x + 1.5f), x));
}

int8_t4_packed PackS8(int32_t4 unpacked)
{
  return pack_clamp_s8(unpacked);
}

void PerformNTCInference(
  ByteAddressBuffer W0,
  ByteAddressBuffer W1,
  ByteAddressBuffer Wout,
  ByteAddressBuffer W0_scale,
  ByteAddressBuffer W1_scale,
  ByteAddressBuffer Wout_scale,
  NTC NTCCBV,
  uint feat[64 / 4],
  out float o_Woutx[12])
{
  uint W0x[64 / 4];
  uint W1x[64 / 4];

  [loop]
  for (int i = 0; i < 64; i += 4)
  {
    int4 acc = int4(0, 0, 0, 0);
    [unroll]
    for (int j = 0; j < 64 / 4; j += 4)
    {
      uint4 W0vx = W0.Load<uint4>((i + 0) * (64) + (j * 4));
      uint4 W0vy = W0.Load<uint4>((i + 1) * (64) + (j * 4));
      uint4 W0vz = W0.Load<uint4>((i + 2) * (64) + (j * 4));
      uint4 W0vw = W0.Load<uint4>((i + 3) * (64) + (j * 4));
      [unroll]
      for (int c = 0; c < 4; c++)
      {
        acc.x = dot4add_i8packed(W0vx[c], feat[j + c], acc.x);
        acc.y = dot4add_i8packed(W0vy[c], feat[j + c], acc.y);
        acc.z = dot4add_i8packed(W0vz[c], feat[j + c], acc.z);
        acc.w = dot4add_i8packed(W0vw[c], feat[j + c], acc.w);
      }
    }
    float4 facc = float4(acc) * W0_scale.Load<float4>(i * 4);
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
      uint4 W1vx = W1.Load<uint4>((i + 0) * (64) + (j * 4));
      uint4 W1vy = W1.Load<uint4>((i + 1) * (64) + (j * 4));
      uint4 W1vz = W1.Load<uint4>((i + 2) * (64) + (j * 4));
      uint4 W1vw = W1.Load<uint4>((i + 3) * (64) + (j * 4));
      [unroll]
      for (int c = 0; c < 4; c++)
      {
        acc.x = dot4add_i8packed(W1vx[c], W0x[j + c], acc.x);
        acc.y = dot4add_i8packed(W1vy[c], W0x[j + c], acc.y);
        acc.z = dot4add_i8packed(W1vz[c], W0x[j + c], acc.z);
        acc.w = dot4add_i8packed(W1vw[c], W0x[j + c], acc.w);
      }
    }
    float4 facc = float4(acc) * W1_scale.Load<float4>(i * 4);
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
      uint4 Woutvx = Wout.Load<uint4>((i + 0) * (64) + (j * 4));
      uint4 Woutvy = Wout.Load<uint4>((i + 1) * (64) + (j * 4));
      uint4 Woutvz = Wout.Load<uint4>((i + 2) * (64) + (j * 4));
      uint4 Woutvw = Wout.Load<uint4>((i + 3) * (64) + (j * 4));
      [unroll]
      for (int c = 0; c < 4; c++)
      {
        acc.x = dot4add_i8packed(Woutvx[c], W1x[j + c], acc.x);
        acc.y = dot4add_i8packed(Woutvy[c], W1x[j + c], acc.y);
        acc.z = dot4add_i8packed(Woutvz[c], W1x[j + c], acc.z);
        acc.w = dot4add_i8packed(Woutvw[c], W1x[j + c], acc.w);
      }
    }
    float4 facc = float4(acc) * Wout_scale.Load<float4>(i * 4);
    o_Woutx[i + 0] = facc.x;
    o_Woutx[i + 1] = facc.y;
    o_Woutx[i + 2] = facc.z;
    o_Woutx[i + 3] = facc.w;
  }
}
#endif