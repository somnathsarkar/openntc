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
  int g0_bits_per_channel;
  int g1_bits_per_channel;
  int g0_channels;
  int g1_channels;
  int dim;
  int mip_count;
  float rcp_s_a1;
  float rcp_s_a2;
};

#if defined(BPP_0_2)
#define G0_BITS 2
#define G1_BITS 4
#define G0_CHANNELS 8
#define G1_CHANNELS 12
#elif defined(BPP_0_5)
#define G0_BITS 4
#define G1_BITS 4
#define G0_CHANNELS 12
#define G1_CHANNELS 20
#else
#error Undefined profile
#endif

#define MAX_LEVELS 5

#define HIDDEN_DIM 64
#define OUT_DIM 9
#define OUT_DIM_PADDED 12
#define POS_ENC_DIM 12
#define FEATURE_DIM_PADDED ((((4 * G0_CHANNELS + G1_CHANNELS + POS_ENC_DIM + 1) + 15) / 16) * 16)
#define FEAT_UINTS (FEATURE_DIM_PADDED / 4)
#define HIDDEN_UINTS (HIDDEN_DIM / 4)
#define G0_FEAT_BASE_UINT 0
#define G1_FEAT_BASE_UINT ((4 * G0_CHANNELS) / 4)
#define POSENC_FEAT_BASE_UINT (G1_FEAT_BASE_UINT + G1_CHANNELS / 4)
#define LOD_FEAT_UINT (POSENC_FEAT_BASE_UINT + POS_ENC_DIM / 4)

// G0/G1 are structured as multi-channel cells, where each channel has a certain amount of bits
//  Based on our profiles we are guaranteed to have cells aligned at certain bit boundaries,
//  this is written with that assumption in mind. Writing arbitrary bits per channel or channels
//  will result in unexpected behavior.
#define CELL_PHASE_MAX(cell_bits) ((cell_bits) % 32 == 0 ? 0 : ((cell_bits) % 16 == 0 ? 16 : 24))
#define CELL_SPAN_UINTS(cell_bits) ((CELL_PHASE_MAX(cell_bits) + (cell_bits) + 31) / 32)
#define G0_CELL_BITS (G0_CHANNELS * G0_BITS)
#define G1_CELL_BITS (G1_CHANNELS * G1_BITS)
#define G0_SPAN_UINTS CELL_SPAN_UINTS(G0_CELL_BITS)
#define G1_SPAN_UINTS CELL_SPAN_UINTS(G1_CELL_BITS)

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

// Extract channel k: n contiguous bits from w0, w1, w2 contiguous uints at offset 'phase'.
uint ExtractCellChannel(uint w0, uint w1, uint w2, uint phase, int k, uint n)
{
  uint bit = phase + (uint)k * n;
  uint w = bit < 32u ? w0 : (bit < 64u ? w1 : w2);
  return (w >> (bit & 31u)) & ((1u << n) - 1u);
}

#ifdef COOP_SUPPORT
void GetFeaturesPacked(Buffer<uint> g0, Buffer<uint> g1, NTC NTCCBV, float2 uv, float2 pos_screen, out vector<uint, FEAT_UINTS> o_feat)
#else
void GetFeaturesPacked(Buffer<uint> g0, Buffer<uint> g1, NTC NTCCBV, float2 uv, float2 pos_screen, out uint o_feat[FEAT_UINTS])
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

  // G0: G0_CHANNELS x G0_BITS bits per cell (8ch x 2b = 16b for BPP_0_2)
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

      // Cell is the index of the G0_CELL_BITS-wide segment (inside the grid level)
      //  containing all data for one feature grid point
      uint cell = (uint)(g0_y[i] * g0_dim + g0_x[j]);

      uint bitpos = cell * G0_CELL_BITS;
      uint base = g0_off + bitpos / 32u;
      uint w0 = g0.Load(base);
      uint w1 = (G0_SPAN_UINTS > 1) ? g0.Load(base + 1u) : 0u;
      uint w2 = (G0_SPAN_UINTS > 2) ? g0.Load(base + 2u) : 0u;
      uint phase = bitpos & 31u;

      [unroll]
      for (int q = 0; q < G0_CHANNELS / 4; q++)
      {
        uint packed = 0u;
        [unroll]
        for (int c = 0; c < 4; c++)
        {
          uint v = ExtractCellChannel(w0, w1, w2, phase, q * 4 + c, G0_BITS);
          packed |= ((v << (8u - G0_BITS)) ^ 0x80u) << (c * 8);
        }
        o_feat[G0_FEAT_BASE_UINT + ij * (G0_CHANNELS / 4) + q] = packed;
      }
    }
  }

  // G1: G1_CHANNELS x G1_BITS bits per cell
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

      // Cell is the index of the G1_CELL_BITS-wide segment (inside the grid level)
      //  containing all data for one feature grid point
      uint cell = (uint)(g1_y[i] * g1_dim + g1_x[j]);

      // Bitpos is the index of the first bit inside the grid level for this segment
      uint bitpos = cell * G1_CELL_BITS;
      uint base = g1_off + bitpos / 32u;
      uint w0 = g1.Load(base);
      uint w1 = (G1_SPAN_UINTS > 1) ? g1.Load(base + 1u) : 0u;
      uint w2 = (G1_SPAN_UINTS > 2) ? g1.Load(base + 2u) : 0u;
      uint phase = bitpos & 31u;

      const float half_range = (float)(1u << (G1_BITS - 1));
      [unroll]
      for (int k = 0; k < G1_CHANNELS; k++)
      {
        uint v = ExtractCellChannel(w0, w1, w2, phase, k, G1_BITS);

        // Remap G1_BITS-wide integers to [-1, 1) floats
        g1_blend[k] += m * (((float)v - half_range) / half_range);
      }
    }
  }

  [unroll]
  for (int q = 0; q < G1_CHANNELS / 4; q++)
  {
    // Remap [-1, 1) floats to 8-bit signed integers
    o_feat[G1_FEAT_BASE_UINT + q] = pack_clamp_s8(int4(round(float4(
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
  for (int q = 0; q < POS_ENC_DIM / 4; q++)
  {
    o_feat[POSENC_FEAT_BASE_UINT + q] = pack_clamp_s8(int4(round(float4(
      pe[q * 4 + 0], pe[q * 4 + 1], pe[q * 4 + 2], pe[q * 4 + 3]) * 128.0f)));
  }

  // Fill out lod feature, then zero pad up to FEATURE_DIM_PADDED
  o_feat[LOD_FEAT_UINT] = pack_clamp_s8(int4(int(round(lod / float(NTCCBV.mip_count - 1) * 128.0f)), 0, 0, 0));
  [unroll]
  for (int z = LOD_FEAT_UINT + 1; z < FEAT_UINTS; z++)
    o_feat[z] = 0u;
}

#ifdef COOP_SUPPORT
vector<float, HIDDEN_DIM> hardgelu_coop(vector<float, HIDDEN_DIM> x)
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
  vector<uint, FEAT_UINTS> feat,
  out vector<float, OUT_DIM_PADDED> o_Woutx)
{
  using namespace dx::linalg;

  vector<uint, HIDDEN_UINTS> W0x;
  vector<uint, HIDDEN_UINTS> W1x;
  const vector<int32_t, HIDDEN_DIM> zero_hidden = (vector<int32_t, HIDDEN_DIM>)0;
  const vector<int32_t, OUT_DIM_PADDED> zero_out = (vector<int32_t, OUT_DIM_PADDED>)0;

  // Coop multiplication.
  typedef Matrix<ComponentType::I8, HIDDEN_DIM, FEATURE_DIM_PADDED, MatrixUse::A, MatrixScope::Thread> W0_t;
  W0_t W0_coop = W0_t::Load<MatrixLayout::RowMajor>(W0, 0, FEATURE_DIM_PADDED);
  InterpretedVector<uint, FEAT_UINTS, ComponentType::I8> feat_coop = MakeInterpretedVector<ComponentType::I8>(feat);
  vector<int32_t, HIDDEN_DIM> W0x_acc = MultiplyAdd<int32_t>(W0_coop, feat_coop, zero_hidden);
  vector<float, HIDDEN_DIM> W0_scale_coop = W0_scale.Load< vector<float, HIDDEN_DIM> >(0);
  vector<float, HIDDEN_DIM> W0x_facc = vector<float, HIDDEN_DIM>(W0x_acc) * W0_scale_coop;
  W0x_facc = hardgelu_coop(W0x_facc) * NTCCBV.rcp_s_a1;
  vector<int32_t, HIDDEN_DIM> W0x_unpacked = vector<int32_t, HIDDEN_DIM>(round(W0x_facc));
  [unroll]
  for (int i = 0 ; i < HIDDEN_DIM; i+=4)
    W0x[i / 4] = pack_clamp_s8(int4(W0x_unpacked[i], W0x_unpacked[i + 1], W0x_unpacked[i + 2], W0x_unpacked[i + 3]));

  typedef Matrix<ComponentType::I8, HIDDEN_DIM, HIDDEN_DIM, MatrixUse::A, MatrixScope::Thread> W1_t;
  W1_t W1_coop = W1_t::Load<MatrixLayout::RowMajor>(W1, 0, HIDDEN_DIM);
  InterpretedVector<uint, HIDDEN_UINTS, ComponentType::I8> W0x_coop = MakeInterpretedVector<ComponentType::I8>(W0x);
  vector<int32_t, HIDDEN_DIM> W1x_acc = MultiplyAdd<int32_t>(W1_coop, W0x_coop, zero_hidden);
  vector<float, HIDDEN_DIM> W1_scale_coop = W1_scale.Load< vector<float, HIDDEN_DIM> >(0);
  vector<float, HIDDEN_DIM> W1x_facc = vector<float, HIDDEN_DIM>(W1x_acc) * W1_scale_coop;
  W1x_facc = hardgelu_coop(W1x_facc) * NTCCBV.rcp_s_a2;
  vector<int32_t, HIDDEN_DIM> W1x_unpacked = vector<int32_t, HIDDEN_DIM>(round(W1x_facc));
  [unroll]
  for (int i = 0 ; i < HIDDEN_DIM; i+=4)
    W1x[i / 4] = pack_clamp_s8(int4(W1x_unpacked[i], W1x_unpacked[i + 1], W1x_unpacked[i + 2], W1x_unpacked[i + 3]));

  typedef Matrix<ComponentType::I8, OUT_DIM_PADDED, HIDDEN_DIM, MatrixUse::A, MatrixScope::Thread> Wout_t;
  Wout_t Wout_coop = Wout_t::Load<MatrixLayout::RowMajor>(Wout, 0, HIDDEN_DIM);
  InterpretedVector<uint, HIDDEN_UINTS, ComponentType::I8> W1x_coop = MakeInterpretedVector<ComponentType::I8>(W1x);
  vector<int32_t, OUT_DIM_PADDED> Woutx_acc = MultiplyAdd<int32_t>(Wout_coop, W1x_coop, zero_out);
  vector<float, OUT_DIM_PADDED> Wout_scale_coop = Wout_scale.Load< vector<float, OUT_DIM_PADDED> >(0);
  o_Woutx = vector<float, OUT_DIM_PADDED>(Woutx_acc) * Wout_scale_coop;
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
  uint feat[FEAT_UINTS],
  out float o_Woutx[OUT_DIM_PADDED])
{
  uint W0x[HIDDEN_UINTS];
  uint W1x[HIDDEN_UINTS];

  // Layer 1
  [loop]
  for (int i = 0; i < HIDDEN_DIM; i += 4)
  {
    int4 acc = int4(0, 0, 0, 0);
    [unroll]
    for (int j = 0; j < FEAT_UINTS; j += 4)
    {
      uint4 W0vx = W0.Load<uint4>((i + 0) * (FEATURE_DIM_PADDED) + (j * 4));
      uint4 W0vy = W0.Load<uint4>((i + 1) * (FEATURE_DIM_PADDED) + (j * 4));
      uint4 W0vz = W0.Load<uint4>((i + 2) * (FEATURE_DIM_PADDED) + (j * 4));
      uint4 W0vw = W0.Load<uint4>((i + 3) * (FEATURE_DIM_PADDED) + (j * 4));
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
  // Layer 2
  [loop]
  for (int i = 0; i < HIDDEN_DIM; i += 4)
  {
    int4 acc = int4(0, 0, 0, 0);
    [unroll]
    for (int j = 0; j < HIDDEN_UINTS; j += 4)
    {
      uint4 W1vx = W1.Load<uint4>((i + 0) * (HIDDEN_DIM) + (j * 4));
      uint4 W1vy = W1.Load<uint4>((i + 1) * (HIDDEN_DIM) + (j * 4));
      uint4 W1vz = W1.Load<uint4>((i + 2) * (HIDDEN_DIM) + (j * 4));
      uint4 W1vw = W1.Load<uint4>((i + 3) * (HIDDEN_DIM) + (j * 4));
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
  // Output layer
  [loop]
  for (int i = 0; i < OUT_DIM; i += 4)
  {
    int4 acc = int4(0, 0, 0, 0);
    [unroll]
    for (int j = 0; j < HIDDEN_UINTS; j += 4)
    {
      uint4 Woutvx = Wout.Load<uint4>((i + 0) * (HIDDEN_DIM) + (j * 4));
      uint4 Woutvy = Wout.Load<uint4>((i + 1) * (HIDDEN_DIM) + (j * 4));
      uint4 Woutvz = Wout.Load<uint4>((i + 2) * (HIDDEN_DIM) + (j * 4));
      uint4 Woutvw = Wout.Load<uint4>((i + 3) * (HIDDEN_DIM) + (j * 4));
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