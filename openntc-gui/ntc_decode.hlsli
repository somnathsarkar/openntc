#ifndef __NTC_DECODE_HLSLI__
#define __NTC_DECODE_HLSLI__

#ifdef COOP_SUPPORT
#include <dx/linalg.h>
#endif

namespace openntc
{

struct NTC
{
  int4 g0_grid_dim_[2];
  int4 g1_grid_dim_[2];
  uint4 g0_offset_[2];
  uint4 g1_offset_[2];
  int g0_bits_per_channel_;
  int g1_bits_per_channel_;
  int g0_channels_;
  int g1_channels_;
  int dim_;
  int mip_count_;
  float rcp_s_a1_;
  float rcp_s_a2_;
  int channel_count_;
  int3 pad0_;
  uint4 channel_semantics_[4];
};

// NOTE: Match this against openntc::Semantic in libopenntc.h
enum class Semantic
{
  None = 0,

  Albedo = 1,
  Alpha = 2,
  Displacement = 3,
  Emissive = 4,
  Gloss = 5,
  Metallic = 6,
  Normal = 7,
  AO = 8,
  Roughness = 9,
  Specular = 10,
  Transmission = 11,
};

// Map channel to semantic
Semantic ChannelSemantic(NTC NTCCBV, int i)
{
  return (Semantic)NTCCBV.channel_semantics_[i / 4][i % 4];
}

// Semantic to bit in Materal::present_
uint SemanticBit(Semantic sem)
{
  return 1u << (uint)sem;
}

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
#elif defined(BPP_1_0)
#define G0_BITS 2
#define G1_BITS 4
#define G0_CHANNELS 12
#define G1_CHANNELS 10
#elif defined(BPP_2_25)
#define G0_BITS 4
#define G1_BITS 4
#define G0_CHANNELS 16
#define G1_CHANNELS 12
#else
#error Undefined profile
#endif

#define MAX_LEVELS 6

#define HIDDEN_DIM 64
#define OUT_DIM 9
#define OUT_DIM_PADDED 12
#define POS_ENC_DIM 12
#define FEATURE_DIM_PADDED ((((4 * G0_CHANNELS + G1_CHANNELS + POS_ENC_DIM + 1) + 15) / 16) * 16)
#define FEAT_UINTS (FEATURE_DIM_PADDED / 4)
#define HIDDEN_UINTS (HIDDEN_DIM / 4)

#define DEC_W0_OFFSET 0
#define DEC_W1_OFFSET (DEC_W0_OFFSET + HIDDEN_DIM * FEATURE_DIM_PADDED)
#define DEC_WOUT_OFFSET (DEC_W1_OFFSET + HIDDEN_DIM * HIDDEN_DIM)
#define DEC_W0_SCALE_OFFSET (DEC_WOUT_OFFSET + OUT_DIM_PADDED * HIDDEN_DIM)
#define DEC_W1_SCALE_OFFSET (DEC_W0_SCALE_OFFSET + HIDDEN_DIM * 4)
#define DEC_WOUT_SCALE_OFFSET (DEC_W1_SCALE_OFFSET + HIDDEN_DIM * 4)
#define DEC_SIZE (DEC_WOUT_SCALE_OFFSET + OUT_DIM_PADDED * 4)
#define G0_FEAT_BASE_SLOT 0
#define G1_FEAT_BASE_SLOT (4 * G0_CHANNELS)
#define POSENC_FEAT_BASE_SLOT (G1_FEAT_BASE_SLOT + G1_CHANNELS)
#define LOD_FEAT_SLOT (POSENC_FEAT_BASE_SLOT + POS_ENC_DIM)

// Insert a value into a slot in the feature vector by OR-ing the corresponding uint
#define PUT_FEAT_SLOT(o_feat, slot, byte_val) (o_feat[(slot) / 4] |= (byte_val) << (((slot) % 4) * 8))

uint ClampS8(float f)
{
  return (uint)clamp(int(round(f)), -128, 127) & 0xFFu;
}

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

// Interleaved Gradient Noise - "Next Generation Post-Processing in Call of Duty Advanced Warfare"
float InterleavedGradientNoise(float2 pos_screen)
{
  return frac(52.9829189 * frac(0.06711056 * pos_screen.x + 0.00583715 * pos_screen.y));
}

// For a uv sample in the pixel shader, computes the LOD using screen-space derivatives.
//  For anisotropic filtering, use the minor axis to compute LOD, then save the UV space
//  spanned by the major axis
float ComputeLod(NTC NTCCBV, float2 uv, out float2 o_footprint)
{
#if __SHADER_TARGET_STAGE != __SHADER_STAGE_PIXEL
  o_footprint = 0.0.xx;
  return 0.0;
#else
  float2 dUvdX = ddx(uv) * NTCCBV.dim_;
  float2 dUvdY = ddy(uv) * NTCCBV.dim_;
  float gUvdX = dot(dUvdX, dUvdX);
  float gUvdY = dot(dUvdY, dUvdY);
  o_footprint = ((gUvdX > gUvdY) ? dUvdX : dUvdY) / NTCCBV.dim_;
  float minor_axis = sqrt(min(gUvdX, gUvdY));
  float major_axis = sqrt(max(gUvdX, gUvdY));
  float max_aniso = 8.0f;
  minor_axis = max(minor_axis, major_axis / max_aniso);
  return clamp(log2(minor_axis), 0.0, float(NTCCBV.mip_count_ - 1));
#endif
}

// NTCs can only sample from integral filter levels efficiently.
// For a float lod a with frac(lod) = p, select floor(a) with probability 1 - p
// and ceil(a) with probability p
int StochasticFilterLod(float lod, float2 pos_screen)
{
  float ign = InterleavedGradientNoise(pos_screen);
  return int(lod) + (ign < frac(lod) ? 1 : 0);
}

// Stochastic anisotropic filtering: Jitter the uv at random in the uv space spanned
// by it along its major axis (Computed in ComputeLod)
float2 StochasticFilterUv(float2 uv, float2 footprint, float2 pos_screen)
{
  float ign = InterleavedGradientNoise(pos_screen + float2(61.0, 37.0));
  return uv + footprint * (ign - 0.5);
}

#ifdef COOP_SUPPORT
void GetFeaturesPacked(
  Buffer<uint> g0,
  Buffer<uint> g1,
  NTC NTCCBV,
  float2 uv,
  int lod,
  out vector<uint, FEAT_UINTS> o_feat)
#else
void GetFeaturesPacked(
  Buffer<uint> g0,
  Buffer<uint> g1,
  NTC NTCCBV,
  float2 uv,
  int lod,
  out uint o_feat[FEAT_UINTS])
#endif
{
  lod = clamp(lod, 0, NTCCBV.mip_count_ - 1);
  [unroll]
  for (int z = 0; z < FEAT_UINTS; z++)
    o_feat[z] = 0u;

  int feature_level = FeatureLevelForLod(lod);

  int fli = (feature_level / 4);
  int flj = (feature_level % 4);
  int g0_dim = NTCCBV.g0_grid_dim_[fli][flj];
  int g1_dim = NTCCBV.g1_grid_dim_[fli][flj];

  // G0: G0_CHANNELS x G0_BITS bits per cell (8ch x 2b = 16b for BPP_0_2)
  int2 g0_xy = int2(floor(uv * g0_dim - 0.5));
  int g0_x[2];
  g0_x[0] = max(g0_xy.x, 0);
  g0_x[1] = min(g0_xy.x + 1, g0_dim - 1);
  int g0_y[2];
  g0_y[0] = max(g0_xy.y, 0);
  g0_y[1] = min(g0_xy.y + 1, g0_dim - 1);

  uint g0_off = NTCCBV.g0_offset_[fli][flj] / 4;
  uint g1_off = NTCCBV.g1_offset_[fli][flj] / 4;

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
      for (int k = 0; k < G0_CHANNELS; k++)
      {
        uint v = ExtractCellChannel(w0, w1, w2, phase, k, G0_BITS);
        // xor with 0x80 is just subtracting 128 on a 8-bit uint without underflow
        PUT_FEAT_SLOT(o_feat, G0_FEAT_BASE_SLOT + ij * G0_CHANNELS + k, (v << (8u - G0_BITS)) ^ 0x80u);
      }
    }
  }

  // G1: G1_CHANNELS x G1_BITS bits per cell
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
  for (int k = 0; k < G1_CHANNELS; k++)
  {
    // Remap [-1, 1) floats to 8-bit signed integers
    PUT_FEAT_SLOT(o_feat, G1_FEAT_BASE_SLOT + k, ClampS8(g1_blend[k] * 128.0f));
  }

  // 12 triangular waves
  // Compare against training code, this should probably be part of a shared header
  float2 cpos = uv * float(NTCCBV.dim_ >> lod);
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
  for (int k = 0; k < POS_ENC_DIM; k++)
  {
    PUT_FEAT_SLOT(o_feat, POSENC_FEAT_BASE_SLOT + k, ClampS8(pe[k] * 128.0f));
  }

  PUT_FEAT_SLOT(o_feat, LOD_FEAT_SLOT, ClampS8(lod / float(NTCCBV.mip_count_ - 1) * 128.0f));
}

#ifdef COOP_SUPPORT
vector<float, HIDDEN_DIM> hardgelu_coop(vector<float, HIDDEN_DIM> x)
{
  return select(x < -1.5f, 0.0, select(x < 1.5f, (x / 3.0f) * (x + 1.5f), x));
}

void PerformInference(
  ByteAddressBuffer decoder,
  NTC NTCCBV,
  vector<uint, FEAT_UINTS> feat,
  out vector<float, OUT_DIM_PADDED> o_Woutx)
{
  using namespace dx::linalg;

  vector<uint, HIDDEN_UINTS> W0x;
  vector<uint, HIDDEN_UINTS> W1x;
  const vector<int32_t, HIDDEN_DIM> zero_hidden = (vector<int32_t, HIDDEN_DIM>)0;
  const vector<int32_t, OUT_DIM_PADDED> zero_out = (vector<int32_t, OUT_DIM_PADDED>)0;

  // Layer 1
  typedef Matrix<ComponentType::I8, HIDDEN_DIM, FEATURE_DIM_PADDED, MatrixUse::A, MatrixScope::Thread> W0_t;
  W0_t W0_coop = W0_t::Load<MatrixLayout::RowMajor>(decoder, DEC_W0_OFFSET, FEATURE_DIM_PADDED);
  InterpretedVector<uint, FEAT_UINTS, ComponentType::I8> feat_coop = MakeInterpretedVector<ComponentType::I8>(feat);
  vector<int32_t, HIDDEN_DIM> W0x_acc = MultiplyAdd<int32_t>(W0_coop, feat_coop, zero_hidden);
  vector<float, HIDDEN_DIM> W0_scale_coop = decoder.Load< vector<float, HIDDEN_DIM> >(DEC_W0_SCALE_OFFSET);
  vector<float, HIDDEN_DIM> W0x_facc = vector<float, HIDDEN_DIM>(W0x_acc) * W0_scale_coop;
  W0x_facc = hardgelu_coop(W0x_facc) * NTCCBV.rcp_s_a1_;
  vector<int32_t, HIDDEN_DIM> W0x_unpacked = vector<int32_t, HIDDEN_DIM>(round(W0x_facc));
  [unroll]
  for (int i = 0 ; i < HIDDEN_DIM; i+=4)
    W0x[i / 4] = pack_clamp_s8(int4(W0x_unpacked[i], W0x_unpacked[i + 1], W0x_unpacked[i + 2], W0x_unpacked[i + 3]));

  // Layer 2
  typedef Matrix<ComponentType::I8, HIDDEN_DIM, HIDDEN_DIM, MatrixUse::A, MatrixScope::Thread> W1_t;
  W1_t W1_coop = W1_t::Load<MatrixLayout::RowMajor>(decoder, DEC_W1_OFFSET, HIDDEN_DIM);
  InterpretedVector<uint, HIDDEN_UINTS, ComponentType::I8> W0x_coop = MakeInterpretedVector<ComponentType::I8>(W0x);
  vector<int32_t, HIDDEN_DIM> W1x_acc = MultiplyAdd<int32_t>(W1_coop, W0x_coop, zero_hidden);
  vector<float, HIDDEN_DIM> W1_scale_coop = decoder.Load< vector<float, HIDDEN_DIM> >(DEC_W1_SCALE_OFFSET);
  vector<float, HIDDEN_DIM> W1x_facc = vector<float, HIDDEN_DIM>(W1x_acc) * W1_scale_coop;
  W1x_facc = hardgelu_coop(W1x_facc) * NTCCBV.rcp_s_a2_;
  vector<int32_t, HIDDEN_DIM> W1x_unpacked = vector<int32_t, HIDDEN_DIM>(round(W1x_facc));
  [unroll]
  for (int i = 0 ; i < HIDDEN_DIM; i+=4)
    W1x[i / 4] = pack_clamp_s8(int4(W1x_unpacked[i], W1x_unpacked[i + 1], W1x_unpacked[i + 2], W1x_unpacked[i + 3]));

  // Output layer
  typedef Matrix<ComponentType::I8, OUT_DIM_PADDED, HIDDEN_DIM, MatrixUse::A, MatrixScope::Thread> Wout_t;
  Wout_t Wout_coop = Wout_t::Load<MatrixLayout::RowMajor>(decoder, DEC_WOUT_OFFSET, HIDDEN_DIM);
  InterpretedVector<uint, HIDDEN_UINTS, ComponentType::I8> W1x_coop = MakeInterpretedVector<ComponentType::I8>(W1x);
  vector<int32_t, OUT_DIM_PADDED> Woutx_acc = MultiplyAdd<int32_t>(Wout_coop, W1x_coop, zero_out);
  vector<float, OUT_DIM_PADDED> Wout_scale_coop = decoder.Load< vector<float, OUT_DIM_PADDED> >(DEC_WOUT_SCALE_OFFSET);
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

void PerformInference(
  ByteAddressBuffer decoder,
  NTC NTCCBV,
  uint feat[FEAT_UINTS],
  out float o_Woutx[OUT_DIM_PADDED])
{
  uint W0x[HIDDEN_UINTS];
  uint W1x[HIDDEN_UINTS];

  // TODO: Templatize matmuls to make architecture experimentation easier

  // Layer 1
  [loop]
  for (int i = 0; i < HIDDEN_DIM; i += 4)
  {
    int4 acc = int4(0, 0, 0, 0);
    [unroll]
    for (int j = 0; j < FEAT_UINTS; j += 4)
    {
      uint4 W0vx = decoder.Load<uint4>(DEC_W0_OFFSET + (i + 0) * (FEATURE_DIM_PADDED) + (j * 4));
      uint4 W0vy = decoder.Load<uint4>(DEC_W0_OFFSET + (i + 1) * (FEATURE_DIM_PADDED) + (j * 4));
      uint4 W0vz = decoder.Load<uint4>(DEC_W0_OFFSET + (i + 2) * (FEATURE_DIM_PADDED) + (j * 4));
      uint4 W0vw = decoder.Load<uint4>(DEC_W0_OFFSET + (i + 3) * (FEATURE_DIM_PADDED) + (j * 4));
      [unroll]
      for (int c = 0; c < 4; c++)
      {
        acc.x = dot4add_i8packed(W0vx[c], feat[j + c], acc.x);
        acc.y = dot4add_i8packed(W0vy[c], feat[j + c], acc.y);
        acc.z = dot4add_i8packed(W0vz[c], feat[j + c], acc.z);
        acc.w = dot4add_i8packed(W0vw[c], feat[j + c], acc.w);
      }
    }
    float4 facc = float4(acc) * decoder.Load<float4>(DEC_W0_SCALE_OFFSET + i * 4);
    facc = hardgelu4(facc) * NTCCBV.rcp_s_a1_;
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
      uint4 W1vx = decoder.Load<uint4>(DEC_W1_OFFSET + (i + 0) * (HIDDEN_DIM) + (j * 4));
      uint4 W1vy = decoder.Load<uint4>(DEC_W1_OFFSET + (i + 1) * (HIDDEN_DIM) + (j * 4));
      uint4 W1vz = decoder.Load<uint4>(DEC_W1_OFFSET + (i + 2) * (HIDDEN_DIM) + (j * 4));
      uint4 W1vw = decoder.Load<uint4>(DEC_W1_OFFSET + (i + 3) * (HIDDEN_DIM) + (j * 4));
      [unroll]
      for (int c = 0; c < 4; c++)
      {
        acc.x = dot4add_i8packed(W1vx[c], W0x[j + c], acc.x);
        acc.y = dot4add_i8packed(W1vy[c], W0x[j + c], acc.y);
        acc.z = dot4add_i8packed(W1vz[c], W0x[j + c], acc.z);
        acc.w = dot4add_i8packed(W1vw[c], W0x[j + c], acc.w);
      }
    }
    float4 facc = float4(acc) * decoder.Load<float4>(DEC_W1_SCALE_OFFSET + i * 4);
    facc = hardgelu4(facc) * NTCCBV.rcp_s_a2_;
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
      uint4 Woutvx = decoder.Load<uint4>(DEC_WOUT_OFFSET + (i + 0) * (HIDDEN_DIM) + (j * 4));
      uint4 Woutvy = decoder.Load<uint4>(DEC_WOUT_OFFSET + (i + 1) * (HIDDEN_DIM) + (j * 4));
      uint4 Woutvz = decoder.Load<uint4>(DEC_WOUT_OFFSET + (i + 2) * (HIDDEN_DIM) + (j * 4));
      uint4 Woutvw = decoder.Load<uint4>(DEC_WOUT_OFFSET + (i + 3) * (HIDDEN_DIM) + (j * 4));
      [unroll]
      for (int c = 0; c < 4; c++)
      {
        acc.x = dot4add_i8packed(Woutvx[c], W1x[j + c], acc.x);
        acc.y = dot4add_i8packed(Woutvy[c], W1x[j + c], acc.y);
        acc.z = dot4add_i8packed(Woutvz[c], W1x[j + c], acc.z);
        acc.w = dot4add_i8packed(Woutvw[c], W1x[j + c], acc.w);
      }
    }
    float4 facc = float4(acc) * decoder.Load<float4>(DEC_WOUT_SCALE_OFFSET + i * 4);
    o_Woutx[i + 0] = facc.x;
    o_Woutx[i + 1] = facc.y;
    o_Woutx[i + 2] = facc.z;
    o_Woutx[i + 3] = facc.w;
  }
}
#endif

// Material parameters from NTC decoder output. Semantics that are not part of
//  the input manifest are given default values. present_ is a bitmask that has
//  ith bit set if sematic index i is part of the manifest.
struct MaterialParams
{
  float3 albedo_;
  float alpha_;
  float displacement_;
  float emissive_;
  float gloss_;
  float metallic_;
  float3 normal_;
  float ao_;
  float roughness_;
  float3 specular_;
  float transmission_;
  uint present_;
};

MaterialParams DefaultMaterialParams()
{
  MaterialParams mat;
  mat.albedo_ = 0.5.xxx;
  mat.alpha_ = 1.0;
  mat.displacement_ = 0.5;
  mat.emissive_ = 0.0;
  mat.gloss_ = 0.0;
  mat.metallic_ = 0.0;
  mat.normal_ = float3(0.5, 0.5, 1.0);
  mat.ao_ = 1.0;
  mat.roughness_ = 1.0;
  mat.specular_ = 0.5.xxx;
  mat.transmission_ = 0.0;
  mat.present_ = 0u;
  return mat;
}

// Distribute vector output of decoder to material struct params, based on channel-to-semantic mapping
#ifdef COOP_SUPPORT
MaterialParams DecodeMaterial(NTC NTCCBV, vector<float, OUT_DIM_PADDED> Woutx)
#else
MaterialParams DecodeMaterial(NTC NTCCBV, float Woutx[OUT_DIM_PADDED])
#endif
{
  MaterialParams mat = DefaultMaterialParams();

  Semantic prev_sem = Semantic::None;
  int comp = 0;
  [unroll]
  for (int i = 0; i < OUT_DIM; i++)
  {
    if (i >= NTCCBV.channel_count_)
      break;
    Semantic sem = ChannelSemantic(NTCCBV, i);
    comp = (sem == prev_sem) ? comp + 1 : 0;
    prev_sem = sem;
    mat.present_ |= SemanticBit(sem);
    float v = saturate(Woutx[i]);
    switch (sem)
    {
      case Semantic::Albedo:
        mat.albedo_[comp] = v;
        break;
      case Semantic::Alpha:
        mat.alpha_ = v;
        break;
      case Semantic::Displacement:
        mat.displacement_ = v;
        break;
      case Semantic::Emissive:
        mat.emissive_ = v;
        break;
      case Semantic::Gloss:
        mat.gloss_ = v;
        break;
      case Semantic::Metallic:
        mat.metallic_ = v;
        break;
      case Semantic::Normal:
        mat.normal_[comp] = v;
        break;
      case Semantic::AO:
        mat.ao_ = v;
        break;
      case Semantic::Roughness:
        mat.roughness_ = v;
        break;
      case Semantic::Specular:
        mat.specular_[comp] = v;
        break;
      case Semantic::Transmission:
        mat.transmission_ = v;
        break;
      default:
        break;
    }
  }
  return mat;
}

MaterialParams SampleMaterial(
  Buffer<uint> g0,
  Buffer<uint> g1,
  ByteAddressBuffer decoder,
  NTC NTCCBV,
  float2 uv,
  int lod)
{
#ifdef COOP_SUPPORT
  vector<uint, FEAT_UINTS> feat;
  vector<float, OUT_DIM_PADDED> Woutx;
#else
  uint feat[FEAT_UINTS];
  float Woutx[OUT_DIM_PADDED];
#endif
  GetFeaturesPacked(g0, g1, NTCCBV, uv, lod, feat);
  PerformInference(decoder, NTCCBV, feat, Woutx);
  return DecodeMaterial(NTCCBV, Woutx);
}


}  // namespace openntc

#endif  // __NTC_DECODE_HLSLI__
