#include "ntc_decode.hlsli"

struct LightingParams
{
  float exposure_;
  float displacement_scale_;
  float normal_scale_;
  float noise_frame_;
  float2 jitter_px_;
  float2 pad1_;
  float3 diffuse_sh_[9];
};


// Unjitter UVs from TAA to prevent texture blurring
//  Details: https://www.elopezr.com/temporal-aa-and-the-quest-for-the-holy-trail/

float2 UnjitterUv(float2 uv, float2 jitter_px)
{
  return uv + ddx(uv) * jitter_px.x + ddy(uv) * jitter_px.y;
}
float SpecularOcclusion(float NoV, float ao, float perceptual_roughness)
{
  return saturate(pow(NoV + ao, exp2(-16.0 * perceptual_roughness - 1.0)) - 1.0 + ao);
}

float3 IrradianceSh(LightingParams LightingParamsCBV, float3 n)
{
  return LightingParamsCBV.diffuse_sh_[0] +
          LightingParamsCBV.diffuse_sh_[1] * n.y +
          LightingParamsCBV.diffuse_sh_[2] * n.z +
          LightingParamsCBV.diffuse_sh_[3] * n.x +
          LightingParamsCBV.diffuse_sh_[4] * (n.y * n.x) +
          LightingParamsCBV.diffuse_sh_[5] * (n.y * n.z) +
          LightingParamsCBV.diffuse_sh_[6] * (3.0 * n.z * n.z - 1.0) +
          LightingParamsCBV.diffuse_sh_[7] * (n.z * n.x) +
          LightingParamsCBV.diffuse_sh_[8] * (n.x * n.x - n.y * n.y);
}

// ACES tonemapping code based on Stephen Hill's (@self_shadow) snippet in BakingLab

static const float3x3 ACESInput = { 0.59719, 0.35458, 0.04823, 0.07600, 0.90834, 0.01566, 0.02840, 0.13383, 0.83777 };
static const float3x3 ACESOutput =
  { 1.60475, -0.53108, -0.07367, -0.10208, 1.10813, -0.00605, -0.00327, -0.07276, 1.07602 };

float3 RRTAndODTFit(float3 v)
{
  return (v * (v + 0.0245786) - 0.000090537) / (v * (0.983729 * v + 0.4329510) + 0.238081);
}
