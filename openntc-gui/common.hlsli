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

// Helper function for PBR using an IBL light source
float4 ShadeMaterial(
  NTCMaterialParams mat,
  float4 pos_view,
  float4 normal_view,
  float4 tangent_view,
  matrix world_to_view,
  LightingParams lp,
  TextureCube<float4> tex_specular_ibl,
  Texture2D<float4> tex_dfg,
  SamplerState samp)
{
  float3 f0 = float3(0.04, 0.04, 0.04);
  float3 f90 = float3(1.0, 1.0, 1.0);

  // Alpha cutout
  clip(mat.alpha_ - 0.5);

  // Normal mapping
  float3 normal;
  if (mat.present_ & (1u << NTC_SEM_NORMAL))
  {
    float4 bitangent_view = float4(cross(normal_view.xyz, tangent_view.xyz), 0.0);
    matrix tbn_view = transpose(matrix(tangent_view, bitangent_view, normal_view, float4(0.0, 0.0, 0.0, 1.0)));
    float4 surface_normal = float4(mat.normal_ * 2.0 - 1.0, 0.0);
    float4 view_normal = mul(tbn_view, surface_normal);
    normal = normalize(float3(view_normal.xy * lp.normal_scale_, view_normal.z));
  }
  else
  {
    normal = normalize(normal_view.xyz);
  }

  // Apply one of gloss or roughness
  float perceptual_roughness = (!(mat.present_ & (1u << NTC_SEM_ROUGHNESS)) && (mat.present_ & (1u << NTC_SEM_GLOSS)))
    ? 1.0 - mat.gloss_
    : mat.roughness_;
  float3 albedo = pow(mat.albedo_, 2.2);

  // Specular
  if (mat.present_ & (1u << NTC_SEM_SPECULAR))
    f0 = pow(mat.specular_, 2.2);
  f0 = lerp(f0, albedo, mat.metallic_);
  albedo *= 1.0 - mat.metallic_;

  float3 view_dir = -normalize(pos_view.xyz / pos_view.w);
  float3 reflect_dir = reflect(-view_dir, normal);

  float NoV = abs(dot(normal, view_dir));

  float3 reflect_world = mul(float4(reflect_dir, 0.0), world_to_view).xyz;
  float3 normal_world = mul(float4(normal, 0.0), world_to_view).xyz;
  float lod_ibl = perceptual_roughness * 4.0;
  float3 specular_ibl = tex_specular_ibl.SampleLevel(samp, reflect_world, lod_ibl).rgb;
  float2 specular_dfg = tex_dfg.Sample(samp, float2(NoV, 1.0 - perceptual_roughness)).rg;
  float3 specular_color = f0 * specular_dfg.x + f90 * specular_dfg.y;
  float3 diffuse_ibl = max(IrradianceSh(lp, normal_world), 0.0);
  float so = SpecularOcclusion(NoV, mat.ao_, perceptual_roughness);

  float3 radiance = diffuse_ibl * albedo * mat.ao_ + specular_color * specular_ibl * so;
  // Emissive
  radiance += mat.emissive_ * pow(mat.albedo_, 2.2);

  // Tonemapping

  radiance *= lp.exposure_;
  radiance = saturate(mul(ACESOutput, RRTAndODTFit(mul(ACESInput, radiance))));

  return float4(pow(radiance, 1.0f / 2.2f), 1.0);
}
