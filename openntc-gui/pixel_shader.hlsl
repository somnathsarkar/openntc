Texture2D<float4> tex_color : register(t1, space0);
SamplerState sampler_bilinear_clamp : register(s0);

struct PixelShaderInput
{
  float4 pos : SV_Position;
	float2 uv : TEXCOORD;
};

struct PixelShaderOutput
{
	float4 color : SV_TARGET;
};

PixelShaderOutput main(PixelShaderInput p_in)
{
  PixelShaderOutput p_out;

  p_out.color = tex_color.Sample(sampler_bilinear_clamp, p_in.uv);
  
  return p_out;
}