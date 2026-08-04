struct PixelShaderInput
{
  float4 pos : SV_Position;
	float4 color : COLOR;
};

struct PixelShaderOutput
{
	float4 color : SV_TARGET;
};

PixelShaderOutput main(PixelShaderInput p_in)
{
  PixelShaderOutput p_out;

  p_out.color = p_in.color;
  
  return p_out;
}