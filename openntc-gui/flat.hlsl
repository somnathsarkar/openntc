struct ModelViewProjection
{
  matrix mvp;
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
  float4 pos : SV_Position;
	float2 uv : TEXCOORD;
};

VertexShaderOutput vs_main(VertexShaderInput v_in)
{
  VertexShaderOutput v_out;

  v_out.pos = mul(ModelViewProjectionCB.mvp, float4(v_in.pos, 1.0f));
  v_out.uv = v_in.uv;

  return v_out;
}

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

PixelShaderOutput ps_main(PixelShaderInput p_in)
{
  PixelShaderOutput p_out;

  p_out.color = tex_color.Sample(sampler_bilinear_clamp, p_in.uv);
  
  return p_out;
}