struct ModelViewProjection
{
  matrix mvp_;
};

ConstantBuffer<ModelViewProjection> ModelViewProjectionCB : register(b0, space0);

struct VertexShaderInput
{
  float3 pos_ : SV_Position;
  float3 normal_ : NORMAL;
  float3 tangent_ : TANGENT;
  float2 uv_ : TEXCOORD;
};

struct VertexShaderOutput
{
  float4 pos_ : SV_Position;
	float2 uv_ : TEXCOORD;
};

VertexShaderOutput vs_main(VertexShaderInput v_in)
{
  VertexShaderOutput v_out;

  v_out.pos_ = mul(ModelViewProjectionCB.mvp_, float4(v_in.pos_, 1.0f));
  v_out.uv_ = v_in.uv_;

  return v_out;
}

Texture2D<float4> tex_color : register(t0, space0);
SamplerState sampler_trilinear : register(s0);

struct PixelShaderInput
{
  float4 pos_ : SV_Position;
	float2 uv_ : TEXCOORD;
};

struct PixelShaderOutput
{
	float4 color_ : SV_TARGET;
};

PixelShaderOutput ps_main(PixelShaderInput p_in)
{
  PixelShaderOutput p_out;

  p_out.color_ = tex_color.Sample(sampler_trilinear, p_in.uv_);
  
  return p_out;
}