struct ModelViewProjection
{
  matrix mvp;
};

ConstantBuffer<ModelViewProjection> ModelViewProjectionCB : register(b0, space0);

struct VertexShaderInput
{
  float3 pos : SV_Position;
  float2 uv : TEXCOORD;
};

struct VertexShaderOutput
{
  float4 pos : SV_Position;
	float2 uv : TEXCOORD;
};

VertexShaderOutput main(VertexShaderInput v_in)
{
  VertexShaderOutput v_out;

  v_out.pos = mul(ModelViewProjectionCB.mvp, float4(v_in.pos, 1.0f));
  v_out.uv = v_in.uv;

  return v_out;
}