struct ModelViewProjection
{
  matrix mvp;
};

ConstantBuffer<ModelViewProjection> ModelViewProjectionCB : register(b0, space0);

struct VertexShaderInput
{
  float3 pos : SV_Position;
  float3 color : COLOR;
};

struct VertexShaderOutput
{
  float4 pos : SV_Position;
	float4 color : COLOR;
};

VertexShaderOutput main(VertexShaderInput v_in)
{
  VertexShaderOutput v_out;

  v_out.pos = mul(ModelViewProjectionCB.mvp, float4(v_in.pos, 1.0f));
  v_out.color = float4(v_in.color, 1.0f);

  return v_out;
}