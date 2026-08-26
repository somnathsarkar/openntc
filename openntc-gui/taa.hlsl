struct VertexShaderOutput
{
  float4 pos : SV_Position0;
};

static const float2 g_map_vid_to_pos[6] = {
  float2(-1, -1),
  float2(-1, 1),
  float2(1, -1),
  float2(-1, 1),
  float2(1, 1),
  float2(1, -1)
};

VertexShaderOutput vs_main(uint vid : SV_VertexID)
{
  VertexShaderOutput v_out;

  v_out.pos = float4(g_map_vid_to_pos[vid], 1.0f, 1.0f);

  return v_out;
}

Texture2D<float4> tex_accum : register(t0, space0);
Texture2D<float4> tex_frame : register(t1, space0);

struct PixelShaderInput
{
  float4 pos : SV_Position0;
};

struct PixelShaderOutput
{
  float4 color : SV_TARGET;
};

PixelShaderOutput ps_main(PixelShaderInput p_in)
{
  PixelShaderOutput p_out;

  int2 texel = int2(p_in.pos.xy);
  float4 old = tex_accum.Load(int3(texel, 0));
  float4 cur = tex_frame.Load(int3(texel, 0));

  p_out.color = old * 0.9 + cur * 0.1;

  return p_out;
}