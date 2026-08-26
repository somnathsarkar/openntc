// Simplified TAA resolve that only accounts for camera movement

struct TaaInfo
{
  float4x4 proj_to_world_unjittered;
  float4x4 prev_world_to_proj_unjittered;
  float3 eye;
  float pad0;
  float2 pane_origin;
  float2 pane_dim;
};

ConstantBuffer<TaaInfo> TaaInfoCB : register(b0, space0);

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
Texture2D<float> tex_depth : register(t2, space0);
Texture2D<float> tex_depth_prev : register(t3, space0);
SamplerState smp_history : register(s0, space0);

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
  float4 cur = tex_frame.Load(int3(texel, 0));

  float pane_i = (p_in.pos.x >= TaaInfoCB.pane_origin.x + TaaInfoCB.pane_dim.x) ? 1.0 : 0.0;
  float2 pane_base = TaaInfoCB.pane_origin + float2(pane_i * TaaInfoCB.pane_dim.x, 0.0);
  float2 uv_local = (p_in.pos.xy - pane_base) / TaaInfoCB.pane_dim;

  float blend = 0.1;
  float2 prev_pixel = p_in.pos.xy;
  if (all(uv_local >= 0.0) && all(uv_local <= 1.0))
  {
    float depth = tex_depth.Load(int3(texel, 0));
    float2 ndc = float2(uv_local.x * 2.0 - 1.0, 1.0 - 2.0 * uv_local.y);
    float4 world = mul(TaaInfoCB.proj_to_world_unjittered, float4(ndc, depth, 1.0));
    world /= world.w;

    // Handle reprojection of cubemap separately

    float4 reproj_pos = (depth == 1.0) ? float4(world.xyz - TaaInfoCB.eye, 0.0)
                                       : float4(world.xyz, 1.0);
    float4 prev_clip = mul(TaaInfoCB.prev_world_to_proj_unjittered, reproj_pos);
    if (prev_clip.w > 0.0)
    {
      float2 prev_uv = float2(prev_clip.x / prev_clip.w + 1.0, 1.0 - prev_clip.y / prev_clip.w) / 2.0;
      if (all(prev_uv >= 0.0) && all(prev_uv <= 1.0))
      {
        prev_pixel = pane_base + prev_uv * TaaInfoCB.pane_dim;

        // Depth rejection: Single object against depth = 1.0 cubemap

        float prev_depth = tex_depth_prev.Load(int3(int2(prev_pixel), 0));
        if (depth == 1.0)
        {
          if (prev_depth != 1.0)
            blend = 1.0;
        }
        else
        {
          float prev_depth_expected = prev_clip.z / prev_clip.w;
          if (prev_depth == 1.0 || abs(prev_depth_expected - prev_depth) > 1e-3)
            blend = 1.0;
        }
      }
      else
      {
        blend = 1.0;
      }
    }
    else
    {
      blend = 1.0;
    }
  }

  float2 full_dim;
  tex_accum.GetDimensions(full_dim.x, full_dim.y);
  float4 old = tex_accum.SampleLevel(smp_history, prev_pixel / full_dim, 0.0);

  p_out.color = lerp(old, cur, blend);

  return p_out;
}