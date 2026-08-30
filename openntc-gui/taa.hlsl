// Simplified TAA resolve that only accounts for camera movement

struct TaaInfo
{
  float4x4 proj_to_world_unjittered_;
  float4x4 prev_world_to_proj_unjittered_;
  float3 eye_;
  float pad0_;
  float2 pane_origin_;
  float2 pane_dim_;
};

ConstantBuffer<TaaInfo> TaaInfoCB : register(b0, space0);

struct VertexShaderOutput
{
  float4 pos_ : SV_Position0;
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

  v_out.pos_ = float4(g_map_vid_to_pos[vid], 1.0f, 1.0f);

  return v_out;
}

Texture2D<float4> tex_accum : register(t0, space0);
Texture2D<float4> tex_frame : register(t1, space0);
Texture2D<float> tex_depth : register(t2, space0);
Texture2D<float> tex_depth_prev : register(t3, space0);
SamplerState smp_history : register(s0, space0);

struct PixelShaderInput
{
  float4 pos_ : SV_Position0;
};

struct PixelShaderOutput
{
  float4 color_ : SV_TARGET;
};

PixelShaderOutput ps_main(PixelShaderInput p_in)
{
  PixelShaderOutput p_out;

  int2 texel = int2(p_in.pos_.xy);
  float4 cur = tex_frame.Load(int3(texel, 0));

  float pane_i = (p_in.pos_.x >= TaaInfoCB.pane_origin_.x + TaaInfoCB.pane_dim_.x) ? 1.0 : 0.0;
  float2 pane_base = TaaInfoCB.pane_origin_ + float2(pane_i * TaaInfoCB.pane_dim_.x, 0.0);
  float2 uv_local = (p_in.pos_.xy - pane_base) / TaaInfoCB.pane_dim_;

  float blend = 0.1;
  float2 prev_pixel = p_in.pos_.xy;
  if (all(uv_local >= 0.0) && all(uv_local <= 1.0))
  {
    float depth = tex_depth.Load(int3(texel, 0));
    float2 ndc = float2(uv_local.x * 2.0 - 1.0, 1.0 - 2.0 * uv_local.y);
    float4 world = mul(TaaInfoCB.proj_to_world_unjittered_, float4(ndc, depth, 1.0));
    world /= world.w;

    // Handle reprojection of cubemap separately

    float4 reproj_pos = (depth == 1.0) ? float4(world.xyz - TaaInfoCB.eye_, 0.0)
                                       : float4(world.xyz, 1.0);
    float4 prev_clip = mul(TaaInfoCB.prev_world_to_proj_unjittered_, reproj_pos);
    if (prev_clip.w > 0.0)
    {
      float2 prev_uv = float2(prev_clip.x / prev_clip.w + 1.0, 1.0 - prev_clip.y / prev_clip.w) / 2.0;
      if (all(prev_uv >= 0.0) && all(prev_uv <= 1.0))
        prev_pixel = pane_base + prev_uv * TaaInfoCB.pane_dim_;
      else
        blend = 1.0;
    }
    else
    {
      blend = 1.0;
    }
  }

  float2 full_dim;
  tex_accum.GetDimensions(full_dim.x, full_dim.y);
  float4 old = tex_accum.SampleLevel(smp_history, prev_pixel / full_dim, 0.0);

  // Color clamp history to current neighborhood
  
  float4 neighbor_min = cur;
  float4 neighbor_max = cur;
  [unroll]
  for (int dy = -1; dy <= 1; dy++)
  {
    [unroll]
    for (int dx = -1; dx <= 1; dx++)
    {
      int2 t = clamp(texel + int2(dx, dy), int2(0, 0), int2(full_dim) - 1);
      float4 c = tex_frame.Load(int3(t, 0));
      neighbor_min = min(neighbor_min, c);
      neighbor_max = max(neighbor_max, c);
    }
  }
  old = clamp(old, neighbor_min, neighbor_max);

  p_out.color_ = lerp(old, cur, blend);

  return p_out;
}