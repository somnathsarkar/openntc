#include <libopenntc/ntc_kernel.cuh>

#include <cassert>
#include <cstdio>

#include <cuda_fp16.h>

#define OUT_DIM 9
#define PI 3.14159265359

__global__ void forward_hardgelu(int n, float* input, float* o_output)
{
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= n) return;
  if (input[tid] < -1.5f) o_output[tid] = 0.0f;
  else if (input[tid] < 1.5f) o_output[tid] = (input[tid] / 3.0f) * (input[tid] + 1.5f);
  else o_output[tid] = input[tid];
}

__global__ void backward_hardgelu(int n, float* inputs, float* incoming_gradients, float* o_outgoing_gradients)
{
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= n) return;
  if (inputs[tid] < -1.5f) o_outgoing_gradients[tid] = 0.0f;
  else if(inputs[tid] < 1.5f) o_outgoing_gradients[tid] = (2.0f * inputs[tid] + 1.5f) * incoming_gradients[tid] / 3.0f;
  else o_outgoing_gradients[tid] = incoming_gradients[tid];
}

__global__ void scalar_product(int n, float a, float* inputs, float* o_outputs)
{
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= n) return;
  o_outputs[tid] = a * inputs[tid];
}

void launch_forward_hardgelu(int n, float* input, float* o_output)
{
  int num_blocks = (n + 1023) / 1024;
  forward_hardgelu<<<num_blocks, 1024>>>(n, input, o_output);
}

void launch_backward_hardgelu(int n, float* inputs, float* incoming_gradients, float* o_outgoing_gradients)
{
  int num_blocks = (n + 1023) / 1024;
  backward_hardgelu<<<num_blocks, 1024>>>(n, inputs, incoming_gradients, o_outgoing_gradients);
}

__device__ float hardgelu(float x)
{
  if (x < -1.5f) return 0.0f;
  if (x > 1.5f) return x;
  return (x / 3.0f) * (x + 1.5f);
}

__device__ float back_hardgelu(float x)
{
  if (x < -1.5f) return 0.0f;
  else if(x < 1.5f) return (2.0f * x + 1.5f) / 3.0f;
  return 1.0f;
}

void launch_scalar_product(int n, float a, float* inputs, float* o_outputs)
{
  int num_blocks = (n + 1023) / 1024;
  scalar_product<<<num_blocks, 1024>>>(n, a, inputs, o_outputs);
}

__device__ int clamp_int(int x, int a, int b)
{
  return (x > b) ? b : ((x < a) ? a : x);
}

__global__ void draw_features(
  int batch_dim,
  int grid_dim,
  int feature_dim,
  int mip_dim,
  int g0_dim,
  int g1_dim,
  int g0_channels,
  int g1_channels,
  float norm_lod,
  int* grid_draws,
  float* g0_noise,
  float* g1_noise,
  float* g0,
  float* g1,
  float* out_features)
{
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= batch_dim * grid_dim * grid_dim) return;
  int batch_i = tid / (grid_dim * grid_dim);
  int xy = tid % (grid_dim * grid_dim);
  int x = (xy % grid_dim) + grid_draws[batch_i * 2];
  int y = (xy / grid_dim) + grid_draws[batch_i * 2 + 1];
  float u = (x + 0.5f) / mip_dim;
  float v = (y + 0.5f) / mip_dim;
  float g0x = u * g0_dim - 0.5f;
  float g0y = v * g0_dim - 0.5f;
  float g1x = u * g1_dim - 0.5f;
  float g1y = v * g1_dim - 0.5f;
  // floorf, not (int) truncation: coords go negative in the first
  // half-grid-texel border band, where truncation would pick the wrong
  // corners (and disagree with the fractions below).
  int g0x0 = (int)floorf(g0x);
  int g0y0 = (int)floorf(g0y);
  int g1x0 = (int)floorf(g1x);
  int g1y0 = (int)floorf(g1y);
  float g1xf = g1x - floorf(g1x);
  float g1yf = g1y - floorf(g1y);
  float g1w[4] = {(1.0f - g1xf) * (1.0f - g1yf), g1xf * (1.0f - g1yf), (1.0f - g1xf) * g1yf, g1xf * g1yf};
  float g1feat[20];
  for (int i = 0; i < g1_channels; i++)
    g1feat[i] = 0.0f;
  for (int i0 = 0; i0 < 4; i0++)
  {
    int j = i0 % 2;
    int k = i0 / 2;
    int g0xc = clamp_int(g0x0 + j, 0, g0_dim - 1);
    int g0yc = clamp_int(g0y0 + k, 0, g0_dim - 1);
    for (int i1 = 0; i1 < g0_channels; i1++)
    {
      // row-major grid layout: (y * W + x) * C, matching the mip npy data
      out_features[(i0 * g0_channels + i1) * grid_dim * grid_dim * batch_dim + batch_i * grid_dim * grid_dim + xy]
        = g0[(g0yc * g0_dim + g0xc) * g0_channels + i1] + g0_noise[(g0yc * g0_dim + g0xc) * g0_channels + i1];
    }
    int g1xc = clamp_int(g1x0 + j, 0, g1_dim - 1);
    int g1yc = clamp_int(g1y0 + k, 0, g1_dim - 1);
    for (int i1 = 0; i1 < g1_channels; i1++)
    {
      float mult = g1[(g1yc * g1_dim + g1xc) * g1_channels + i1] + g1_noise[(g1yc * g1_dim + g1xc) * g1_channels + i1];
      g1feat[i1] += g1w[i0] * mult;
    }
  }
  for (int i1 = 0; i1 < g1_channels; i1++)
  {
    out_features[(4 * g0_channels + i1) * grid_dim * grid_dim * batch_dim + batch_i * grid_dim * grid_dim + xy]
     = g1feat[i1];
  }

  int periods[3] = {8, 4, 2};
  int pos_off = 0;
  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 2; j++)
    {
      for (int k = 0; k < 2; k++)
      {
        int P = periods[i];
        int c = (j == 0) ? x : y;
        float phase = (k == 0) ? 0.0f : (0.25f * P);
        float t = (c + 0.5f + phase) / P;
        float s = t - floorf(t);
        float o = 1 - 4.0f * fabsf(s - 0.5f);
        int idx = (4 * g0_channels + g1_channels + pos_off) * grid_dim * grid_dim * batch_dim +
                    batch_i * grid_dim * grid_dim + xy;
        out_features[idx] = o;
        pos_off++;
      }
    }
  }
  int idx = (4 * g0_channels + g1_channels + pos_off) * grid_dim * grid_dim * batch_dim +
              batch_i * grid_dim * grid_dim + xy;
  out_features[idx] = norm_lod;
  pos_off++;
  while ((4 * g0_channels + g1_channels + pos_off) < feature_dim)
  {
    int idx = (4 * g0_channels + g1_channels + pos_off) * grid_dim * grid_dim * batch_dim +
                batch_i * grid_dim * grid_dim + xy;
    out_features[idx] = 0.0f;
    pos_off++;
  }
}

void launch_draw_features(
  int batch_dim,
  int grid_dim,
  int feature_dim,
  int mip_dim,
  int g0_dim,
  int g1_dim,
  int g0_channels,
  int g1_channels,
  float norm_lod,
  int* grid_draws,
  float* g0_noise,
  float* g1_noise,
  float* g0,
  float* g1,
  float* out_features)
{
  int thread_count = batch_dim * grid_dim * grid_dim;
  int block_count = (thread_count + 1023) / 1024;
  draw_features<<<block_count, 1024>>>(
    batch_dim,
    grid_dim,
    feature_dim,
    mip_dim,
    g0_dim,
    g1_dim,
    g0_channels,
    g1_channels,
    norm_lod,
    grid_draws,
    g0_noise,
    g1_noise,
    g0,
    g1,
    out_features);
}

__global__ void draw_targets(
  int batch_dim,
  int grid_dim,
  int mip_dim,
  int pred_dim,
  int* grid_draws,
  uint8_t* mip,
  float* out_targets)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= batch_dim * grid_dim * grid_dim) return;
  int batch_i = tid / (grid_dim * grid_dim);
  int xy = tid % (grid_dim * grid_dim);
  int x = xy % grid_dim + grid_draws[batch_i + batch_i];
  int y = xy / grid_dim + grid_draws[batch_i + batch_i + 1];
  for (int i = 0; i < pred_dim; i++)
  {
    out_targets[i * grid_dim * grid_dim * batch_dim + batch_i * grid_dim * grid_dim + xy] =
      mip[(y * mip_dim * pred_dim) + x * pred_dim + i] / 255.0f;
  }
}

void launch_draw_targets(
  int batch_dim,
  int grid_dim,
  int mip_dim,
  int pred_dim,
  int* grid_draws,
  uint8_t* mip,
  float* out_targets)
{
  int thread_count = batch_dim * grid_dim * grid_dim;
  int block_count = (thread_count + 1023) / 1024;
  draw_targets<<<block_count, 1024>>>(
    batch_dim,
    grid_dim,
    mip_dim,
    pred_dim,
    grid_draws,
    mip,
    out_targets);
}

__device__ void mask_aggregate_atomic_increment(float* loc, int pos, float val)
{
  unsigned peers = __match_any_sync(__activemask(), pos);
  int lane = threadIdx.x % 32;
  int leader = __ffs(peers) - 1;
  
  float total_val = val;
  unsigned others = peers & (~(1u << lane));
  while (others)
  {
    int next = __ffs(others) - 1;
    total_val += __shfl_sync(peers, val, next);
    others &= (~(1 << next));
  }

  if (lane == leader)
    atomicAdd(&loc[pos], total_val);
}

__global__ void accumulate_grid_gradients(
  int batch_dim,
  int grid_dim,
  int feature_dim,
  int mip_dim,
  int g0_dim,
  int g1_dim,
  int g0_channels,
  int g1_channels,
  int* grid_draws,
  float* dLdx,
  float* o_dLdG0,
  float* o_dLdG1)
{
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= batch_dim * grid_dim * grid_dim) return;
  int batch_i = tid / (grid_dim * grid_dim);
  int xy = tid % (grid_dim * grid_dim);
  int x = (xy % grid_dim) + grid_draws[batch_i * 2];
  int y = (xy / grid_dim) + grid_draws[batch_i * 2 + 1];
  float u = (x + 0.5f) / mip_dim;
  float v = (y + 0.5f) / mip_dim;
  float g0x = u * g0_dim - 0.5f;
  float g0y = v * g0_dim - 0.5f;
  float g1x = u * g1_dim - 0.5f;
  float g1y = v * g1_dim - 0.5f;

  int g0x0 = (int)floorf(g0x);
  int g0y0 = (int)floorf(g0y);
  int g1x0 = (int)floorf(g1x);
  int g1y0 = (int)floorf(g1y);
  float g1xf = g1x - floorf(g1x);
  float g1yf = g1y - floorf(g1y);
  float g1w[4] = {(1.0f - g1xf) * (1.0f - g1yf), g1xf * (1.0f - g1yf), (1.0f - g1xf) * g1yf, g1xf * g1yf};
  
  for (int i0 = 0; i0 < 4; i0++)
  {
    int j = i0 % 2;
    int k = i0 / 2;
    int g0xc = clamp_int(g0x0 + j, 0, g0_dim - 1);
    int g0yc = clamp_int(g0y0 + k, 0, g0_dim - 1);
    
    for (int i1 = 0; i1 < g0_channels; i1++)
    {
      int pos = (g0yc * g0_dim + g0xc) * g0_channels + i1;
      float val = dLdx[(i0 * g0_channels + i1) * grid_dim * grid_dim * batch_dim + batch_i * grid_dim * grid_dim + xy];
      mask_aggregate_atomic_increment(o_dLdG0, pos, val);
      
    }
    int g1xc = clamp_int(g1x0 + j, 0, g1_dim - 1);
    int g1yc = clamp_int(g1y0 + k, 0, g1_dim - 1);
    for (int i1 = 0; i1 < g1_channels; i1++)
    {
      int pos = (g1yc * g1_dim + g1xc) * g1_channels + i1;
      float mult = dLdx[(4 * g0_channels + i1) * grid_dim * grid_dim * batch_dim + batch_i * grid_dim * grid_dim + xy];
      float val = g1w[i0] * mult;
      mask_aggregate_atomic_increment(o_dLdG1, pos, val);
    }
  }
}

void launch_accumulate_grid_gradients(
  int batch_dim,
  int grid_dim,
  int feature_dim,
  int mip_dim,
  int g0_dim,
  int g1_dim,
  int g0_channels,
  int g1_channels,
  int* grid_draws,
  float* dLdx,
  float* o_dLdG0,
  float* o_dLdG1)
{
  int thread_count = batch_dim * grid_dim * grid_dim;
  int block_count = (thread_count + 1023) / 1024;
  accumulate_grid_gradients<<<block_count, 1024>>>(
    batch_dim,
    grid_dim,
    feature_dim,
    mip_dim,
    g0_dim,
    g1_dim,
    g0_channels,
    g1_channels,
    grid_draws,
    dLdx,
    o_dLdG0,
    o_dLdG1
  );
}

__global__ void update_adam(
  int n,
  float lr,
  float beta_1,
  float beta_2,
  float bias_1,
  float bias_2,
  float* grad,
  float* m,
  float* v,
  float* params)
{
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= n) return;
  m[tid] = beta_1 * m[tid] + (1.0f - beta_1) * grad[tid];
  v[tid] = beta_2 * v[tid] + (1.0f - beta_2) * grad[tid] * grad[tid];
  float m_bias = m[tid] * bias_1;
  float v_bias = v[tid] * bias_2;
  params[tid] = params[tid] - lr * m_bias / (sqrtf(v_bias) + 1e-8f);
}

void launch_update_adam(
  int n,
  float lr,
  float beta_1,
  float beta_2,
  float bias_1,
  float bias_2,
  float* grad,
  float* m,
  float* v,
  float* params)
{
  int block_count = (n + 1023) / 1024;
  update_adam<<<block_count, 1024>>>(n, lr, beta_1, beta_2, bias_1, bias_2, grad, m, v, params);
}

__global__ void initialize_rand(
  int n,
  curandState* o_rstate)
{
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= n) return;
  curand_init(123, 0, tid, &o_rstate[tid]);
}

void launch_initialize_rand(int n, curandState* o_rstate)
{
  int block_count = (n + 1023) / 1024;
  initialize_rand<<<block_count, 1024>>>(n, o_rstate);
}

__global__ void generate_noise(
  int n,
  float delta,
  curandState* rstate,
  float* o_noise)
{
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= n) return;
  int stride = gridDim.x * blockDim.x;
  curandState rs = rstate[tid];
  for (int i = tid; i < n; i += stride)
    o_noise[i] = curand_uniform(&rs) * delta - (delta / 2.0f);
  rstate[tid] = rs;
}

void launch_generate_noise(int n, int rand_n, float delta, curandState* rstate, float* o_noise)
{
  int block_count = (rand_n + 1023) / 1024;
  generate_noise<<<block_count, 1024>>>(n, delta, rstate, o_noise);
}

__global__ void quantize_grid(
  int n,
  int num_bytes,
  float delta,
  float* g)
{
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= n) return;
  float k = rintf(g[tid] / delta);
  float kmin = -(float)(1 << (num_bytes - 1));
  float kmax = (float)(1 << (num_bytes - 1)) - 1.0f;
  g[tid] = fminf(fmaxf(k, kmin), kmax) * delta;
}

void launch_quantize_grid(int n, int num_bytes, float delta, float* g)
{
  int block_count = (n + 1023) / 1024;
  quantize_grid<<<block_count, 1024>>>(n, num_bytes, delta, g);
}

__global__ void clamp_grid(
  int n,
  float delta,
  float* g)
{
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= n) return;
  g[tid] = max(-1.0f, min(1.0f - delta, g[tid]));
}

void launch_clamp_grid(int n, int num_bytes, float delta, float* g)
{
  int block_count = (n + 1023) / 1024;
  clamp_grid<<<block_count, 1024>>>(n, delta, g);
}

// quantize float to 1/2/4/8 bits

__global__ void quantize_pack(int n_packs, int bits, float* input, uint32_t* output)
{
  int pack_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (pack_i >= n_packs) return;
  int vals_per_pack = 32 / bits;
  uint32_t packed = 0;
  for (int s = 0; s < vals_per_pack; s++)
  {
    int x = (int)roundf(input[pack_i * vals_per_pack + s] * (1 << (bits - 1)));
    x = clamp_int(x, -(1 << (bits - 1)), (1 << (bits - 1)) - 1);
    packed |= (uint32_t)(x + (1 << (bits - 1))) << (s * bits);
  }
  output[pack_i] = packed;
}

void launch_quantize_pack(int n, int n_packs, int bits, float* input, uint32_t* output)
{
  assert((n * bits) % 32 == 0);
  assert((n * bits) / 32 == n_packs);
  int block_count = (n_packs + 1023) / 1024;
  quantize_pack<<<block_count, 1024>>>(n_packs, bits, input, output);
}

// Simple max reduce on data, storing the answer for each block in a single global result.

__global__ void max_abs(int n, float* data, float* result)
{
  __shared__ float smax[1024];
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  smax[threadIdx.x] = (tid < n) ? fabsf(data[tid]) : 0.0f;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1)
  {
    if (threadIdx.x < s)
      smax[threadIdx.x] = fmaxf(smax[threadIdx.x], smax[threadIdx.x + s]);
    __syncthreads();
  }
  if (threadIdx.x == 0)
    atomicMax((int*)result, __float_as_int(smax[0]));
}

void launch_max_abs(int n, float* data, float* result)
{
  int block_count = (n + 1023) / 1024;
  max_abs<<<block_count, 1024>>>(n, data, result);
}

__global__ void prepare_tex(
  int w,
  int h,
  int c,
  int cmap_count,
  PrepareTexInput pt,
  uint8_t *tex,
  uint8_t *o_mip0)
{
  int tidx = blockDim.x * blockIdx.x + threadIdx.x;
  int tidy = blockDim.y * blockIdx.y + threadIdx.y;
  if (tidx >= w || tidy >= h) return;
  int cpre_src = tidy * w * c + tidx * c;
  int cpre_dst = tidy * w * cmap_count + tidx * cmap_count;
  for (int i = 0; i < cmap_count; i++)
  {
    o_mip0[cpre_dst + i] = tex[cpre_src + pt.cmap[i]];
  }
}

void launch_prepare_tex(int dim, int c, PrepareTexInput pt, uint8_t* tex, uint8_t* o_mip0)
{
  int cmap_count = 0;
  for (int i = 0; i < 4; i++)
  {
    if (pt.cmap[i] < 0 || pt.cmap[i] >= 4)
      break;
    cmap_count++;
  }
  assert(cmap_count > 0);
  unsigned int block_dim = (dim + 31) / 32;
  dim3 launch_dims = {block_dim, block_dim, 1u};
  dim3 block_size = {32u, 32u, 1u};
  prepare_tex<<<launch_dims, block_size>>>(dim, dim, c, cmap_count, pt, tex, o_mip0);
}

__device__ float sinc(float x)
{
  if (abs(x) < 1e-8)
    return 1.0f;
  return sin(x) / x;
}

__device__ float lanczos(float x, float a)
{
  return sinc(PI * x) * sinc(PI * x / a);
}

__global__ void filter_lanczos(int w_src, int h_src, int c, int a, bool is_y, void* mip_src, void* mip_dst)
{
  int tidx = blockDim.x * blockIdx.x + threadIdx.x;
  int tidy = blockDim.y * blockIdx.y + threadIdx.y;
  int2 tid = make_int2(tidx, tidy);
  int2 out_dim = is_y ? make_int2(w_src, h_src / 2) : make_int2(w_src / 2, h_src);
  if (tid.x >= out_dim.x || tid.y >= out_dim.y) return;

  int2 texel_center = is_y ? make_int2(tid.x, tid.y * 2 + 1) : make_int2(tid.x * 2 + 1, tid.y);
  int lim = is_y ? h_src : w_src;

  float total[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float total_weight = 0.0f;
  for (int i = -2 * a; i < 2 * a; i++)
  {
    int sample_loc = is_y ? texel_center.y + i : texel_center.x + i;
    if (sample_loc < 0 || sample_loc >= lim)
      continue;
    float weight = lanczos((0.5f + i) * 0.5f, a);
    for (int j = 0; j < c; j++)
    {
      int sx = is_y ? tid.x : sample_loc;
      int sy = is_y ? sample_loc : tid.y;
      float sampc = 0.0f;
      int src_idx = sy * w_src * c + sx * c + j;
      if (is_y)
      {
        // Source: float temp buffer
        sampc = ((float*)mip_src)[src_idx];
      }
      else
      {
        // Source: uint8 texture data
        sampc = (((uint8_t*)mip_src)[src_idx]) / 255.0f;
      }
      total[j] += sampc * weight;
    }
    total_weight += weight;
  }
  for (int j = 0; j < c; j++)
  {
    int dst_idx = tid.y * out_dim.x * c + tid.x * c + j;
    float dst_val = (total[j] / total_weight);
    if (is_y)
    {
      // Dest: uint8 texture data
      float dst_val_uint = min(255.0f, max(0.0f, dst_val * 255.0f));
      ((uint8_t*)mip_dst)[dst_idx] = (uint8_t)dst_val_uint;
    }
    else
    {
      // Dest: float temp buffer
      ((float*)mip_dst)[dst_idx] = (total[j] / total_weight);
    }
  }
}

void launch_filter_lanczos(int dim_src, int c, int a, uint8_t* mip_src, float* tmp, uint8_t* mip_dst)
{
  assert(dim_src % 2 == 0);
  unsigned int block_dim_small = ((dim_src / 2) + 31) / 32;
  unsigned int block_dim_big = (dim_src + 31) / 32;
  dim3 launch_dims_0 = {block_dim_small, block_dim_big, 1u};
  dim3 launch_dims_1 = {block_dim_small, block_dim_small, 1u};
  dim3 block_size = {32u, 32u, 1u};

  filter_lanczos<<<launch_dims_0, block_size>>>(
    dim_src,
    dim_src,
    c,
    a,
    false,
    (void*)mip_src,
    (void*)tmp);
  filter_lanczos<<<launch_dims_1, block_size>>>(
    dim_src / 2,
    dim_src,
    c,
    a,
    true,
    (void*)tmp,
    (void*)mip_dst);
}

__global__ void package_tex(int dim, int c, PackageTexInput pt, uint8_t* o_package)
{
  int tidx = blockDim.x * blockIdx.x + threadIdx.x;
  int tidy = blockDim.y * blockIdx.y + threadIdx.y;
  if (tidx >= dim || tidy >= dim) return;

  for (int i = 0; i < c; i++)
  {
    int sid = pt.map_feat_id_to_source_id[i];
    int cid = tidy * dim * pt.source_channels[sid] + tidx * pt.source_channels[sid] + pt.map_feat_id_to_channel_id[i];
    float samp = pt.tex[sid][cid];
    o_package[tidy * dim * c + tidx * c + i] = samp;
  }
}

void launch_package_tex(int dim, int c, PackageTexInput pt, uint8_t* o_package)
{
  unsigned int block_dim = (dim + 31) / 32;
  dim3 launch_dims = {block_dim, block_dim, 1u};
  dim3 block_size = {32u, 32u, 1u};
  package_tex<<<launch_dims, block_size>>>(dim, c, pt, o_package);
}