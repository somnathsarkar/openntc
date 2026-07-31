#include <openntc/ntc_kernel.cuh>
#include <cstdio>

// out_channels = 1 + 3 + 1 + 3 + 1 (AO + Color + Displacement + Normals + Roughness)

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
      g1feat[i1] += g1w[i0] * (g1[(g1yc * g1_dim + g1xc) * g1_channels + i1] + g1_noise[(g1yc * g1_dim + g1xc) * g1_channels + i1]);
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
        out_features[(4 * g0_channels + g1_channels + pos_off) * grid_dim * grid_dim * batch_dim + batch_i * grid_dim * grid_dim + xy]
         = o;
        pos_off++;
      }
    }
  }
  out_features[(4 * g0_channels + g1_channels + pos_off) * grid_dim * grid_dim * batch_dim + batch_i * grid_dim * grid_dim + xy] = norm_lod;
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
  draw_features<<<block_count, 1024>>>(batch_dim, grid_dim, feature_dim, mip_dim, g0_dim, g1_dim, g0_channels, g1_channels, norm_lod, grid_draws, g0_noise, g1_noise, g0, g1, out_features);
}

__global__ void draw_targets(int batch_dim, int grid_dim, int mip_dim, int pred_dim, int* grid_draws, float* mip, float* out_targets)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= batch_dim * grid_dim * grid_dim) return;
  int batch_i = tid / (grid_dim * grid_dim);
  int xy = tid % (grid_dim * grid_dim);
  int x = xy % grid_dim + grid_draws[batch_i + batch_i];
  int y = xy / grid_dim + grid_draws[batch_i + batch_i + 1];
  for (int i = 0; i < pred_dim; i++)
  {
    out_targets[i * grid_dim * grid_dim * batch_dim + batch_i * grid_dim * grid_dim + xy] = mip[(y * mip_dim * pred_dim) + x * pred_dim + i];
  }
}

void launch_draw_targets(int batch_dim, int grid_dim, int mip_dim, int pred_dim, int* grid_draws, float* mip, float* out_targets)
{
  int thread_count = batch_dim * grid_dim * grid_dim;
  int block_count = (thread_count + 1023) / 1024;
  draw_targets<<<block_count, 1024>>>(batch_dim, grid_dim, mip_dim, pred_dim, grid_draws, mip, out_targets);
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
  
  for (int i0 = 0; i0 < 4; i0++)
  {
    int j = i0 % 2;
    int k = i0 / 2;
    int g0xc = clamp_int(g0x0 + j, 0, g0_dim - 1);
    int g0yc = clamp_int(g0y0 + k, 0, g0_dim - 1);
    for (int i1 = 0; i1 < g0_channels; i1++)
    {
      // row-major grid layout: (y * W + x) * C, matching the mip npy data
      atomicAdd(&o_dLdG0[(g0yc * g0_dim + g0xc) * g0_channels + i1], dLdx[(i0 * g0_channels + i1) * grid_dim * grid_dim * batch_dim + batch_i * grid_dim * grid_dim + xy]);
    }
    int g1xc = clamp_int(g1x0 + j, 0, g1_dim - 1);
    int g1yc = clamp_int(g1y0 + k, 0, g1_dim - 1);
    for (int i1 = 0; i1 < g1_channels; i1++)
    {
      atomicAdd(&o_dLdG1[(g1yc * g1_dim + g1xc) * g1_channels + i1], g1w[i0] * dLdx[(4 * g0_channels + i1) * grid_dim * grid_dim * batch_dim + batch_i * grid_dim * grid_dim + xy]);
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
  o_noise[tid] = curand_uniform(&rstate[tid]) * delta - (delta / 2.0f);
}

void launch_generate_noise(int n, float delta, curandState* rstate, float* o_noise)
{
  int block_count = (n + 1023) / 1024;
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