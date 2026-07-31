#include <cuda_runtime.h>
#include <curand_kernel.h>

void launch_forward_hardgelu(int n, float* input, float* o_output);
void launch_backward_hardgelu(int n, float* inputs, float* incoming_gradients, float* o_outgoing_gradients);
void launch_scalar_product(int n, float a, float* inputs, float* o_outputs);
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
  float* out_features);
void launch_draw_targets(int batch_dim, int grid_dim, int mip_dim, int pred_dim, int* grid_draws, float* mip, float* out_targets);
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
  float* o_dLdG1);
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
  float* params);
void launch_initialize_rand(int n, curandState* o_rstate);
void launch_generate_noise(int n, float delta, curandState* rstate, float* o_noise);
void launch_quantize_grid(int n, int num_bytes, float delta, float* g);
void launch_clamp_grid(int n, int num_bytes, float delta, float* g);
