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
void launch_draw_targets(int batch_dim, int grid_dim, int mip_dim, int pred_dim, int* grid_draws, uint8_t* mip, float* out_targets);
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
void launch_forward_pass(
  int n,
  float* x,
  float* W0,
  float* W1,
  float* Wout,
  float* o_W0x,
  float* o_W0xa,
  float* o_W1x,
  float* o_W1xa,
  float* o_Woutx);
void launch_backward_pass(
  int n,
  float* W0,
  float* W1,
  float* Wout,
  float* dLdPred,
  float* W0x,
  float* W1x,
  float* o_dLdx,
  float* o_dLdW0x,
  float* o_dLdW1x);
void launch_quantize_pack(int n, int n_bytes, int bits, float* input, uint32_t* output);
void launch_max_abs(int n, float* data, float* result);

typedef struct
{
  int cmap[4];
} PrepareTexInput;
void launch_prepare_tex(int dim, int c, PrepareTexInput pt, uint8_t* tex, uint8_t* o_mip0);

void launch_filter_lanczos(int dim_src, int c, int a, uint8_t* mip_src, float* tmp, uint8_t* mip_dst);

typedef struct
{
  int source_channels[16];
  int map_feat_id_to_source_id[16];
  int map_feat_id_to_channel_id[16];
  uint8_t* tex[16];
} PackageTexInput;
void launch_package_tex(int dim, int c, PackageTexInput pt, uint8_t* o_package);
