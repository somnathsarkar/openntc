#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <ShObjIdl.h>

#include <wrl.h>
using namespace Microsoft::WRL;

#include <d3d12.h>
#include <dxgi1_6.h>
#include <d3dcompiler.h>
#include <DirectXMath.h>
using namespace DirectX;
#include <DirectXTex.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <thread>
#include <mutex>
#include <future>
#include <sstream>
#include <string>

#include <imgui.h>
#include <imgui_impl_win32.h>
#include <imgui_impl_dx12.h>

#include <openntc-gui/datapack.h>
#include <openntc-gui/thread.h>
#include <openntc-gui/model.h>
#include <openntc-gui/rhi.h>

extern "C" { __declspec(dllexport) extern const UINT D3D12SDKVersion = 721;}
extern "C" { __declspec(dllexport) extern const char* D3D12SDKPath = ".\\D3D12\\"; }

#if defined(_DEBUG)
#define VERIFY(hr) do { assert(!FAILED(hr)); } while(0)
#else
#define VERIFY(hr) do { (hr); } while(0)
#endif

const char* const g_map_semantic_to_name[static_cast<int32_t>(openntc::Semantic::Count)] = {
  "Albedo",
  "Alpha",
  "Displacement",
  "Emissive",
  "Gloss",
  "Metallic",
  "Normal",
  "AO",
  "Roughness",
  "Specular",
  "Transmission",
};

const char* const g_map_model_to_name[g_kModelCount] = {
  "Cube",
  "Sphere",
  "Plane",
  "Knob",
};

SharedContext g_ctx;
std::future<void> g_train_job;
std::atomic<bool> g_stop_training;

constexpr uint8_t g_numframes = 2;
uint32_t g_width = 1280;
uint32_t g_height = 720;
double g_total_seconds = 0;

HWND g_hwnd;
RECT g_window_rect;

ComPtr<ID3D12Device2> g_device;
ComPtr<ID3D12CommandQueue> g_queue;
ComPtr<IDXGISwapChain4> g_swapchain;
ComPtr<ID3D12Resource> g_buffers[g_numframes];
ComPtr<ID3D12Resource> g_depthbuffer;
ComPtr<ID3D12GraphicsCommandList10> g_commandlist;
ComPtr<ID3D12CommandAllocator> g_commandallocators[g_numframes];
ComPtr<ID3D12PipelineState> g_pipelinestate_flat;
ComPtr<ID3D12RootSignature> g_rootsignature_flat;
ComPtr<ID3D12PipelineState> g_pipelinestate_pbr;
ComPtr<ID3D12RootSignature> g_rootsignature_pbr;
constexpr int32_t g_kProfileCount = static_cast<int32_t>(openntc::Profile::Count);
const char* const g_map_profile_to_name[g_kProfileCount] = { "BPP 0.2", "BPP 0.5", "BPP 1.0", "BPP 2.25" };
constexpr int32_t g_kQualityCount = static_cast<int32_t>(openntc::Quality::Count);
const char* const g_map_quality_to_name[g_kQualityCount] = { "Low", "Medium", "High", "Ultra" };
openntc::Profile g_profile = openntc::Profile::Bpp_0_2;
std::string g_manifest_path;
DataPack g_datapack;
#define PACK_BLOB_ARGS(name) g_datapack.Get(name).data_, g_datapack.Get(name).size_
ComPtr<ID3D12PipelineState> g_pipelinestate_pbr_ntc[g_kProfileCount];
ComPtr<ID3D12PipelineState> g_pipelinestate_flat_ntc[g_kProfileCount];
ComPtr<ID3D12RootSignature> g_rootsignature_pbr_ntc;
ComPtr<ID3D12PipelineState> g_pipelinestate_pbr_ntc_coop[g_kProfileCount];
ComPtr<ID3D12PipelineState> g_pipelinestate_flat_ntc_coop[g_kProfileCount];
ComPtr<ID3D12RootSignature> g_rootsignature_pbr_ntc_coop;
ComPtr<ID3D12PipelineState> g_pipelinestate_cubemap;
ComPtr<ID3D12RootSignature> g_rootsignature_cubemap;
ComPtr<ID3D12PipelineState> g_pipelinestate_taa;
ComPtr<ID3D12RootSignature> g_rootsignature_taa;
ComPtr<ID3D12Resource> g_vertex_buffer[g_kModelCount];
ComPtr<ID3D12Resource> g_index_buffer[g_kModelCount];
ComPtr<ID3D12Resource> g_tex[openntc::kMaxSources];
ComPtr<ID3D12Resource> g_buffer_scratch;
ComPtr<ID3D12Resource> g_tex_specular_ibl;
ComPtr<ID3D12Resource> g_tex_specular_dfg;
XMFLOAT3A g_diffuse_sh[9] = {};
DXGI_FORMAT g_formats[openntc::kMaxSources];
UINT g_descriptorsize;
UINT g_frame_i;
bool g_initialized;
bool g_contentloaded = false;
bool g_compressed_dirty = false;
bool g_compressed_from_training = false;
bool g_compressed_matches_manifest = false;
std::atomic<SharedFields> g_shared_fields;
openntc::FileData g_fil_data;

DescriptorAllocator g_dalloc_rtv;
DescriptorHandle g_dhandle_rtv[g_numframes];

ComPtr<ID3D12Resource> g_tex_frame;
ComPtr<ID3D12Resource> g_tex_taa_accum[2];
DescriptorHandle g_dhandle_frame_rtv;
DescriptorHandle g_dhandle_frame_srv;
DescriptorHandle g_dhandle_taa_rtv[2];
DescriptorHandle g_dhandle_taa_srv[2];
DescriptorHandle g_dhandle_depth_srv;
ComPtr<ID3D12Resource> g_tex_depth_prev;
DescriptorHandle g_dhandle_depth_prev_srv;
D3D12_RESOURCE_STATES g_depth_prev_state = D3D12_RESOURCE_STATE_COMMON;
uint32_t g_taa_write_i = 0;
bool g_taa_history_valid = false;
XMMATRIX g_prev_world_to_proj;
D3D12_RESOURCE_STATES g_frame_state = D3D12_RESOURCE_STATE_COMMON;
D3D12_RESOURCE_STATES g_taa_accum_state[2] = { D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COMMON };
constexpr FLOAT g_clear_color[4] = { 0.4f, 0.6f, 0.9f, 1.0f };

DescriptorAllocator g_dalloc_dsv;
DescriptorHandle g_dhandle_dsv;

DescriptorAllocator g_dalloc_srv;
DescriptorHandle g_dhandle_tex[openntc::kMaxSources];
DescriptorHandle g_dhandle_pbr_tex[5];
DescriptorHandle g_dhandle_ntc_info;
DescriptorHandle g_dhandle_ntc_data[1 + 1 + 1];
DescriptorHandle g_dhandle_lparams;
DescriptorHandle g_dhandle_ibl[2];

ComPtr<ID3D12Resource> g_buffer_ntc_info;
ComPtr<ID3D12Resource> g_buffer_g0;
ComPtr<ID3D12Resource> g_buffer_g1;
ComPtr<ID3D12Resource> g_buffer_decoder;
ComPtr<ID3D12Resource> g_buffer_lighting_params;

ComPtr<ID3D12Fence> g_fence;
uint64_t g_fenceval = 0;
uint64_t g_framefenceval[g_numframes] = {};
HANDLE g_fence_event;

bool g_vsync = true;
bool g_gsync = false;
bool g_fullscreen = false;

struct ModelViewProjection
{
  XMMATRIX model_to_world_;
  XMMATRIX world_to_view_;
  XMMATRIX view_to_proj_;
};

struct CubemapTransforms
{
  XMMATRIX proj_to_view_;
  XMMATRIX view_to_world_;
};

struct TaaConstants
{
  XMMATRIX proj_to_world_unjittered_;
  XMMATRIX prev_world_to_proj_unjittered_;
  XMFLOAT3 eye_;
  float pad0_;
  XMFLOAT2 pane_origin_;
  XMFLOAT2 pane_dim_;
};

struct LightingParams
{
  float exposure_;
  float displacement_scale_;
  float normal_scale_;
  float noise_frame_;
  XMFLOAT2 jitter_px_;
  XMFLOAT2 pad1_;
  XMFLOAT3A diffuse_sh_[9];
};

D3D12_VERTEX_BUFFER_VIEW g_vbv[g_kModelCount];
D3D12_INDEX_BUFFER_VIEW g_ibv[g_kModelCount];

uint32_t g_map_model_to_index_count[g_kModelCount];

XMMATRIX g_model_mat, g_view_mat, g_proj_mat;
float g_pitch = 0.0f;
float g_yaw = 0.0f;
float y_roll = 0.0f;
float g_fov_y = 65.0f;

constexpr int32_t g_nonimgui_srv_count = openntc::kMaxSources + 5 + 1 + (1 + 1 + 1) + 3 + 5;
constexpr int32_t g_imgui_srv_count = 64;
constexpr int32_t g_srv_count = g_nonimgui_srv_count + g_imgui_srv_count;

enum class GuiState : int32_t
{
  Empty = 0,
  MaterialLoaded = 1,
  Training = 2,
  Compressed = 3,      // NTC data available, either trained or loaded from file
};

enum class GuiEvent : int32_t
{
  ManifestLoaded,
  TrainStarted,
  TrainFinished,
  TrainStopped,
  CompressedLoaded,
  Saved,
};

GuiState g_app_state = GuiState::Empty;
char g_status_text[512] = "Ready";

const char* GetNameForGuiState(GuiState s)
{
  switch (s)
  {
    case GuiState::Empty:
      return "No Material";
    case GuiState::MaterialLoaded:
      return "Material Loaded";
    case GuiState::Training:
      return "Training";
    case GuiState::Compressed:
      return "Compressed";
  }
  return "?";
}

void SetStatus(const char* fmt, ...)
{
  va_list args;
  va_start(args, fmt);
  vsnprintf(g_status_text, sizeof(g_status_text), fmt, args);
  va_end(args);
}

void TransitionGuiState(GuiEvent e)
{
  const GuiState s = g_app_state;
  switch (e)
  {
    case GuiEvent::ManifestLoaded:
      if (s != GuiState::Training)
        g_app_state = GuiState::MaterialLoaded;
      break;
    case GuiEvent::TrainStarted:
      if (s == GuiState::MaterialLoaded || s == GuiState::Compressed)
        g_app_state = GuiState::Training;
      break;
    case GuiEvent::TrainFinished:
      if (s == GuiState::Training)
        g_app_state = GuiState::Compressed;
      break;
    case GuiEvent::TrainStopped:
      if (s == GuiState::Training)
        g_app_state = GuiState::MaterialLoaded;
      break;
    case GuiEvent::CompressedLoaded:
      if (s != GuiState::Training)
        g_app_state = GuiState::Compressed;
      break;
    case GuiEvent::Saved:
      break;
  }
}

void PerformTrainingJob(openntc::Quality quality)
{
  openntc::TrainInfo train_info = {};
  train_info.grids_per_batch_ = 1;
  train_info.quality_ = quality;
  SharedFields fields = g_shared_fields.load(std::memory_order_seq_cst);
  fields.train_complete_ = false;
  fields.train_in_progress_ = true;
  g_shared_fields.store(fields, std::memory_order_seq_cst);
  {
    SharedContext::Access access = g_ctx.Acquire();
    access.ctx_.BeginTraining(train_info);
  }

  while(!g_stop_training.load(std::memory_order_seq_cst))
  { 
    SharedContext::Access access = g_ctx.Acquire();
    SharedFields fields = g_shared_fields.load(std::memory_order_seq_cst);
    openntc::TrainProgress tprogress = access.ctx_.Train(128);
    if (tprogress.phase_ == openntc::TrainPhase::TrainComplete)
    {
      openntc::EvalResults eval_results = access.ctx_.Eval();
      fields.eval_psnr_ = eval_results.psnr_;
      fields.eval_mse_ = eval_results.mse_;
      fields.train_in_progress_ = false;
      fields.train_complete_ = true;
      g_shared_fields.store(fields);
      break;
    }
    else
    {
      fields.train_steps_ = tprogress.batches_complete_;
      fields.train_total_steps_ = tprogress.total_batches_;
      g_shared_fields.store(fields);
    }
  }
}

std::vector<UINT> g_imgui_available_srv_slots;

enum class Shader: int32_t
{
  PBR,
  PBR_NTC,
  PBR_NTC_COOP,

  Count,
};

const char* g_map_shader_to_name[] = {
  "PBR",
  "PBR_NTC",
  "PBR_NTC_COOP"
};

enum class CameraMode: int32_t
{
  Static,
  Orbit,
  Controlled,

  Count,
};

const char* g_map_camera_mode_to_name[] = {
  "Static",
  "Orbit",
  "Controlled"
};

int32_t g_gui_shader_left = static_cast<int32_t>(Shader::PBR);
int32_t g_gui_shader_right = static_cast<int32_t>(Shader::PBR);

enum class FlatSource : int32_t
{
  Reference,
  NTC,
  NTC_COOP,

  Count
};
const char* const g_map_flat_source_to_name[] = {"Reference", "NTC", "NTC_COOP"};
bool g_gui_flat_view = false;
int32_t g_gui_flat_left = static_cast<int32_t>(FlatSource::Reference);
int32_t g_gui_flat_right = static_cast<int32_t>(FlatSource::NTC);
int32_t g_gui_flat_channel = 0;
int32_t g_tex_count = 0;
openntc::Semantic g_tex_semantics[openntc::kMaxSources] = {};

struct FlatParams
{
  float pane_dim_[2];
  int32_t semantic_;
  int32_t channels_;
  float pad2_[12];
};
int32_t g_gui_camera_mode = static_cast<int32_t>(CameraMode::Orbit);
float g_gui_displacement_scale = 0.01f;
float g_gui_normal_scale = 1.0f;
float g_gui_exposure = 1.0f;
bool g_gui_taa = true;
int32_t g_gui_model = 0;
int32_t g_gui_profile = 0;
int32_t g_gui_quality = static_cast<int32_t>(openntc::Quality::Medium);

static inline UINT64 RoundUpTo(UINT64 a, UINT64 b)
{
  return ((a + b - 1) / b) * b;
}

void ImguiDescriptorSrvAlloc(
  ImGui_ImplDX12_InitInfo* init_info,
  D3D12_CPU_DESCRIPTOR_HANDLE* cpu_handle,
  D3D12_GPU_DESCRIPTOR_HANDLE* gpu_handle)
{
  UINT slot = g_imgui_available_srv_slots.back();
  g_imgui_available_srv_slots.pop_back();
  DescriptorHandle dhandle = g_dalloc_srv.At(slot);
  *cpu_handle = dhandle.cpu_;
  *gpu_handle = dhandle.gpu_;
}

void ImguiDescriptorSrvFree(
  ImGui_ImplDX12_InitInfo* init_info,
  D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle,
  D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle)
{
  UINT slot = g_dalloc_srv.SlotForHandle(cpu_handle);
  g_imgui_available_srv_slots.push_back(slot);
}

uint64_t SignalFence(ComPtr<ID3D12CommandQueue> command_queue, ComPtr<ID3D12Fence> fence, uint64_t* fenceval)
{
  (*fenceval)++;
  VERIFY(command_queue->Signal(fence.Get(), *fenceval));
  return *fenceval;
}

void WaitForFenceValue(
  ComPtr<ID3D12Fence> fence,
  uint64_t fenceval,
  HANDLE fenceevent,
  std::chrono::milliseconds duration_ms = std::chrono::milliseconds::max())
{
  if (fence->GetCompletedValue() < fenceval)
  {
    VERIFY(fence->SetEventOnCompletion(fenceval, fenceevent));
    ::WaitForSingleObject(fenceevent, static_cast<DWORD>(duration_ms.count()));
  }
}

void TransitionResource(
  ComPtr<ID3D12GraphicsCommandList10> command_list,
  ComPtr<ID3D12Resource> res,
  D3D12_RESOURCE_STATES initial_state,
  D3D12_RESOURCE_STATES final_state)
{
  D3D12_RESOURCE_BARRIER barrier = {};
  barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
  barrier.Transition.pResource = res.Get();
  barrier.Transition.Subresource = 0;
  barrier.Transition.StateBefore = initial_state;
  barrier.Transition.StateAfter = final_state;
  command_list->ResourceBarrier(1, &barrier);
}

void TransitionIfRequired(
  ComPtr<ID3D12GraphicsCommandList10> command_list,
  ComPtr<ID3D12Resource> res,
  D3D12_RESOURCE_STATES& state,
  D3D12_RESOURCE_STATES final_state)
{
  if (state == final_state)
    return;
  TransitionResource(command_list, res, state, final_state);
  state = final_state;
}

void CreateTaaResources()
{
  g_tex_frame = CreateTexture2D(
    g_device.Get(),
    D3D12_HEAP_TYPE_DEFAULT,
    g_width,
    g_height,
    1,
    1,
    DXGI_FORMAT_R8G8B8A8_UNORM,
    D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
    g_clear_color);
  g_device->CreateRenderTargetView(g_tex_frame.Get(), nullptr, g_dhandle_frame_rtv.cpu_);
  g_device->CreateShaderResourceView(g_tex_frame.Get(), nullptr, g_dhandle_frame_srv.cpu_);
  g_frame_state = D3D12_RESOURCE_STATE_COMMON;

  for (int i = 0; i < 2; i++)
  {
    g_tex_taa_accum[i] = CreateTexture2D(
      g_device.Get(),
      D3D12_HEAP_TYPE_DEFAULT,
      g_width,
      g_height,
      1,
      1,
      DXGI_FORMAT_R8G8B8A8_UNORM,
      D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
    g_device->CreateRenderTargetView(g_tex_taa_accum[i].Get(), nullptr, g_dhandle_taa_rtv[i].cpu_);
    g_device->CreateShaderResourceView(g_tex_taa_accum[i].Get(), nullptr, g_dhandle_taa_srv[i].cpu_);
    g_taa_accum_state[i] = D3D12_RESOURCE_STATE_COMMON;
  }

  g_tex_depth_prev = CreateTexture2D(
    g_device.Get(),
    D3D12_HEAP_TYPE_DEFAULT,
    g_width,
    g_height,
    1,
    1,
    DXGI_FORMAT_R32_FLOAT);
  g_device->CreateShaderResourceView(g_tex_depth_prev.Get(), nullptr, g_dhandle_depth_prev_srv.cpu_);
  g_depth_prev_state = D3D12_RESOURCE_STATE_COMMON;

  g_taa_history_valid = false;
}

void CreateTextureBarrier(
  ComPtr<ID3D12GraphicsCommandList10> command_list,
  ComPtr<ID3D12Resource> res_tex,
  UINT mip,
  D3D12_RESOURCE_STATES initial_state,
  D3D12_RESOURCE_STATES final_state)
{
  D3D12_RESOURCE_BARRIER barrier = {};
  barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
  barrier.Transition.pResource = res_tex.Get();
  barrier.Transition.Subresource = mip;
  barrier.Transition.StateBefore = initial_state;
  barrier.Transition.StateAfter = final_state;
  command_list->ResourceBarrier(1, &barrier);
}

void Flush(ComPtr<ID3D12CommandQueue> command_queue, ComPtr<ID3D12Fence> fence, uint64_t* fenceval, HANDLE fenceevent)
{
  uint64_t signalval = SignalFence(command_queue, fence, fenceval);
  WaitForFenceValue(fence, signalval, fenceevent);
}

inline int CeilDiv(int a, int b)
{
  return (a + b - 1) / b;
}

void ResizeDepthBuffer(uint32_t width, uint32_t height)
{
  if (!g_contentloaded)
    return;

  g_width = std::max(1u, width);
  g_height = std::max(1u, height);

  Flush(g_queue, g_fence, &g_fenceval, g_fence_event);

  D3D12_HEAP_PROPERTIES heap_props = {};
  heap_props.Type = D3D12_HEAP_TYPE_DEFAULT;
  heap_props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
  heap_props.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
  heap_props.VisibleNodeMask = 1;
  heap_props.CreationNodeMask = 1;

  D3D12_RESOURCE_DESC res_desc = {};
  res_desc.Format = DXGI_FORMAT_D32_FLOAT;
  res_desc.Width = g_width;
  res_desc.Height = g_height;
  res_desc.DepthOrArraySize = 1;
  res_desc.MipLevels = 1;
  res_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
  res_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
  res_desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
  res_desc.SampleDesc.Count = 1;
  res_desc.SampleDesc.Quality = 0;

  D3D12_CLEAR_VALUE optimized_clear_val = {};
  optimized_clear_val.Format = DXGI_FORMAT_D32_FLOAT;
  optimized_clear_val.DepthStencil.Depth = 1.0f;
  optimized_clear_val.DepthStencil.Stencil = 0;

  VERIFY(g_device->CreateCommittedResource(
    &heap_props,
    D3D12_HEAP_FLAG_NONE,
    &res_desc,
    D3D12_RESOURCE_STATE_DEPTH_WRITE,
    &optimized_clear_val,
    IID_PPV_ARGS(&g_depthbuffer)));

  D3D12_DEPTH_STENCIL_VIEW_DESC dsv_desc = {};
  dsv_desc.Format = DXGI_FORMAT_D32_FLOAT;
  dsv_desc.Flags = D3D12_DSV_FLAG_NONE;
  dsv_desc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
  dsv_desc.Texture2D.MipSlice = 0;

  g_device->CreateDepthStencilView(g_depthbuffer.Get(), &dsv_desc, g_dhandle_dsv.cpu_);

  D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
  srv_desc.Format = DXGI_FORMAT_R32_FLOAT;
  srv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
  srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
  srv_desc.Texture2D.MipLevels = 1;
  g_device->CreateShaderResourceView(g_depthbuffer.Get(), &srv_desc, g_dhandle_depth_srv.cpu_);
}

static void RebuildTextureResources(SharedContext::Access& access)
{
  Flush(g_queue, g_fence, &g_fenceval, g_fence_event);
  openntc::TextureData tex_data = access.ctx_.GetTextureData();
  g_tex_count = tex_data.tex_count_;
  for (int tex_i = 0; tex_i < tex_data.tex_count_; tex_i++)
    g_tex_semantics[tex_i] = tex_data.semantics_[tex_i];
  g_gui_flat_channel = std::clamp(g_gui_flat_channel, 0, std::max(g_tex_count - 1, 0));

  for (int tex_i = 0; tex_i < tex_data.tex_count_; tex_i++)
  {
    g_formats[tex_i] = (tex_data.channels_[tex_i] == 1) ?
                        DXGI_FORMAT_R8_UNORM :
                        DXGI_FORMAT_R8G8B8A8_UNORM;

    g_tex[tex_i] = CreateTexture2D(
      g_device.Get(),
      D3D12_HEAP_TYPE_GPU_UPLOAD,
      access.ctx_.GetMipDim(0),
      access.ctx_.GetMipDim(0),
      1,
      tex_data.mip_count_,
      g_formats[tex_i]);

    D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
    srv_desc.Format = g_formats[tex_i];
    srv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv_desc.Texture2D.MipLevels = tex_data.mip_count_;
    srv_desc.Texture2D.MostDetailedMip = 0;
    srv_desc.Texture2D.PlaneSlice = 0;
    srv_desc.Texture2D.ResourceMinLODClamp = 0.0f;
    srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;

    g_device->CreateShaderResourceView(g_tex[tex_i].Get(), &srv_desc, g_dhandle_tex[tex_i].cpu_);
  }

  {
    const openntc::Semantic pbr_order[5] = {
      openntc::Semantic::AO,
      openntc::Semantic::Albedo,
      openntc::Semantic::Displacement,
      openntc::Semantic::Normal,
      openntc::Semantic::Roughness};
    for (int slot = 0; slot < 5; slot++)
    {
      int src = 0;
      for (int tex_i = 0; tex_i < tex_data.tex_count_; tex_i++)
        if (tex_data.semantics_[tex_i] == pbr_order[slot])
          src = tex_i;
      g_device->CopyDescriptorsSimple(
        1,
        g_dhandle_pbr_tex[slot].cpu_,
        g_dhandle_tex[src].cpu_,
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    }
  }

  UINT64 scratch_size = 0llu;
  for (int mip_i = 0; mip_i < tex_data.mip_count_; mip_i++)
  {
    int mip_dim = access.ctx_.GetMipDim(mip_i);
    UINT64 mip_row = RoundUpTo(sizeof(uint8_t) * mip_dim * 4, D3D12_TEXTURE_DATA_PITCH_ALIGNMENT);
    scratch_size += mip_row * mip_dim;
  }

  g_buffer_scratch = CreateBuffer(g_device.Get(), D3D12_HEAP_TYPE_GPU_UPLOAD, scratch_size);

  for (int tex_i = 0; tex_i < tex_data.tex_count_; tex_i++)
  {
    g_commandlist->Reset(g_commandallocators[g_frame_i].Get(), nullptr);
    UINT64 mip_off = 0llu;

    void* mapped = nullptr;
    D3D12_RANGE read_range = {0, 0};
    g_buffer_scratch->Map(0, &read_range, &mapped);
    for (int mip_i = 0; mip_i < tex_data.mip_count_; mip_i++)
    {
      UINT64 mip_dim = access.ctx_.GetMipDim(mip_i);
      UINT64 channels_padded = (tex_data.channels_[tex_i] == 1) ? 1 : 4; // Pad 3 channels to 4
      UINT64 mip_row = sizeof(uint8_t) * mip_dim * channels_padded;
      UINT64 mip_row_padded = RoundUpTo(mip_row, D3D12_TEXTURE_DATA_PITCH_ALIGNMENT);
      UINT64 mip_size = mip_row_padded * mip_dim;
      if (channels_padded == 1)
      {
        for (int y = 0; y < mip_dim; y++)
          memcpy(
            static_cast<char*>(mapped) + y * mip_row_padded + mip_off,
            tex_data.mips_[tex_i][mip_i] + y * mip_dim * tex_data.channels_[tex_i],
            mip_row);
      }
      else
      {
        for (int y = 0; y < mip_dim; y++)
        {
          for (int x = 0; x < mip_dim; x++)
          {
            memcpy(
              static_cast<char*>(mapped) + y * mip_row_padded + mip_off + x * channels_padded,
              tex_data.mips_[tex_i][mip_i] + y * mip_dim * tex_data.channels_[tex_i] + x * tex_data.channels_[tex_i],
              3 * sizeof(uint8_t));
            static_cast<char*>(mapped)[y * mip_row_padded + mip_off + x * channels_padded + 3] = 255;
          }
        }
      }

      D3D12_TEXTURE_COPY_LOCATION src_tex_loc = {};
      src_tex_loc.pResource = g_buffer_scratch.Get();
      src_tex_loc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
      src_tex_loc.PlacedFootprint.Footprint.Width = mip_dim;
      src_tex_loc.PlacedFootprint.Footprint.Height = mip_dim;
      src_tex_loc.PlacedFootprint.Footprint.Depth = 1;
      src_tex_loc.PlacedFootprint.Footprint.Format = g_formats[tex_i];
      src_tex_loc.PlacedFootprint.Footprint.RowPitch = mip_row_padded;
      src_tex_loc.PlacedFootprint.Offset = mip_off;

      D3D12_TEXTURE_COPY_LOCATION dst_tex_loc = {};
      dst_tex_loc.pResource = g_tex[tex_i].Get();
      dst_tex_loc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
      dst_tex_loc.SubresourceIndex = mip_i;
      g_commandlist->CopyTextureRegion(&dst_tex_loc, 0, 0, 0, &src_tex_loc, nullptr);

      mip_off += mip_size;
    }

    D3D12_RESOURCE_BARRIER rbar = {};
    rbar.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    rbar.Transition.pResource = g_tex[tex_i].Get();
    rbar.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    rbar.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    rbar.Transition.StateAfter =
      D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    rbar.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    g_commandlist->ResourceBarrier(1, &rbar);

    g_buffer_scratch->Unmap(0, nullptr);
    g_commandlist->Close();
    ID3D12CommandList* lists[] = { g_commandlist.Get() };
    g_queue->ExecuteCommandLists(1, lists);
    Flush(g_queue, g_fence, &g_fenceval, g_fence_event);
  }
}

static void LoadDiffuseSH(const void* data, size_t size)
{
  std::istringstream stream(std::string(static_cast<const char*>(data), size));
  int count = 0;
  std::string line;
  while (count < 9 && std::getline(stream, line))
  {
    float x, y, z;
    if (sscanf_s(line.c_str(), " ( %f , %f , %f", &x, &y, &z) == 3)
    {
      g_diffuse_sh[count] = XMFLOAT3A(x, y, z);
      count++;
    }
  }
  VERIFY(count == 9);
}

static void LoadIBL()
{
  LoadDiffuseSH(PACK_BLOB_ARGS("sh.txt"));
  {
    ScratchImage img;
    LoadFromDDSMemory(
      static_cast<const uint8_t*>(g_datapack.Get("specular_cube.dds").data_),
      g_datapack.Get("specular_cube.dds").size_,
      DDS_FLAGS_NONE,
      nullptr,
      img);

    g_tex_specular_ibl = CreateTexture2D(
      g_device.Get(),
      D3D12_HEAP_TYPE_GPU_UPLOAD,
      256,
      256,
      6,
      5,
      DXGI_FORMAT_R16G16B16A16_FLOAT);

    D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
    srv_desc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    srv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
    srv_desc.TextureCube.MipLevels = 5;
    srv_desc.TextureCube.MostDetailedMip = 0;
    srv_desc.TextureCube.ResourceMinLODClamp = 0.0f;
    srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;

    g_device->CreateShaderResourceView(g_tex_specular_ibl.Get(), &srv_desc, g_dhandle_ibl[0].cpu_);

    for (int face_i = 0; face_i < 6; face_i++)
    {
      for (int mip_i = 0; mip_i < 5; mip_i++)
      {
        const Image* subimg = img.GetImage(mip_i, face_i, 0);
        VERIFY(g_tex_specular_ibl->WriteToSubresource(
          face_i * 5 + mip_i,
          nullptr,
          subimg->pixels,
          (UINT)subimg->rowPitch,
          (UINT)subimg->slicePitch));
      }
    }
  }

  {
    ScratchImage img;
    LoadFromDDSMemory(
      static_cast<const uint8_t*>(g_datapack.Get("dfg.dds").data_),
      g_datapack.Get("dfg.dds").size_,
      DDS_FLAGS_NONE,
      nullptr,
      img);

    g_tex_specular_dfg = CreateTexture2D(
      g_device.Get(),
      D3D12_HEAP_TYPE_GPU_UPLOAD,
      256,
      256,
      1,
      1,
      DXGI_FORMAT_R16G16B16A16_FLOAT);

    D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
    srv_desc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    srv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv_desc.Texture2D.MipLevels = 1;
    srv_desc.Texture2D.MostDetailedMip = 0;
    srv_desc.Texture2D.ResourceMinLODClamp = 0.0f;
    srv_desc.Texture2D.PlaneSlice = 0;
    srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;

    g_device->CreateShaderResourceView(g_tex_specular_dfg.Get(), &srv_desc, g_dhandle_ibl[1].cpu_);

    const Image* pimg = img.GetImage(0, 0, 0);
    VERIFY(g_tex_specular_dfg->WriteToSubresource(0, nullptr, pimg->pixels, pimg->rowPitch, pimg->slicePitch));
  }

  {
    uint64_t cbv_size = RoundUpTo(sizeof(LightingParams), 256);

    g_buffer_lighting_params = CreateBuffer(g_device.Get(), D3D12_HEAP_TYPE_GPU_UPLOAD, cbv_size);

    D3D12_CONSTANT_BUFFER_VIEW_DESC cbv_desc = {};
    cbv_desc.BufferLocation = g_buffer_lighting_params->GetGPUVirtualAddress();
    cbv_desc.SizeInBytes = cbv_size;

    g_device->CreateConstantBufferView(&cbv_desc, g_dhandle_lparams.cpu_);
  }
}

void LoadContent()
{

  {
    D3D12_FEATURE_DATA_D3D12_OPTIONS16 options16 = {};
    VERIFY(g_device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS16, &options16, sizeof(options16)));
    assert(options16.GPUUploadHeapSupported);
  }

  for (int i = 0; i < g_kModelCount; i++)
  {
    Model model;
    InitModel(static_cast<ModelType>(i), 200, model, PACK_BLOB_ARGS("knob.bin"));
    g_map_model_to_index_count[i] = model.GetIndexCount();
    g_vertex_buffer[i] = CreateBufferWithData(
      g_device.Get(),
      D3D12_HEAP_TYPE_GPU_UPLOAD,
      sizeof(VertexDescriptor) * model.GetVertexCount(),
      sizeof(VertexDescriptor) * model.GetVertexCount(),
      model.GetVertices());
    g_index_buffer[i] = CreateBufferWithData(
      g_device.Get(),
      D3D12_HEAP_TYPE_GPU_UPLOAD,
      sizeof(uint32_t) * model.GetIndexCount(),
      sizeof(uint32_t) * model.GetIndexCount(),
      model.GetIndices());
    g_vbv[i].BufferLocation = g_vertex_buffer[i]->GetGPUVirtualAddress();
    g_vbv[i].SizeInBytes = sizeof(VertexDescriptor) * model.GetVertexCount();
    g_vbv[i].StrideInBytes = sizeof(VertexDescriptor);
    g_ibv[i].BufferLocation = g_index_buffer[i]->GetGPUVirtualAddress();
    g_ibv[i].SizeInBytes = sizeof(uint32_t) * model.GetIndexCount();
    g_ibv[i].Format = DXGI_FORMAT_R32_UINT;
  }

  // Flat

  {
    D3D12_INPUT_ELEMENT_DESC input_layout[] = {
      {"SV_Position", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT,
        D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
      {"NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT,
        D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
      {"TANGENT", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT,
        D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
      {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT,
        D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
    };

    D3D12_FEATURE_DATA_ROOT_SIGNATURE feature_data = {};
    feature_data.HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_1;
    if (FAILED(g_device->CheckFeatureSupport(D3D12_FEATURE_ROOT_SIGNATURE, &feature_data, sizeof(feature_data))))
    {
      feature_data.HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_0;
    }

    RootSignatureBuilder rsb;
    rsb.RootConstants(sizeof(FlatParams) / 4, D3D12_SHADER_VISIBILITY_ALL)
        .Range(
          1,
          D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
          D3D12_DESCRIPTOR_RANGE_FLAG_DATA_STATIC,
          D3D12_SHADER_VISIBILITY_PIXEL)
        .StaticSampler(D3D12_FILTER_MIN_MAG_MIP_LINEAR, D3D12_TEXTURE_ADDRESS_MODE_WRAP, D3D12_SHADER_VISIBILITY_PIXEL);
    g_rootsignature_flat = rsb.Build(g_device.Get());
  }

  // PBR

  {
    D3D12_INPUT_ELEMENT_DESC input_layout[] = {
      {"SV_Position", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT,
        D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
      {"NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT,
        D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
      {"TANGENT", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT,
        D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
      {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT,
        D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
    };

    D3D12_FEATURE_DATA_ROOT_SIGNATURE feature_data = {};
    feature_data.HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_1;
    if (FAILED(g_device->CheckFeatureSupport(D3D12_FEATURE_ROOT_SIGNATURE, &feature_data, sizeof(feature_data))))
    {
      feature_data.HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_0;
    }

    D3D12_ROOT_SIGNATURE_FLAGS root_signature_flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT |
                                                      D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS |
                                                      D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS |
                                                      D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS;

    RootSignatureBuilder rsb;
    rsb.RootConstants(sizeof(ModelViewProjection) / 4, D3D12_SHADER_VISIBILITY_ALL)
        .Range(5, D3D12_DESCRIPTOR_RANGE_TYPE_SRV, D3D12_DESCRIPTOR_RANGE_FLAG_DATA_STATIC, D3D12_SHADER_VISIBILITY_ALL)
        .Range(
          2,
          D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
          D3D12_DESCRIPTOR_RANGE_FLAG_DATA_STATIC,
          D3D12_SHADER_VISIBILITY_PIXEL)
        .Range(
          1,
          D3D12_DESCRIPTOR_RANGE_TYPE_CBV,
          D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE,
          D3D12_SHADER_VISIBILITY_ALL)
        .StaticSampler(D3D12_FILTER_ANISOTROPIC, D3D12_TEXTURE_ADDRESS_MODE_WRAP, D3D12_SHADER_VISIBILITY_ALL);
    g_rootsignature_pbr = rsb.Build(g_device.Get());
    
    GraphicsPipelineBuilder gpb;
    gpb.RootSignature(g_rootsignature_pbr.Get())
        .Input(input_layout, _countof(input_layout))
        .VS(PACK_BLOB_ARGS("pbr_vs"))
        .PS(PACK_BLOB_ARGS("pbr_ps"))
        .DepthEnable(true)
        .CullMode(D3D12_CULL_MODE_BACK);
    g_pipelinestate_pbr = gpb.Build(g_device.Get());
  }

  // PBR NTC

  {
    D3D12_INPUT_ELEMENT_DESC input_layout[] = {
      {"SV_Position", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT,
        D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
      {"NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT,
        D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
      {"TANGENT", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT,
        D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
      {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT,
        D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
    };

    D3D12_FEATURE_DATA_ROOT_SIGNATURE feature_data = {};
    feature_data.HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_1;
    if (FAILED(g_device->CheckFeatureSupport(D3D12_FEATURE_ROOT_SIGNATURE, &feature_data, sizeof(feature_data))))
    {
      feature_data.HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_0;
    }

    RootSignatureBuilder rsb;
    rsb.RootConstants(sizeof(ModelViewProjection) / 4, D3D12_SHADER_VISIBILITY_ALL)
        .Range(1, D3D12_DESCRIPTOR_RANGE_TYPE_CBV, D3D12_DESCRIPTOR_RANGE_FLAG_DATA_STATIC, D3D12_SHADER_VISIBILITY_ALL)
        .Range(
          1 + 1 + 1,
          D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
          D3D12_DESCRIPTOR_RANGE_FLAG_DATA_STATIC,
          D3D12_SHADER_VISIBILITY_ALL)
        .Range(
          2,
          D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
          D3D12_DESCRIPTOR_RANGE_FLAG_DATA_STATIC,
          D3D12_SHADER_VISIBILITY_PIXEL)
        .Range(
          1,
          D3D12_DESCRIPTOR_RANGE_TYPE_CBV,
          D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE,
          D3D12_SHADER_VISIBILITY_ALL)
        .StaticSampler(D3D12_FILTER_MIN_MAG_MIP_LINEAR, D3D12_TEXTURE_ADDRESS_MODE_WRAP, D3D12_SHADER_VISIBILITY_PIXEL);
    g_rootsignature_pbr_ntc = rsb.Build(g_device.Get());
    
    const char* vs_names[g_kProfileCount] = {"pbr_ntc_vs", "pbr_ntc_bpp05_vs", "pbr_ntc_bpp10_vs", "pbr_ntc_bpp225_vs"};
    const char* ps_names[g_kProfileCount] = {"pbr_ntc_ps", "pbr_ntc_bpp05_ps", "pbr_ntc_bpp10_ps", "pbr_ntc_bpp225_ps"};
    for (int32_t profile_i = 0; profile_i < g_kProfileCount; profile_i++)
    {
      GraphicsPipelineBuilder gpb;
      gpb.RootSignature(g_rootsignature_pbr_ntc.Get())
          .Input(input_layout, _countof(input_layout))
          .VS(PACK_BLOB_ARGS(vs_names[profile_i]))
          .PS(PACK_BLOB_ARGS(ps_names[profile_i]))
          .DepthEnable(true)
          .CullMode(D3D12_CULL_MODE_BACK);
      g_pipelinestate_pbr_ntc[profile_i] = gpb.Build(g_device.Get());
    }
  }

  // Flat
  {
    GraphicsPipelineBuilder gpb;
    gpb.RootSignature(g_rootsignature_flat.Get())
        .VS(PACK_BLOB_ARGS("flat_vs"))
        .PS(PACK_BLOB_ARGS("flat_ps"))
        .DepthEnable(false)
        .CullMode(D3D12_CULL_MODE_NONE);
    g_pipelinestate_flat = gpb.Build(g_device.Get());

    const char* vs_names[g_kProfileCount] = {"flat_ntc_vs", "flat_ntc_bpp05_vs", "flat_ntc_bpp10_vs", "flat_ntc_bpp225_vs"};
    const char* ps_names[g_kProfileCount] = {"flat_ntc_ps", "flat_ntc_bpp05_ps", "flat_ntc_bpp10_ps", "flat_ntc_bpp225_ps"};
    for (int32_t profile_i = 0; profile_i < g_kProfileCount; profile_i++)
    {
      GraphicsPipelineBuilder gpb_ntc;
      gpb_ntc.RootSignature(g_rootsignature_pbr_ntc.Get())
          .VS(PACK_BLOB_ARGS(vs_names[profile_i]))
          .PS(PACK_BLOB_ARGS(ps_names[profile_i]))
          .DepthEnable(false)
          .CullMode(D3D12_CULL_MODE_NONE);
      g_pipelinestate_flat_ntc[profile_i] = gpb_ntc.Build(g_device.Get());
    }
  }

#if OPENNTC_COOP
  {
    D3D12_INPUT_ELEMENT_DESC input_layout[] = {
      {"SV_Position", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT,
        D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
      {"NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT,
        D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
      {"TANGENT", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT,
        D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
      {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT,
        D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
    };

    D3D12_FEATURE_DATA_ROOT_SIGNATURE feature_data = {};
    feature_data.HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_1;
    if (FAILED(g_device->CheckFeatureSupport(D3D12_FEATURE_ROOT_SIGNATURE, &feature_data, sizeof(feature_data))))
    {
      feature_data.HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_0;
    }

    RootSignatureBuilder rsb;
    rsb.RootConstants(sizeof(ModelViewProjection) / 4, D3D12_SHADER_VISIBILITY_ALL)
        .Range(1, D3D12_DESCRIPTOR_RANGE_TYPE_CBV, D3D12_DESCRIPTOR_RANGE_FLAG_DATA_STATIC, D3D12_SHADER_VISIBILITY_ALL)
        .Range(
          1 + 1 + 1,
          D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
          D3D12_DESCRIPTOR_RANGE_FLAG_DATA_STATIC,
          D3D12_SHADER_VISIBILITY_ALL)
        .Range(
          2,
          D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
          D3D12_DESCRIPTOR_RANGE_FLAG_DATA_STATIC,
          D3D12_SHADER_VISIBILITY_PIXEL)
        .Range(
          1,
          D3D12_DESCRIPTOR_RANGE_TYPE_CBV,
          D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE,
          D3D12_SHADER_VISIBILITY_ALL)
        .StaticSampler(D3D12_FILTER_MIN_MAG_MIP_LINEAR, D3D12_TEXTURE_ADDRESS_MODE_WRAP, D3D12_SHADER_VISIBILITY_PIXEL);
    g_rootsignature_pbr_ntc_coop = rsb.Build(g_device.Get());

    const char* vs_names[g_kProfileCount] = {"pbr_ntc_coop_vs", "pbr_ntc_coop_bpp05_vs", "pbr_ntc_coop_bpp10_vs", "pbr_ntc_coop_bpp225_vs"};
    const char* ps_names[g_kProfileCount] = {"pbr_ntc_coop_ps", "pbr_ntc_coop_bpp05_ps", "pbr_ntc_coop_bpp10_ps", "pbr_ntc_coop_bpp225_ps"};
    for (int32_t profile_i = 0; profile_i < g_kProfileCount; profile_i++)
    {
      GraphicsPipelineBuilder gpb;
      gpb.RootSignature(g_rootsignature_pbr_ntc_coop.Get())
          .Input(input_layout, _countof(input_layout))
          .VS(PACK_BLOB_ARGS(vs_names[profile_i]))
          .PS(PACK_BLOB_ARGS(ps_names[profile_i]))
          .DepthEnable(true)
          .CullMode(D3D12_CULL_MODE_BACK);
      g_pipelinestate_pbr_ntc_coop[profile_i] = gpb.Build(g_device.Get());
    }

    const char* flat_vs_names[g_kProfileCount] = {"flat_ntc_coop_vs", "flat_ntc_coop_bpp05_vs", "flat_ntc_coop_bpp10_vs", "flat_ntc_coop_bpp225_vs"};
    const char* flat_ps_names[g_kProfileCount] = {"flat_ntc_coop_ps", "flat_ntc_coop_bpp05_ps", "flat_ntc_coop_bpp10_ps", "flat_ntc_coop_bpp225_ps"};
    for (int32_t profile_i = 0; profile_i < g_kProfileCount; profile_i++)
    {
      GraphicsPipelineBuilder gpb;
      gpb.RootSignature(g_rootsignature_pbr_ntc_coop.Get())
          .VS(PACK_BLOB_ARGS(flat_vs_names[profile_i]))
          .PS(PACK_BLOB_ARGS(flat_ps_names[profile_i]))
          .DepthEnable(false)
          .CullMode(D3D12_CULL_MODE_NONE);
      g_pipelinestate_flat_ntc_coop[profile_i] = gpb.Build(g_device.Get());
    }
  }
#endif

  // Cubemap

  {
    D3D12_FEATURE_DATA_ROOT_SIGNATURE feature_data = {};
    feature_data.HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_1;
    if (FAILED(g_device->CheckFeatureSupport(D3D12_FEATURE_ROOT_SIGNATURE, &feature_data, sizeof(feature_data))))
    {
      feature_data.HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_0;
    }

    RootSignatureBuilder rsb;
    rsb.RootConstants(sizeof(CubemapTransforms) / 4, D3D12_SHADER_VISIBILITY_PIXEL)
        .Range(
          1,
          D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
          D3D12_DESCRIPTOR_RANGE_FLAG_DATA_STATIC,
          D3D12_SHADER_VISIBILITY_PIXEL)
        .Range(
          1,
          D3D12_DESCRIPTOR_RANGE_TYPE_CBV,
          D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE,
          D3D12_SHADER_VISIBILITY_PIXEL)
        .StaticSampler(D3D12_FILTER_MIN_MAG_MIP_LINEAR, D3D12_TEXTURE_ADDRESS_MODE_WRAP, D3D12_SHADER_VISIBILITY_PIXEL);
    g_rootsignature_cubemap = rsb.Build(g_device.Get());

    GraphicsPipelineBuilder gpb;
    gpb.RootSignature(g_rootsignature_cubemap.Get())
        .VS(PACK_BLOB_ARGS("cubemap_vs"))
        .PS(PACK_BLOB_ARGS("cubemap_ps"))
        .DepthEnable(false)
        .CullMode(D3D12_CULL_MODE_BACK);
    g_pipelinestate_cubemap = gpb.Build(g_device.Get());
  }

  // TAA resolve. Scene renders into g_tex_frame, the resolve blends it into the write half of the
  //  accum ping-pong, and the result is copied to the backbuffer. All three textures match the
  //  swapchain format so the display step and the history seed can be plain CopyResource.

  {
    CreateTaaResources();

    RootSignatureBuilder rsb;
    rsb.RootConstants(sizeof(TaaConstants) / 4, D3D12_SHADER_VISIBILITY_PIXEL)
        .Range(
          1,
          D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
          D3D12_DESCRIPTOR_RANGE_FLAG_DATA_STATIC_WHILE_SET_AT_EXECUTE,
          D3D12_SHADER_VISIBILITY_PIXEL)
        .Range(
          1,
          D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
          D3D12_DESCRIPTOR_RANGE_FLAG_DATA_STATIC_WHILE_SET_AT_EXECUTE,
          D3D12_SHADER_VISIBILITY_PIXEL)
        .Range(
          1,
          D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
          D3D12_DESCRIPTOR_RANGE_FLAG_DATA_STATIC_WHILE_SET_AT_EXECUTE,
          D3D12_SHADER_VISIBILITY_PIXEL)
        .Range(
          1,
          D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
          D3D12_DESCRIPTOR_RANGE_FLAG_DATA_STATIC_WHILE_SET_AT_EXECUTE,
          D3D12_SHADER_VISIBILITY_PIXEL)
        .StaticSampler(
          D3D12_FILTER_MIN_MAG_MIP_LINEAR,
          D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
          D3D12_SHADER_VISIBILITY_PIXEL);
    g_rootsignature_taa = rsb.Build(g_device.Get());

    GraphicsPipelineBuilder gpb;
    gpb.RootSignature(g_rootsignature_taa.Get())
        .VS(PACK_BLOB_ARGS("taa_vs"))
        .PS(PACK_BLOB_ARGS("taa_ps"))
        .DepthEnable(false)
        .DsvFormat(DXGI_FORMAT_UNKNOWN)
        .CullMode(D3D12_CULL_MODE_NONE);
    g_pipelinestate_taa = gpb.Build(g_device.Get());
  }

  SetStatus("Ready");

  LoadIBL();

  g_contentloaded = true;

  ResizeDepthBuffer(g_width, g_height);
}

void UploadCompressedData(openntc::CompressedData& cdata)
{
  Flush(g_queue, g_fence, &g_fenceval, g_fence_event);

  {
    openntc::NTCConstants ntc_info;
    openntc::FillNTCConstants(cdata, ntc_info);

    uint64_t cbv_size = RoundUpTo(sizeof(openntc::NTCConstants), 256);

    g_buffer_ntc_info = CreateBufferWithData(
      g_device.Get(),
      D3D12_HEAP_TYPE_GPU_UPLOAD,
      cbv_size,
      sizeof(openntc::NTCConstants),
      &ntc_info);
    
    D3D12_CONSTANT_BUFFER_VIEW_DESC cbv_desc = {};
    cbv_desc.BufferLocation = g_buffer_ntc_info->GetGPUVirtualAddress();
    cbv_desc.SizeInBytes = cbv_size;
    g_device->CreateConstantBufferView(&cbv_desc, g_dhandle_ntc_info.cpu_);
  }

  uint64_t g0_size = 0;
  uint64_t g1_size = 0;
  for (int level_i = 0; level_i < cdata.level_count_; level_i++)
  {
    g0_size += cdata.g0_size_[level_i];
    g1_size += cdata.g1_size_[level_i];
  }

  {
    g_buffer_g0 = CreateBuffer(g_device.Get(), D3D12_HEAP_TYPE_GPU_UPLOAD, g0_size);
    
    D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
    srv_desc.Format = DXGI_FORMAT_R32_UINT;
    srv_desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srv_desc.Buffer.FirstElement = 0;
    srv_desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
    srv_desc.Buffer.NumElements = g0_size / sizeof(uint32_t);
    srv_desc.Buffer.StructureByteStride = 0;
    srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;

    g_device->CreateShaderResourceView(g_buffer_g0.Get(), &srv_desc, g_dhandle_ntc_data[0].cpu_);
    
    void* mapped = nullptr;
    D3D12_RANGE read_range = {0, 0};
    g_buffer_g0->Map(0, &read_range, &mapped);
    for (int level_i = 0; level_i < cdata.level_count_; level_i++)
      memcpy(static_cast<char*>(mapped) + cdata.g0_offset_[level_i], cdata.g0_[level_i], cdata.g0_size_[level_i]);
    g_buffer_g0->Unmap(0, nullptr);
  }

  {
    g_buffer_g1 = CreateBuffer(g_device.Get(), D3D12_HEAP_TYPE_GPU_UPLOAD, g1_size);

    D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
    srv_desc.Format = DXGI_FORMAT_R32_UINT;
    srv_desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srv_desc.Buffer.FirstElement = 0;
    srv_desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
    srv_desc.Buffer.NumElements = g1_size / sizeof(uint32_t);
    srv_desc.Buffer.StructureByteStride = 0;
    srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;

    g_device->CreateShaderResourceView(g_buffer_g1.Get(), &srv_desc, g_dhandle_ntc_data[1].cpu_);
    
    void* mapped = nullptr;
    D3D12_RANGE read_range = {0, 0};
    g_buffer_g1->Map(0, &read_range, &mapped);
    for (int level_i = 0; level_i < cdata.level_count_; level_i++)
      memcpy(static_cast<char*>(mapped) + cdata.g1_offset_[level_i], cdata.g1_[level_i], cdata.g1_size_[level_i]);
    g_buffer_g1->Unmap(0, nullptr);
  }

  {
    g_buffer_decoder = CreateBufferWithData(
      g_device.Get(),
      D3D12_HEAP_TYPE_GPU_UPLOAD,
      cdata.decoder_size_,
      cdata.decoder_size_,
      cdata.decoder_);

    D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
    srv_desc.Format = DXGI_FORMAT_R32_TYPELESS;
    srv_desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srv_desc.Buffer.FirstElement = 0;
    srv_desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
    srv_desc.Buffer.NumElements = cdata.decoder_size_ / (sizeof(uint32_t));
    srv_desc.Buffer.StructureByteStride = 0;
    srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;

    g_device->CreateShaderResourceView(g_buffer_decoder.Get(), &srv_desc, g_dhandle_ntc_data[2].cpu_);
  }
}

void LoadCompressedDataFromContext(SharedContext::Access& access)
{
  openntc::CompressedData cdata = access.ctx_.GetCompressedData();
  UploadCompressedData(cdata);
  g_compressed_from_training = true;
  g_compressed_dirty = true;
  g_compressed_matches_manifest = true;
  g_profile = cdata.profile_;
}

void LoadCompressedDataFromFile(const std::string& path)
{
  openntc::Result res = openntc::Load(path, g_fil_data);
  if (res != openntc::Result::Success)
  {
    SetStatus("Compressed load failed (Error Code %d): %s", static_cast<int>(res), path.c_str());
    return;
  }

  openntc::CompressedData cdata = g_fil_data.Data();
  openntc::Profile file_profile = cdata.profile_;

  // Note: This only checks that the dimensions of the uncompressed and compressed versions match.
  //  Which might be fine, but it would be useful to bake a manifest path inside the .ntc for
  //  error-checking.
  int32_t manifest_dim = 0;
  int32_t manifest_mips = 0;
  if (!g_manifest_path.empty())
  {
    auto oaccess = g_ctx.TryAcquire();
    if (oaccess.has_value())
    {
      manifest_dim = oaccess.value().ctx_.GetMipDim(0);
      manifest_mips = oaccess.value().ctx_.GetTextureData().mip_count_;
    }
  }
  g_compressed_matches_manifest = (cdata.dim_ == manifest_dim) && (cdata.mip_count_ == manifest_mips);

  UploadCompressedData(cdata);
  g_compressed_from_training = false;
  g_compressed_dirty = false;
  const bool switched_profile = (file_profile != g_profile);
  g_profile = file_profile;
  TransitionGuiState(GuiEvent::CompressedLoaded);
  if (switched_profile)
    SetStatus(
      "Loaded Compressed Data: %s (switched to profile %s)",
      path.c_str(),
      g_map_profile_to_name[static_cast<int32_t>(g_profile)]);
  else if (g_compressed_matches_manifest)
    SetStatus("Loaded Compressed Data: %s", path.c_str());
  else
    SetStatus(
      "Loaded Compressed Data: %s (Warning: Dimension mismatch! %dx%d/%d mips vs manifest %dx%d/%d mips)",
      path.c_str(),
      cdata.dim_,
      cdata.dim_,
      cdata.mip_count_,
      manifest_dim,
      manifest_dim,
      manifest_mips);
}

static void SetPipelineStateForShader(Shader shader)
{
  if (shader == Shader::PBR)
  {
    g_commandlist->SetPipelineState(g_pipelinestate_pbr.Get());
    g_commandlist->SetGraphicsRootSignature(g_rootsignature_pbr.Get());
  }
  else if (shader == Shader::PBR_NTC)
  {
    g_commandlist->SetPipelineState(g_pipelinestate_pbr_ntc[static_cast<int32_t>(g_profile)].Get());
    g_commandlist->SetGraphicsRootSignature(g_rootsignature_pbr_ntc.Get());
  }
  else if (shader == Shader::PBR_NTC_COOP)
  {
    g_commandlist->SetPipelineState(g_pipelinestate_pbr_ntc_coop[static_cast<int32_t>(g_profile)].Get());
    g_commandlist->SetGraphicsRootSignature(g_rootsignature_pbr_ntc_coop.Get());
  }
}

static void SetDescriptorsForShader(Shader shader)
{
  if (shader == Shader::PBR)
  {
    ModelViewProjection mvp = {};
    mvp.model_to_world_ = g_model_mat;
    mvp.world_to_view_ = g_view_mat;
    mvp.view_to_proj_ = g_proj_mat;

    D3D12_GPU_DESCRIPTOR_HANDLE tex_color_handle = g_dhandle_pbr_tex[0].gpu_;
    D3D12_GPU_DESCRIPTOR_HANDLE srv_handle = g_dhandle_ibl[0].gpu_;
    D3D12_GPU_DESCRIPTOR_HANDLE cbv_lighing_handle = g_dhandle_lparams.gpu_;
    g_commandlist->SetGraphicsRoot32BitConstants(0, sizeof(ModelViewProjection) / 4, &mvp, 0);
    g_commandlist->SetGraphicsRootDescriptorTable(1, tex_color_handle);
    g_commandlist->SetGraphicsRootDescriptorTable(2, srv_handle);
    g_commandlist->SetGraphicsRootDescriptorTable(3, cbv_lighing_handle);
  }
  else if (shader == Shader::PBR_NTC || shader == Shader::PBR_NTC_COOP)
  {
    ModelViewProjection mvp = {};
    mvp.model_to_world_ = g_model_mat;
    mvp.world_to_view_ = g_view_mat;
    mvp.view_to_proj_ = g_proj_mat;

    D3D12_GPU_DESCRIPTOR_HANDLE cbv_handle = g_dhandle_ntc_info.gpu_;
    D3D12_GPU_DESCRIPTOR_HANDLE srv_handle_ntc = g_dhandle_ntc_data[0].gpu_;
    D3D12_GPU_DESCRIPTOR_HANDLE srv_handle_ibl = g_dhandle_ibl[0].gpu_;
    D3D12_GPU_DESCRIPTOR_HANDLE cbv_lighing_handle = g_dhandle_lparams.gpu_;
    g_commandlist->SetGraphicsRoot32BitConstants(0, sizeof(ModelViewProjection) / 4, &mvp, 0);
    g_commandlist->SetGraphicsRootDescriptorTable(1, cbv_handle);
    g_commandlist->SetGraphicsRootDescriptorTable(2, srv_handle_ntc);
    g_commandlist->SetGraphicsRootDescriptorTable(3, srv_handle_ibl);
    g_commandlist->SetGraphicsRootDescriptorTable(4, cbv_lighing_handle);
  }
}

static void DrawFlatPane(FlatSource source, const D3D12_VIEWPORT& viewport)
{
  if (g_tex_count == 0)
    return;
  FlatParams params = {};
  params.pane_dim_[0] = viewport.Width;
  params.pane_dim_[1] = viewport.Height;
  params.semantic_ = static_cast<int32_t>(g_tex_semantics[g_gui_flat_channel]);
  params.channels_ = openntc::GetChannelCountForSemantic(g_tex_semantics[g_gui_flat_channel]);

  if (source == FlatSource::NTC && g_app_state == GuiState::Compressed)
  {
    g_commandlist->SetPipelineState(g_pipelinestate_flat_ntc[static_cast<int32_t>(g_profile)].Get());
    g_commandlist->SetGraphicsRootSignature(g_rootsignature_pbr_ntc.Get());
    SetDescriptorsForShader(Shader::PBR_NTC);
    g_commandlist->SetGraphicsRoot32BitConstants(0, sizeof(FlatParams) / 4, &params, 0);
  }
  else if (source == FlatSource::NTC_COOP && OPENNTC_COOP && g_app_state == GuiState::Compressed)
  {
    g_commandlist->SetPipelineState(g_pipelinestate_flat_ntc_coop[static_cast<int32_t>(g_profile)].Get());
    g_commandlist->SetGraphicsRootSignature(g_rootsignature_pbr_ntc_coop.Get());
    SetDescriptorsForShader(Shader::PBR_NTC_COOP);
    g_commandlist->SetGraphicsRoot32BitConstants(0, sizeof(FlatParams) / 4, &params, 0);
  }
  else
  {
    g_commandlist->SetPipelineState(g_pipelinestate_flat.Get());
    g_commandlist->SetGraphicsRootSignature(g_rootsignature_flat.Get());
    g_commandlist->SetGraphicsRoot32BitConstants(0, sizeof(FlatParams) / 4, &params, 0);
    g_commandlist->SetGraphicsRootDescriptorTable(1, g_dhandle_tex[g_gui_flat_channel].gpu_);
  }
  g_commandlist->DrawInstanced(6, 1, 0, 0);
}

void Update()
{
  static uint64_t framecounter = 0;
  static double elapsed_seconds = 0.0;
  static std::chrono::high_resolution_clock clock;
  static auto t0 = clock.now();

  framecounter++;
  auto t1 = clock.now();
  std::chrono::duration<double> dT = t1 - t0;
  t0 = t1;

  elapsed_seconds += dT.count();
  g_total_seconds += dT.count();
  if (elapsed_seconds > 1.0)
  {
    char buffer[500];
    auto fps = framecounter / elapsed_seconds;
    sprintf_s(buffer, 500, "FPS: %f\n", fps);
    OutputDebugStringA(buffer);
    framecounter = 0;
    elapsed_seconds = 0.0;
  }
}

static bool GetPathFromShellItem(IShellItem* item, std::string& o_path)
{
  PWSTR wpath = nullptr;
  if (FAILED(item->GetDisplayName(SIGDN_FILESYSPATH, &wpath)))
    return false;
  int size = WideCharToMultiByte(CP_UTF8, 0, wpath, -1, nullptr, 0, nullptr, nullptr);
  o_path.assign(size - 1, 0);
  WideCharToMultiByte(CP_UTF8, 0, wpath, -1, o_path.data(), size, nullptr, nullptr);
  CoTaskMemFree(wpath);
  return true;
}

static bool ShowNtcFileDialog(bool save, std::string& o_path)
{
  static const COMDLG_FILTERSPEC kFilter = { L"NTC compressed data (*.ntc)", L"*.ntc" };

  ComPtr<IFileDialog> dialog;
  if (save)
    VERIFY(CoCreateInstance(CLSID_FileSaveDialog, NULL, CLSCTX_ALL, IID_IFileDialog, &dialog));
  else
    VERIFY(CoCreateInstance(CLSID_FileOpenDialog, NULL, CLSCTX_ALL, IID_IFileDialog, &dialog));
  dialog->SetFileTypes(1, &kFilter);
  dialog->SetDefaultExtension(L"ntc");

  if (FAILED(dialog->Show(NULL)))
    return false;
  ComPtr<IShellItem> item;
  if (FAILED(dialog->GetResult(&item)))
    return false;
  return GetPathFromShellItem(item.Get(), o_path);
}

static bool ConfirmDiscardUnsaved()
{
  if (!g_compressed_dirty)
    return true;
  return MessageBoxA(
    g_hwnd,
    "The trained compression has not been saved. Discard it?",
    "openntc-gui",
    MB_YESNO | MB_ICONWARNING) == IDYES;
}

void PerformLoadManifest()
{
  if (!ConfirmDiscardUnsaved())
    return;

  ComPtr<IFileOpenDialog> open_dialog;

  VERIFY(CoCreateInstance(CLSID_FileOpenDialog, NULL, CLSCTX_ALL, IID_IFileOpenDialog, &open_dialog));

  if (SUCCEEDED(open_dialog->Show(NULL)))
  {
    ComPtr<IShellItem> open_item;
    if (SUCCEEDED(open_dialog->GetResult(&open_item)))
    {
      PWSTR wfile_path;
      VERIFY(open_item->GetDisplayName(SIGDN_FILESYSPATH, &wfile_path));
      int file_path_size = WideCharToMultiByte(CP_UTF8, 0, wfile_path, -1, nullptr, 0, nullptr, nullptr);
      std::string file_path(file_path_size - 1, 0);
      WideCharToMultiByte(CP_UTF8, 0, wfile_path, -1, file_path.data(), file_path_size, nullptr, nullptr);
      CoTaskMemFree(wfile_path);
      auto oaccess = g_ctx.TryAcquire();
      if (oaccess.has_value())
      {
        SharedContext::Access& access = oaccess.value();
        openntc::Result load_res = access.ctx_.LoadManifest(file_path);

        if (load_res == openntc::Result::Success)
        {
          RebuildTextureResources(access);
          g_manifest_path = file_path;
          g_compressed_matches_manifest = false;
          TransitionGuiState(GuiEvent::ManifestLoaded);
          SetStatus("Loaded Manifest %s", file_path.c_str());
        }
        else
        {
          SetStatus("Manifest load failure (Error Code %d): %s", static_cast<int>(load_res), file_path.c_str());
        }
      }
      else
      {
        SetStatus("Context busy: Manifest load failed");
      }
    }
  }
}

static bool PerformProfileChange(openntc::Profile profile)
{
  if (!ConfirmDiscardUnsaved())
    return false;

  auto oaccess = g_ctx.TryAcquire();
  if (!oaccess.has_value())
  {
    SetStatus("Context busy: Profile switch failed");
    return false;
  }
  SharedContext::Access& access = oaccess.value();

  access.ctx_.Destroy();
  openntc::ContextInitInfo init_info = {};
  init_info.profile_ = profile;
  openntc::Result res = access.ctx_.Init(init_info);
  if (res != openntc::Result::Success)
  {
    SetStatus("Profile switch failed (Error Code %d)", static_cast<int>(res));
    return false;
  }

  if (g_manifest_path.empty())
  {
    SetStatus("Training profile set to %s", g_map_profile_to_name[static_cast<int32_t>(profile)]);
    return true;
  }

  res = access.ctx_.LoadManifest(g_manifest_path);
  if (res != openntc::Result::Success)
  {
    SetStatus("Profile switch failed (Error Code %d): %s",
              static_cast<int>(res), g_manifest_path.c_str());
    return false;
  }
  RebuildTextureResources(access);

  g_compressed_dirty = false;
  g_compressed_from_training = false;
  g_compressed_matches_manifest = false;
  TransitionGuiState(GuiEvent::ManifestLoaded);
  SetStatus("Profile switched to %s", g_map_profile_to_name[static_cast<int32_t>(profile)]);
  return true;
}

void PerformLoadCompressed()
{
  if (!ConfirmDiscardUnsaved())
    return;
  std::string path;
  if (!ShowNtcFileDialog(false, path))
    return;
  LoadCompressedDataFromFile(path);
}

void PerformSaveCompressed()
{
  if (g_app_state != GuiState::Compressed)
  {
    SetStatus("Compressed Data Unavailable");
    return;
  }
  std::string path;
  if (!ShowNtcFileDialog(true, path))
    return;

  openntc::Result res = openntc::Result::Success;
  if (g_compressed_from_training)
  {
    auto oaccess = g_ctx.TryAcquire();
    if (!oaccess.has_value())
    {
      SetStatus("Context busy: Save Failed");
      return;
    }
    res = openntc::Dump(path, oaccess.value().ctx_.GetCompressedData());
  }
  else
  {
    res = openntc::Dump(path, g_fil_data.Data());
  }

  if (res != openntc::Result::Success)
  {
    SetStatus("Save failed (Error Code %d): %s", static_cast<int>(res), path.c_str());
    return;
  }
  g_compressed_dirty = false;
  TransitionGuiState(GuiEvent::Saved);
  SetStatus("Saved %s", path.c_str());
}

// Do we have all the data required to render the model with the desired shader?

static bool IsShaderAvailable(Shader shader)
{
  switch (shader)
  {
    case Shader::PBR:
      return !g_manifest_path.empty();
    case Shader::PBR_NTC:
      return g_app_state == GuiState::Compressed;
    case Shader::PBR_NTC_COOP:
      return OPENNTC_COOP && g_app_state == GuiState::Compressed;
    default:
      return false;
  }
}

// Display only the shader options that are valid for our current GUI state
static void ShaderCombo(const char* label, int32_t* value)
{
  if (!IsShaderAvailable(static_cast<Shader>(*value)) &&
    IsShaderAvailable(Shader::PBR) &&
    (*value == static_cast<int32_t>(Shader::PBR_NTC) || *value == static_cast<int32_t>(Shader::PBR_NTC_COOP)))
    *value = static_cast<int32_t>(Shader::PBR);

  if (ImGui::BeginCombo(label, g_map_shader_to_name[*value]))
  {
    for (int32_t i = 0; i < static_cast<int32_t>(Shader::Count); i++)
    {
      ImGui::BeginDisabled(!IsShaderAvailable(static_cast<Shader>(i)));
      if (ImGui::Selectable(g_map_shader_to_name[i], *value == i))
        *value = i;
      ImGui::EndDisabled();
    }
    ImGui::EndCombo();
  }
}

static bool ModeButton(const char* label, bool selected, float width)
{
  if (selected)
  {
    const ImVec4 pressed = ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive);
    ImGui::PushStyleColor(ImGuiCol_Button, pressed);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, pressed);
  }
  const bool clicked = ImGui::Button(label, {width, 0.0f});
  if (selected)
    ImGui::PopStyleColor(2);
  return clicked;
}

static void ModeButtonRow(const char* const* names, int32_t count, int32_t* value)
{
  const float width =
    (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x * (count - 1)) / count;
  for (int32_t i = 0; i < count; i++)
  {
    if (i > 0)
      ImGui::SameLine();
    if (ModeButton(names[i], *value == i, width))
      *value = i;
  }
}

void Render()
{
  ImGui_ImplDX12_NewFrame();
  ImGui_ImplWin32_NewFrame();
  ImGui::NewFrame();

  // Build Menu Bar
  {
    if (ImGui::BeginMainMenuBar())
    {
      const bool loads_enabled = (g_app_state != GuiState::Training);
      const bool save_enabled = (g_app_state == GuiState::Compressed);
      if (ImGui::BeginMenu("File"))
      {
        if (ImGui::MenuItem("Load Manifest...", "Ctrl+O", false, loads_enabled))
        {
          PerformLoadManifest();
        }
        if (ImGui::MenuItem("Load Compressed...", "Ctrl+Shift+O", false, loads_enabled))
        {
          PerformLoadCompressed();
        }
        if (ImGui::MenuItem("Save Compressed...", "Ctrl+S", false, save_enabled))
        {
          PerformSaveCompressed();
        }
        ImGui::Separator();
        if (ImGui::MenuItem("Exit", "Alt+F4"))
        {
          if (ConfirmDiscardUnsaved())
            PostQuitMessage(0);
        }
        ImGui::EndMenu();
      }
      ImGui::EndMainMenuBar();
    }
  }

  if (ImGui::Shortcut(ImGuiMod_Ctrl | ImGuiKey_O, ImGuiInputFlags_RouteGlobal) &&
      g_app_state != GuiState::Training)
  {
    PerformLoadManifest();
  }

  if (ImGui::Shortcut(
    ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_O,
    ImGuiInputFlags_RouteGlobal) && g_app_state != GuiState::Training)
  {
    PerformLoadCompressed();
  }

  if (ImGui::Shortcut(ImGuiMod_Ctrl | ImGuiKey_S, ImGuiInputFlags_RouteGlobal) &&
      g_app_state == GuiState::Compressed)
  {
    PerformSaveCompressed();
  }

  const ImGuiViewport* vp = ImGui::GetMainViewport();
  const float sidebar_w = 350.0f;
  const float footer_h = std::ceil(ImGui::GetTextLineHeight() + 2.0f * ImGui::GetStyle().WindowPadding.y);
  const ImGuiWindowFlags pinned_flags = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                                        ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar;
                                      
  ImGui::SetNextWindowPos({vp->WorkPos.x + vp->WorkSize.x - sidebar_w, vp->WorkPos.y});
  ImGui::SetNextWindowSize({sidebar_w, vp->WorkSize.y - footer_h});
  ImGui::Begin("Sidebar", nullptr, pinned_flags);
  Shader shader_left = static_cast<Shader>(g_gui_shader_left);
  Shader shader_right = static_cast<Shader>(g_gui_shader_right);
  ImGui::SeparatorText("View");
  {
    const char* const view_names[] = {"3D", "2D"};
    int32_t view = g_gui_flat_view ? 1 : 0;
    ModeButtonRow(view_names, 2, &view);
    g_gui_flat_view = view == 1;
  }
  if (g_gui_flat_view)
  {
    ImGui::SeparatorText("Channel");
    const bool ntc_ok = g_app_state == GuiState::Compressed;
    const bool coop_ok = OPENNTC_COOP && ntc_ok;
    auto IsSourceValid = [&](int32_t source) {
      return source == static_cast<int32_t>(FlatSource::NTC) ? ntc_ok
           : source == static_cast<int32_t>(FlatSource::NTC_COOP) ? coop_ok
           : true;
    };
    for (int32_t* value : {&g_gui_flat_left, &g_gui_flat_right})
    {
      if (!IsSourceValid(*value))
        *value = static_cast<int32_t>(FlatSource::Reference);
    }
    auto source_combo = [&](const char* label, int32_t* value) {
      if (ImGui::BeginCombo(label, g_map_flat_source_to_name[*value]))
      {
        for (int32_t i = 0; i < static_cast<int32_t>(FlatSource::Count); i++)
        {
          ImGui::BeginDisabled(!IsSourceValid(i));
          if (ImGui::Selectable(g_map_flat_source_to_name[i], *value == i))
            *value = i;
          ImGui::EndDisabled();
        }
        ImGui::EndCombo();
      }
    };
    source_combo("Left", &g_gui_flat_left);
    source_combo("Right", &g_gui_flat_right);
    if (g_tex_count > 0)
    {
      const char* current = g_map_semantic_to_name[static_cast<int32_t>(g_tex_semantics[g_gui_flat_channel]) - 1];
      if (ImGui::BeginCombo("Channel", current))
      {
        for (int32_t i = 0; i < g_tex_count; i++)
        {
          const char* name = g_map_semantic_to_name[static_cast<int32_t>(g_tex_semantics[i]) - 1];
          if (ImGui::Selectable(name, g_gui_flat_channel == i))
            g_gui_flat_channel = i;
        }
        ImGui::EndCombo();
      }
    }
    ImGui::Checkbox("TAA", &g_gui_taa);
  }
  else
  {
  ImGui::SeparatorText("Shading");
  ModeButtonRow(g_map_camera_mode_to_name, static_cast<int32_t>(CameraMode::Count), &g_gui_camera_mode);
  ShaderCombo("Left", &g_gui_shader_left);
  ShaderCombo("Right", &g_gui_shader_right);
  shader_left = static_cast<Shader>(g_gui_shader_left);
  shader_right = static_cast<Shader>(g_gui_shader_right);
  if (g_app_state == GuiState::Compressed && !g_compressed_matches_manifest)
  {
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.8f, 0.2f, 1.0f));
    ImGui::TextWrapped(
      "Warning: Compressed data does not match the loaded manifest dimensions. Comparison may not be meaningful!");
    ImGui::PopStyleColor();
  }
  ImGui::SliderFloat("FOV", &g_fov_y, 10.0f, 180.0f);
  ImGui::SliderFloat("Displacement Scale", &g_gui_displacement_scale, 0.0f, 0.5f);
  ImGui::SliderFloat("Normal Scale", &g_gui_normal_scale, 0.0f, 10.0f);
  ImGui::SliderFloat("Exposure", &g_gui_exposure, 0.5f, 1.5f);
  ImGui::Combo("Model", &g_gui_model, g_map_model_to_name, g_kModelCount);
  ImGui::Checkbox("TAA", &g_gui_taa);
  }

  ImGui::SeparatorText("Train");
  const bool can_train = !g_manifest_path.empty() &&
    (g_app_state == GuiState::MaterialLoaded || g_app_state == GuiState::Compressed);
  ImGui::BeginDisabled(!can_train);
  {
    const int32_t prev_profile = g_gui_profile;
    if (ImGui::Combo(
      "Profile",
      &g_gui_profile,
      g_map_profile_to_name,
      g_kProfileCount) && g_gui_profile != prev_profile)
    {
      if (!PerformProfileChange(static_cast<openntc::Profile>(g_gui_profile)))
        g_gui_profile = prev_profile;
    }
  }
  ImGui::Combo("Quality", &g_gui_quality, g_map_quality_to_name, g_kQualityCount);
  bool train_button = ImGui::Button("Train");
  ImGui::EndDisabled();
  {
    std::optional<SharedContext::Access> oaccess = g_ctx.TryAcquire();
    SharedFields fields = g_shared_fields.load(std::memory_order_seq_cst);
    if (oaccess.has_value())
    {
      SharedContext::Access& access = oaccess.value();
      if (train_button)
      {
        if (!fields.train_in_progress_ && ConfirmDiscardUnsaved())
        {
          g_compressed_dirty = false;
          fields.train_in_progress_ = true;
          fields.train_complete_ = false;
          fields.train_steps_ = 0;
          const openntc::Quality quality = static_cast<openntc::Quality>(g_gui_quality);
          fields.train_total_steps_ = openntc::GetStepsForQuality(quality);
          g_shared_fields.store(fields, std::memory_order_seq_cst);
          g_stop_training.store(false, std::memory_order_seq_cst);
          g_train_job = std::async(std::launch::async, PerformTrainingJob, quality);
          TransitionGuiState(GuiEvent::TrainStarted);
          SetStatus("Training Started: %d total steps", fields.train_total_steps_);
        }
        else if (fields.train_in_progress_)
        {
          SetStatus("Training already in progress");
        }
      }
    }
    else if (train_button)
    {
      SetStatus("Context busy: Train start failed");
    }
    if (fields.train_in_progress_)
    {
      ImGui::ProgressBar((float)fields.train_steps_ / fields.train_total_steps_);
    }
    // Worker ended without completing (cancelled/aborted)
    if (g_app_state == GuiState::Training && !fields.train_in_progress_ && !fields.train_complete_)
    {
      TransitionGuiState(GuiEvent::TrainStopped);
      SetStatus("Training stopped before completion");
    }
    if (g_app_state == GuiState::Training && fields.train_complete_ && oaccess.has_value())
    {
      SharedContext::Access& access = oaccess.value();
      LoadCompressedDataFromContext(access);
      SetStatus("Training Complete: PSNR %.2f dB", fields.eval_psnr_);
      TransitionGuiState(GuiEvent::TrainFinished);
    }
    if (fields.train_complete_ && g_compressed_from_training)
    {
      ImGui::LabelText("PSNR", "%f", fields.eval_psnr_);
      ImGui::LabelText("MSE", "%f", fields.eval_mse_);
    }
  }
  ImGui::End();

  ImGui::SetNextWindowPos({vp->WorkPos.x, vp->WorkPos.y + vp->WorkSize.y - footer_h});
  ImGui::SetNextWindowSize({vp->WorkSize.x, footer_h});
  ImGui::Begin("Footer", nullptr, pinned_flags | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
  ImGui::Text("openntc-gui v0.1 pre-release");
  ImGui::SameLine();
  ImGui::TextDisabled("|");
  ImGui::SameLine();
  ImGui::TextDisabled("%s", GetNameForGuiState(g_app_state));
  ImGui::SameLine();
  ImGui::TextDisabled("|");
  ImGui::SameLine();
  ImGui::TextDisabled("%s", g_map_profile_to_name[static_cast<int32_t>(g_profile)]);
  ImGui::SameLine();
  ImGui::TextDisabled("|");
  ImGui::SameLine();
  ImGui::TextUnformatted(g_status_text);
  ImGui::End();

  float cx = vp->WorkPos.x;
  float cy = vp->WorkPos.y;
  float cw = vp->WorkSize.x - sidebar_w;
  float ch = vp->WorkSize.y - footer_h;

  float aspect_ratio = cw / (2.0f * static_cast<float>(ch));
  float fov_y = XMConvertToRadians(g_fov_y);
  float fov_x = 2.0f * std::atanf(std::tanf(fov_y / 2.0f) * aspect_ratio);
  float fov = std::min(fov_y, fov_x);

  CameraMode cmode = static_cast<CameraMode>(g_gui_camera_mode);

  // Bounding sphere radius for cube with corner extent at +-1
  float bounding_sphere_radius = std::sqrtf(3.0f);
  // Some breathing room for the distance quantity
  float distance_spacing_factor = 1.05f;
  float eye_distance = distance_spacing_factor * bounding_sphere_radius / std::sinf(fov / 2.0f);
  const XMVECTOR focus_pos = XMVectorSet(0.0f, 0.0f, 0.0f, 1.0f);
  const XMVECTOR up_dir = XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f);

  XMVECTOR eye_pos = XMVectorSet(0.0f, 0.0f, -eye_distance, 1.0f);
  if (cmode == CameraMode::Static || cmode == CameraMode::Orbit)
  {
    float angle = (cmode == CameraMode::Orbit) ? static_cast<float>(std::fmod(g_total_seconds, std::acos(-1.0) * 2.0)) : 0.0f;
    g_model_mat = XMMatrixIdentity();
    eye_pos = XMVector3Transform(XMVectorSet(0.0f, 0.0f, -eye_distance, 1.0f), XMMatrixRotationY(-angle));
    g_view_mat = XMMatrixLookAtLH(eye_pos, focus_pos, up_dir);
    g_proj_mat = XMMatrixPerspectiveFovLH(fov, aspect_ratio, 0.1f, 100.0f);
  }
  else if (cmode == CameraMode::Controlled)
  {
    eye_pos = focus_pos + eye_distance * XMVectorSet(
      std::cosf(g_pitch) * std::sinf(g_yaw),
      std::sinf(g_pitch),
      std::cosf(g_pitch) * std::cosf(g_yaw),
      0.0f);
    g_model_mat = XMMatrixIdentity();
    g_view_mat = XMMatrixLookAtLH(eye_pos, focus_pos, up_dir);
    g_proj_mat = XMMatrixPerspectiveFovLH(fov_y, aspect_ratio, 0.1f, 100.0f);
  }

  const XMMATRIX view_proj_nojitter = XMMatrixMultiply(g_view_mat, g_proj_mat);

  XMFLOAT2 taa_jitter_px = XMFLOAT2(0.0f, 0.0f);
  if (g_gui_taa)
  {
    static const float kHalton2[8] = { 0.5f, 0.25f, 0.75f, 0.125f, 0.625f, 0.375f, 0.875f, 0.0625f };
    static const float kHalton3[8] = {0.333333f, 0.666667f, 0.111111f, 0.444444f, 0.777778f, 0.222222f, 0.555556f,
      0.888889f};
    static uint32_t jitter_frame = 0;
    jitter_frame = (jitter_frame + 1) % 8;
    float jitter_x = (kHalton2[jitter_frame] - 0.5f) * 2.0f / (cw * 0.5f);
    float jitter_y = (kHalton3[jitter_frame] - 0.5f) * 2.0f / ch;
    g_proj_mat.r[2] = XMVectorAdd(g_proj_mat.r[2], XMVectorSet(jitter_x, jitter_y, 0.0f, 0.0f));
    taa_jitter_px = XMFLOAT2(kHalton2[jitter_frame] - 0.5f, -(kHalton3[jitter_frame] - 0.5f));
  }

  {
    // Remove default ImGUI padding for hidden drag control
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGuiWindowFlags drag_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoScrollbar |
                ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoBringToFrontOnFocus |
                ImGuiWindowFlags_NoNavFocus;
    ImGui::SetNextWindowPos({cx, cy});
    ImGui::SetNextWindowSize({cw, ch});
    ImGui::Begin("Viewport Input", nullptr, drag_flags);
    ImGui::InvisibleButton("Viewport Drag", {cw, ch}, ImGuiButtonFlags_MouseButtonLeft);
    if (ImGui::IsItemActive() && !g_gui_flat_view)
    {
      g_gui_camera_mode = static_cast<int32_t>(CameraMode::Controlled);

      const float sensitivity = 0.008f;
      ImGuiIO& io = ImGui::GetIO();
      g_yaw   += io.MouseDelta.x * sensitivity;
      g_pitch += io.MouseDelta.y * sensitivity;

      const float pitch_limit = XMConvertToRadians(89.0f);
      g_pitch = std::clamp(g_pitch, -pitch_limit, pitch_limit);
    }

    {
      ImDrawList* dl = ImGui::GetWindowDrawList();
      const float mid_x = cx + cw / 2.0f;
      dl->AddLine({mid_x, cy}, {mid_x, cy + ch}, ImGui::GetColorU32(ImGuiCol_Separator), 2.0f);

      const char* left_name = g_gui_flat_view ? g_map_flat_source_to_name[g_gui_flat_left]
                                              : g_map_shader_to_name[g_gui_shader_left];
      const char* right_name = g_gui_flat_view ? g_map_flat_source_to_name[g_gui_flat_right]
                                               : g_map_shader_to_name[g_gui_shader_right];
      auto DrawCaption = [&](const char* text, float x0) {
        const float pad = 4.0f;
        ImVec2 size = ImGui::CalcTextSize(text);
        ImVec2 p0(x0 + 8.0f, cy + 8.0f);
        ImVec2 p1(p0.x + size.x + 2.0f * pad, p0.y + size.y + 2.0f * pad);
        dl->AddRectFilled(p0, p1, IM_COL32(0, 0, 0, 160), 3.0f);
        dl->AddText({p0.x + pad, p0.y + pad}, IM_COL32_WHITE, text);
      };
      DrawCaption(left_name, cx);
      DrawCaption(right_name, mid_x);
    }
    ImGui::End();
    ImGui::PopStyleVar();
  }

  auto command_allocator = g_commandallocators[g_frame_i];
  auto buffer = g_buffers[g_frame_i];
  command_allocator->Reset();
  g_commandlist->Reset(command_allocator.Get(), nullptr);
  D3D12_CPU_DESCRIPTOR_HANDLE rtv_handle = g_dhandle_rtv[g_frame_i].cpu_;
  D3D12_CPU_DESCRIPTOR_HANDLE dsv_handle = g_dhandle_dsv.cpu_;
  UINT tex_color_size = g_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

  D3D12_VIEWPORT viewport_left = {};
  viewport_left.TopLeftX = cx;
  viewport_left.TopLeftY = cy;
  viewport_left.Width = cw / 2.0f;
  viewport_left.Height = ch;
  viewport_left.MinDepth = 0.0f;
  viewport_left.MaxDepth = 1.0f;

  D3D12_VIEWPORT viewport_right = {};
  viewport_right.TopLeftX = cx + cw / 2.0f;
  viewport_right.TopLeftY = cy;
  viewport_right.Width = cw / 2.0f;
  viewport_right.Height = ch;
  viewport_right.MinDepth = 0.0f;
  viewport_right.MaxDepth = 1.0f;

  D3D12_RECT scissor = {};
  scissor.top = 0;
  scissor.left = 0;
  scissor.bottom = LONG_MAX;
  scissor.right = LONG_MAX;

  // Write directly to backbuffer if we're not using TAA

  const bool use_taa = g_gui_taa;
  D3D12_CPU_DESCRIPTOR_HANDLE scene_rtv = use_taa ? g_dhandle_frame_rtv.cpu_ : rtv_handle;

  {
    if (use_taa)
      TransitionIfRequired(g_commandlist, g_tex_frame, g_frame_state, D3D12_RESOURCE_STATE_RENDER_TARGET);
    else
      TransitionResource(g_commandlist, buffer, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET);
    g_commandlist->ClearRenderTargetView(scene_rtv, g_clear_color, 0, nullptr);
    g_commandlist->ClearDepthStencilView(dsv_handle, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
  }

  ID3D12DescriptorHeap* heaps[1] = {g_dalloc_srv.GetHeapUnsafe()};
  g_commandlist->SetDescriptorHeaps(_countof(heaps), heaps);
  g_commandlist->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
  g_commandlist->IASetVertexBuffers(0, 1, &g_vbv[g_gui_model]);
  g_commandlist->IASetIndexBuffer(&g_ibv[g_gui_model]);

  {
    LightingParams lp = {};
    lp.exposure_ = g_gui_exposure;
    lp.displacement_scale_ = g_gui_displacement_scale;
    lp.normal_scale_ = g_gui_normal_scale;
    lp.jitter_px_ = taa_jitter_px;

    // Using temporal noise without TAA results in jitter. Turn it off when TAA is off.
    static uint32_t noise_frame = 0;
    if (g_gui_taa)
      noise_frame = (noise_frame + 1) % 64;
    lp.noise_frame_ = static_cast<float>(noise_frame);
    for (int i = 0; i < 9; i++)
      lp.diffuse_sh_[i] = g_diffuse_sh[i];
    void* mapped = 0;
    D3D12_RANGE map_range = {0, 0};
    g_buffer_lighting_params->Map(0, &map_range, &mapped);
    memcpy(mapped, &lp, sizeof(lp));
    g_buffer_lighting_params->Unmap(0, nullptr);
  }

  if (g_gui_flat_view)
  {
    g_commandlist->RSSetViewports(1, &viewport_left);
    g_commandlist->RSSetScissorRects(1, &scissor);
    g_commandlist->OMSetRenderTargets(1, &scene_rtv, FALSE, &dsv_handle);
    DrawFlatPane(static_cast<FlatSource>(g_gui_flat_left), viewport_left);
    g_commandlist->RSSetViewports(1, &viewport_right);
    DrawFlatPane(static_cast<FlatSource>(g_gui_flat_right), viewport_right);
  }
  else
  {
  {
    g_commandlist->RSSetViewports(1, &viewport_left);
    g_commandlist->RSSetScissorRects(1, &scissor);
    g_commandlist->OMSetRenderTargets(1, &scene_rtv, FALSE, &dsv_handle);

    CubemapTransforms cubemap_transforms = {};
    cubemap_transforms.proj_to_view_ = XMMatrixInverse(nullptr, g_proj_mat);
    cubemap_transforms.view_to_world_ = XMMatrixInverse(nullptr, g_view_mat);
    g_commandlist->SetPipelineState(g_pipelinestate_cubemap.Get());
    g_commandlist->SetGraphicsRootSignature(g_rootsignature_cubemap.Get());
    g_commandlist->SetGraphicsRoot32BitConstants(0, sizeof(CubemapTransforms) / 4, &cubemap_transforms, 0);
    g_commandlist->SetGraphicsRootDescriptorTable(1, g_dhandle_ibl[0].gpu_);
    g_commandlist->SetGraphicsRootDescriptorTable(2, g_dhandle_lparams.gpu_);
    g_commandlist->DrawInstanced(6, 1, 0, 0);

    if (IsShaderAvailable(shader_left))
    {
      SetPipelineStateForShader(shader_left);
      SetDescriptorsForShader(shader_left);
      g_commandlist->DrawIndexedInstanced(g_map_model_to_index_count[g_gui_model], 1, 0, 0, 0);
    }
  }
  {
    g_commandlist->RSSetViewports(1, &viewport_right);
    g_commandlist->RSSetScissorRects(1, &scissor);
    g_commandlist->OMSetRenderTargets(1, &scene_rtv, FALSE, &dsv_handle);

    CubemapTransforms cubemap_transforms = {};
    cubemap_transforms.proj_to_view_ = XMMatrixInverse(nullptr, g_proj_mat);
    cubemap_transforms.view_to_world_ = XMMatrixInverse(nullptr, g_view_mat);
    g_commandlist->SetPipelineState(g_pipelinestate_cubemap.Get());
    g_commandlist->SetGraphicsRootSignature(g_rootsignature_cubemap.Get());
    g_commandlist->SetGraphicsRoot32BitConstants(0, sizeof(CubemapTransforms) / 4, &cubemap_transforms, 0);
    g_commandlist->SetGraphicsRootDescriptorTable(1, g_dhandle_ibl[0].gpu_);
    g_commandlist->SetGraphicsRootDescriptorTable(2, g_dhandle_lparams.gpu_);
    g_commandlist->DrawInstanced(6, 1, 0, 0);

    if (IsShaderAvailable(shader_right))
    {
      SetPipelineStateForShader(shader_right);
      SetDescriptorsForShader(shader_right);
      g_commandlist->DrawIndexedInstanced(g_map_model_to_index_count[g_gui_model], 1, 0, 0, 0);
    }
  }
  }

  if (use_taa)
  {
    const uint32_t w = g_taa_write_i;
    const uint32_t r = w ^ 1;

    if (!g_taa_history_valid)
    {
      g_prev_world_to_proj = view_proj_nojitter;
      TransitionIfRequired(g_commandlist, g_tex_frame, g_frame_state, D3D12_RESOURCE_STATE_COPY_SOURCE);
      TransitionIfRequired(g_commandlist, g_tex_taa_accum[r], g_taa_accum_state[r], D3D12_RESOURCE_STATE_COPY_DEST);
      g_commandlist->CopyResource(g_tex_taa_accum[r].Get(), g_tex_frame.Get());
      g_taa_history_valid = true;
    }

    TransitionIfRequired(g_commandlist, g_tex_frame, g_frame_state, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    TransitionIfRequired(
      g_commandlist,
      g_tex_taa_accum[r],
      g_taa_accum_state[r],
      D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    TransitionIfRequired(g_commandlist, g_tex_taa_accum[w], g_taa_accum_state[w], D3D12_RESOURCE_STATE_RENDER_TARGET);
    TransitionResource(
      g_commandlist,
      g_depthbuffer,
      D3D12_RESOURCE_STATE_DEPTH_WRITE,
      D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    TransitionIfRequired(
      g_commandlist,
      g_tex_depth_prev,
      g_depth_prev_state,
      D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);

    TaaConstants tc = {};
    tc.proj_to_world_unjittered_ = XMMatrixInverse(nullptr, view_proj_nojitter);
    tc.prev_world_to_proj_unjittered_ = g_gui_flat_view ? view_proj_nojitter : g_prev_world_to_proj;
    XMStoreFloat3(&tc.eye_, eye_pos);
    tc.pane_origin_ = XMFLOAT2(cx, cy);
    tc.pane_dim_ = XMFLOAT2(cw / 2.0f, ch);

    D3D12_VIEWPORT viewport_full = {0.0f, 0.0f, static_cast<FLOAT>(g_width), static_cast<FLOAT>(g_height), 0.0f, 1.0f};
    g_commandlist->RSSetViewports(1, &viewport_full);
    g_commandlist->RSSetScissorRects(1, &scissor);
    g_commandlist->OMSetRenderTargets(1, &g_dhandle_taa_rtv[w].cpu_, FALSE, nullptr);
    g_commandlist->SetPipelineState(g_pipelinestate_taa.Get());
    g_commandlist->SetGraphicsRootSignature(g_rootsignature_taa.Get());
    g_commandlist->SetGraphicsRoot32BitConstants(0, sizeof(TaaConstants) / 4, &tc, 0);
    g_commandlist->SetGraphicsRootDescriptorTable(1, g_dhandle_taa_srv[r].gpu_);
    g_commandlist->SetGraphicsRootDescriptorTable(2, g_dhandle_frame_srv.gpu_);
    g_commandlist->SetGraphicsRootDescriptorTable(3, g_dhandle_depth_srv.gpu_);
    g_commandlist->SetGraphicsRootDescriptorTable(4, g_dhandle_depth_prev_srv.gpu_);
    g_commandlist->DrawInstanced(6, 1, 0, 0);

    TransitionResource(
      g_commandlist,
      g_depthbuffer,
      D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
      D3D12_RESOURCE_STATE_COPY_SOURCE);
    TransitionIfRequired(g_commandlist, g_tex_depth_prev, g_depth_prev_state, D3D12_RESOURCE_STATE_COPY_DEST);
    g_commandlist->CopyResource(g_tex_depth_prev.Get(), g_depthbuffer.Get());
    TransitionResource(
      g_commandlist,
      g_depthbuffer,
      D3D12_RESOURCE_STATE_COPY_SOURCE,
      D3D12_RESOURCE_STATE_DEPTH_WRITE);
    g_prev_world_to_proj = view_proj_nojitter;

    TransitionIfRequired(g_commandlist, g_tex_taa_accum[w], g_taa_accum_state[w], D3D12_RESOURCE_STATE_COPY_SOURCE);
    TransitionResource(g_commandlist, buffer, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_COPY_DEST);
    g_commandlist->CopyResource(buffer.Get(), g_tex_taa_accum[w].Get());
    TransitionResource(g_commandlist, buffer, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_RENDER_TARGET);
    g_commandlist->OMSetRenderTargets(1, &rtv_handle, FALSE, &dsv_handle);

    g_taa_write_i = r;
  }
  else
  {
    g_taa_history_valid = false;
  }

  ImGui::Render();
  ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), g_commandlist.Get());

  {
    TransitionResource(g_commandlist, buffer, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT);
    VERIFY(g_commandlist->Close());

    ID3D12CommandList* const command_lists[] = { g_commandlist.Get() };
    g_queue->ExecuteCommandLists(_countof(command_lists), command_lists);

    UINT sync_interval = g_vsync ? 1 : 0;
    UINT present_flags = (g_gsync && !g_vsync) ? DXGI_PRESENT_ALLOW_TEARING : 0;
    VERIFY(g_swapchain->Present(sync_interval, present_flags));
    g_framefenceval[g_frame_i] = SignalFence(g_queue, g_fence, &g_fenceval);

    g_frame_i = g_swapchain->GetCurrentBackBufferIndex();
    WaitForFenceValue(g_fence, g_framefenceval[g_frame_i], g_fence_event);
  }
}

void UpdateRenderTargetViews(
  ComPtr<ID3D12Device2> device,
  ComPtr<IDXGISwapChain4> swapchain,
  ComPtr<ID3D12DescriptorHeap> descriptor_heap)
{
  auto rtv_descriptor_size = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

  D3D12_CPU_DESCRIPTOR_HANDLE rtv_handle = descriptor_heap->GetCPUDescriptorHandleForHeapStart();

  for(int i = 0; i < g_numframes; i++)
  {
    ComPtr<ID3D12Resource> buffer;
    VERIFY(swapchain->GetBuffer(i, IID_PPV_ARGS(&buffer)));
    device->CreateRenderTargetView(buffer.Get(), nullptr, rtv_handle);
    g_buffers[i] = buffer;
    rtv_handle.ptr += rtv_descriptor_size;
  }

  g_descriptorsize = rtv_descriptor_size;
}

void Resize(uint32_t width, uint32_t height)
{
  if (g_width != width || g_height != height)
  {
    g_width = std::max(1u, width);
    g_height = std::max(1u, height);

    Flush(g_queue, g_fence, &g_fenceval, g_fence_event);

    for (int i = 0; i < g_numframes; i++)
    {
      g_buffers[i].Reset();
      g_framefenceval[i] = g_framefenceval[g_frame_i];
    }

    DXGI_SWAP_CHAIN_DESC swapchain_desc = {};
    VERIFY(g_swapchain->GetDesc(&swapchain_desc));
    VERIFY(g_swapchain->ResizeBuffers(
      g_numframes,
      g_width,
      g_height,
      swapchain_desc.BufferDesc.Format,
      swapchain_desc.Flags));
    g_frame_i = g_swapchain->GetCurrentBackBufferIndex();
    UpdateRenderTargetViews(g_device, g_swapchain, g_dalloc_rtv.GetHeapUnsafe());
    CreateTaaResources();
  }

  ResizeDepthBuffer(width, height);
}

void SetFullscreen(bool fullscreen)
{
  if (g_fullscreen != fullscreen)
  {
    g_fullscreen = fullscreen;

    if (g_fullscreen)
    {
      ::GetWindowRect(g_hwnd, &g_window_rect);
      UINT window_style =
        WS_OVERLAPPEDWINDOW & ~(WS_CAPTION | WS_SYSMENU | WS_THICKFRAME | WS_MINIMIZEBOX | WS_MAXIMIZEBOX);
      ::SetWindowLongA(g_hwnd, GWL_STYLE, window_style);

      HMONITOR hmonitor = ::MonitorFromWindow(g_hwnd, MONITOR_DEFAULTTONEAREST);
      MONITORINFOEXA monitor_info = {};
      monitor_info.cbSize = sizeof(MONITORINFOEXA);
      ::GetMonitorInfoA(hmonitor, &monitor_info);

      ::SetWindowPos(
        g_hwnd,
        HWND_TOP,
        monitor_info.rcMonitor.left,
        monitor_info.rcMonitor.top,
        monitor_info.rcMonitor.right - monitor_info.rcMonitor.left,
        monitor_info.rcMonitor.bottom - monitor_info.rcMonitor.top,
        SWP_FRAMECHANGED | SWP_NOACTIVATE);

      ::ShowWindow(g_hwnd, SW_MAXIMIZE);
    }
  }
}

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND, UINT, WPARAM, LPARAM);

LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

LRESULT CALLBACK WndProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
  if (ImGui_ImplWin32_WndProcHandler(hWnd, uMsg, wParam, lParam))
    return 1;

  if (g_initialized)
  {
    switch (uMsg)
    {
      case WM_SYSKEYDOWN:
      case WM_KEYDOWN:
      {
        bool alt = (::GetAsyncKeyState(VK_MENU) & 0x8000) != 0;

        switch (wParam)
        {
          case 'V':
            g_vsync = !g_vsync;
            break;
          case VK_ESCAPE:
            if (ConfirmDiscardUnsaved())
              ::PostQuitMessage(0);
            break;
          case VK_RETURN:
            if (alt)
            {
          case VK_F11:
              SetFullscreen(!g_fullscreen);
            }
            break;
        }
      }
      break;

      case WM_SYSCHAR:
      break;

      case WM_SIZE:
      {
        RECT client_rect = {};
        ::GetClientRect(g_hwnd, &client_rect);
        int width = client_rect.right - client_rect.left;
        int height = client_rect.bottom - client_rect.top;
        Resize(width, height);
      }
      break;

      case WM_CLOSE:
        if (ConfirmDiscardUnsaved())
          ::DestroyWindow(hWnd);
        return 0;

      case WM_DESTROY:
        ::PostQuitMessage(0);
        break;

      default:
        return ::DefWindowProcA(hWnd, uMsg, wParam, lParam);
    }
  }
  return ::DefWindowProcA(hWnd, uMsg, wParam, lParam);
}

void EnableDebugLayer()
{
#if defined(_DEBUG)
  ComPtr<ID3D12Debug> debug_interface;
  VERIFY(D3D12GetDebugInterface(IID_PPV_ARGS(&debug_interface)));
  debug_interface->EnableDebugLayer();
#endif
}

HWND CreateAndShowWindow(HINSTANCE hInstance, const char* class_name, const char* title)
{
  WNDCLASSEXA wndcls = {0};
  wndcls.cbSize = sizeof(WNDCLASSEXA);
  wndcls.style = CS_HREDRAW | CS_VREDRAW;
  wndcls.lpfnWndProc = &WndProc;
  wndcls.cbClsExtra = 0;
  wndcls.cbWndExtra = 0;
  wndcls.hInstance = hInstance;
  wndcls.hIcon = ::LoadIcon(hInstance, NULL);
  wndcls.hCursor = ::LoadCursor(NULL, IDC_ARROW);
  wndcls.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
  wndcls.lpszClassName = class_name;
  wndcls.hIconSm = ::LoadIcon(hInstance, NULL);

  static ATOM atom = ::RegisterClassExA(&wndcls);
  assert(atom > 0);

  HWND hWnd = ::CreateWindowExA(
    NULL,
    class_name,
    title,
    WS_OVERLAPPEDWINDOW,
    100,
    100,
    g_width,
    g_height,
    NULL,
    NULL,
    hInstance,
    NULL);

  assert(hWnd);

  return hWnd;
}

ComPtr<IDXGIAdapter4> GetAdapter()
{
  ComPtr<IDXGIFactory4> dxgiFactory;
  UINT create_factory_flags = 0;
#if defined(_DEBUG)
  create_factory_flags = DXGI_CREATE_FACTORY_DEBUG;
#endif
  VERIFY(CreateDXGIFactory2(create_factory_flags, IID_PPV_ARGS(&dxgiFactory)));

  ComPtr<IDXGIAdapter1> dxgi_adapter1;
  ComPtr<IDXGIAdapter4> dxgi_adapter4;
  SIZE_T max_dedicated_vram = 0;
  for (UINT i = 0; dxgiFactory->EnumAdapters1(i, &dxgi_adapter1) != DXGI_ERROR_NOT_FOUND; i++)
  {
    DXGI_ADAPTER_DESC1 dxgi_adapter_desc1;
    dxgi_adapter1->GetDesc1(&dxgi_adapter_desc1);

    if ((dxgi_adapter_desc1.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0 && SUCCEEDED(D3D12CreateDevice(
      dxgi_adapter1.Get(),
      D3D_FEATURE_LEVEL_12_2,
      __uuidof(ID3D12Device),
      nullptr)) && dxgi_adapter_desc1.DedicatedVideoMemory > max_dedicated_vram)
    {
      max_dedicated_vram = dxgi_adapter_desc1.DedicatedVideoMemory;
      VERIFY(dxgi_adapter1.As(&dxgi_adapter4));
    }
  }
  return dxgi_adapter4;
}

ComPtr<ID3D12Device2> CreateDevice(ComPtr<IDXGIAdapter4> adapter)
{
  char path[MAX_PATH];
  GetModuleFileNameA(GetModuleHandleA("D3D12Core.dll"), path, MAX_PATH);

  UUID experimental[1] = { D3D12ExperimentalShaderModels };
  HRESULT hr_exp = D3D12EnableExperimentalFeatures(_countof(experimental), experimental, nullptr, nullptr);

  ComPtr<ID3D12Device2> device2;
  VERIFY(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_2, IID_PPV_ARGS(&device2)));

#if defined(_DEBUG)
  ComPtr<ID3D12InfoQueue> info_queue;
  if (SUCCEEDED(device2.As(&info_queue)))
  {
    info_queue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_CORRUPTION, FALSE);
    info_queue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, FALSE);
    info_queue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_WARNING, FALSE);

    D3D12_MESSAGE_SEVERITY severities[] =
    {
      D3D12_MESSAGE_SEVERITY_INFO
    };

    D3D12_MESSAGE_ID deny_ids[] = {
      D3D12_MESSAGE_ID_CLEARRENDERTARGETVIEW_MISMATCHINGCLEARVALUE,
      D3D12_MESSAGE_ID_MAP_INVALID_NULLRANGE,
      D3D12_MESSAGE_ID_UNMAP_INVALID_NULLRANGE,
    };

    D3D12_INFO_QUEUE_FILTER new_filter = {};
    new_filter.DenyList.NumSeverities = _countof(severities);
    new_filter.DenyList.pSeverityList = severities;
    new_filter.DenyList.NumIDs = _countof(deny_ids);
    new_filter.DenyList.pIDList = deny_ids;

    VERIFY(info_queue->PushStorageFilter(&new_filter));
  }
#endif

#if OPENNTC_COOP
  D3D12_FEATURE_DATA_LINEAR_ALGEBRA_SUPPORT la = {};
  VERIFY(device2->CheckFeatureSupport(D3D12_FEATURE_LINEAR_ALGEBRA_SUPPORT, &la, sizeof(la)));
  assert(la.LinearAlgebraTier != D3D12_LINEAR_ALGEBRA_TIER_NOT_SUPPORTED);

  D3D12_FEATURE_DATA_LINEAR_ALGEBRA_MATRIX_OPERATION_SUPPORT op = {};
  op.OperationType = D3D12_LINEAR_ALGEBRA_OPERATION_TYPE_THREAD_VECTOR_MATRIX_MULTIPLY;
  op.ThreadVectorMatrixMultiply.VectorInputType = D3D12_LINEAR_ALGEBRA_DATATYPE_SINT8;
  op.ThreadVectorMatrixMultiply.MatrixInputType = D3D12_LINEAR_ALGEBRA_DATATYPE_SINT8;
  op.ThreadVectorMatrixMultiply.BiasInputType = D3D12_LINEAR_ALGEBRA_DATATYPE_SINT32;
  op.ThreadVectorMatrixMultiply.VectorResultType = D3D12_LINEAR_ALGEBRA_DATATYPE_SINT32;
  VERIFY(device2->CheckFeatureSupport(
    D3D12_FEATURE_LINEAR_ALGEBRA_LINEAR_ALGEBRA_MATRIX_OPERATION_SUPPORT,
    &op,
    sizeof(op)));

  D3D12_LINEAR_ALGEBRA_MULTIPLICATION_SUPPORT_FLAGS flags = op.ThreadVectorMatrixMultiply.SupportFlags;
  bool native_support = flags & D3D12_LINEAR_ALGEBRA_MULTIPLICATION_SUPPORT_FLAG_SUPPORTED;
  bool emu_input = flags & D3D12_LINEAR_ALGEBRA_MULTIPLICATION_SUPPORT_FLAG_EMULATED_INPUTS;
  bool emu_output = flags & D3D12_LINEAR_ALGEBRA_MULTIPLICATION_SUPPORT_FLAG_EMULATED_OUTPUTS;
#endif

  return device2;
}

ComPtr<ID3D12CommandQueue> CreateCommandQueue(ComPtr<ID3D12Device2> device, D3D12_COMMAND_LIST_TYPE type)
{
  ComPtr<ID3D12CommandQueue> command_queue;

  D3D12_COMMAND_QUEUE_DESC desc = {};
  desc.Type = type;
  desc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
  desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
  desc.NodeMask = 0;

  VERIFY(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&command_queue)));
  return command_queue;
}

bool CheckTearingSupport()
{
  BOOL allow_tearing = FALSE;

  ComPtr<IDXGIFactory4> factory4;
  if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory4))))
  {
    ComPtr<IDXGIFactory5> factory5;
    if (SUCCEEDED(factory4.As(&factory5)))
    {
      if (FAILED(factory5->CheckFeatureSupport(
        DXGI_FEATURE_PRESENT_ALLOW_TEARING,
        &allow_tearing,
        sizeof(allow_tearing))))
      {
        allow_tearing = FALSE;
      }
    }
  }

  return (allow_tearing == TRUE);
}

ComPtr<IDXGISwapChain4> CreateSwapChain(
  HWND hwnd,
  ComPtr<ID3D12CommandQueue> command_queue,
  uint32_t width,
  uint32_t height,
  uint32_t buffer_count)
{
  ComPtr<IDXGISwapChain4> dxgi_swapchain4;
  ComPtr<IDXGIFactory4> dxgi_factory4;
  UINT create_factory_flags = 0;
#if defined(_DEBUG)
  create_factory_flags = DXGI_CREATE_FACTORY_DEBUG;
#endif
  VERIFY(CreateDXGIFactory2(create_factory_flags, IID_PPV_ARGS(&dxgi_factory4)));

  DXGI_SWAP_CHAIN_DESC1 swapchain_desc = {0};
  swapchain_desc.Width = width;
  swapchain_desc.Height = height;
  swapchain_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
  swapchain_desc.Stereo = FALSE;
  swapchain_desc.SampleDesc = {1, 0};
  swapchain_desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
  swapchain_desc.BufferCount = buffer_count;
  swapchain_desc.Scaling = DXGI_SCALING_STRETCH;
  swapchain_desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
  swapchain_desc.AlphaMode = DXGI_ALPHA_MODE_UNSPECIFIED;
  swapchain_desc.Flags = CheckTearingSupport() ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;

  ComPtr<IDXGISwapChain1> swapchain1;
  VERIFY(dxgi_factory4->CreateSwapChainForHwnd(
    command_queue.Get(),
    g_hwnd,
    &swapchain_desc,
    nullptr,
    nullptr,
    &swapchain1));
  VERIFY(dxgi_factory4->MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER));
  VERIFY(swapchain1.As(&dxgi_swapchain4));

  return dxgi_swapchain4;
}

ComPtr<ID3D12CommandAllocator> CreateCommandAllocator(ComPtr<ID3D12Device> device, D3D12_COMMAND_LIST_TYPE type)
{
  ComPtr<ID3D12CommandAllocator> command_allocator;
  VERIFY(device->CreateCommandAllocator(type, IID_PPV_ARGS(&command_allocator)));

  return command_allocator;
}

ComPtr<ID3D12GraphicsCommandList10> CreateCommandList(
  ComPtr<ID3D12Device2> device,
  ComPtr<ID3D12CommandAllocator> command_allocator,
  D3D12_COMMAND_LIST_TYPE type)
{
  ComPtr<ID3D12GraphicsCommandList> command_list;
  VERIFY(device->CreateCommandList(0, type, command_allocator.Get(), nullptr, IID_PPV_ARGS(&command_list)));
  VERIFY(command_list->Close());

  ComPtr<ID3D12GraphicsCommandList10> command_list10;
  VERIFY(command_list.As(&command_list10));

  return command_list10;
}

ComPtr<ID3D12Fence> CreateFence(ComPtr<ID3D12Device2> device)
{
  ComPtr<ID3D12Fence> fence;
  VERIFY(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
  return fence;
}

HANDLE CreateEventHandle()
{
  HANDLE fence_event;
  fence_event = ::CreateEventA(nullptr, FALSE, FALSE, NULL);
  assert(fence_event);
  return fence_event;
}

// Style based off "Catppucin Mocha"
// From https://github.com/ocornut/imgui/issues/707#issuecomment-4107169777
//  With some tweaks for our case

static void SetupImGuiCatppuccinMochaStyle()
{
  ImGui::StyleColorsDark();

  ImGuiStyle& style = ImGui::GetStyle();
  ImVec4* colors = style.Colors;

  style.WindowPadding = ImVec2(12.0f, 12.0f);
  style.FramePadding = ImVec2(6.0f, 4.0f);
  style.ItemSpacing = ImVec2(8.0f, 6.0f);
  style.ScrollbarSize = 14.0f;
  style.GrabMinSize = 12.0f;

  // Removing rounded window corners to avoid exposing clear color

  style.WindowRounding = 0.0f;
  style.FrameRounding = 5.0f;
  style.PopupRounding = 5.0f;
  style.ScrollbarRounding = 12.0f;
  style.GrabRounding = 5.0f;
  style.TabRounding = 5.0f;

  style.WindowBorderSize = 1.0f;
  style.FrameBorderSize = 0.0f;
  style.PopupBorderSize = 1.0f;

  // Text
  colors[ImGuiCol_Text] = ImVec4(0.80f, 0.84f, 0.96f, 1.00f);          // Text
  colors[ImGuiCol_TextDisabled] = ImVec4(0.42f, 0.45f, 0.55f, 1.00f);  // Surface1

  // Backgrounds
  colors[ImGuiCol_WindowBg] = ImVec4(0.12f, 0.12f, 0.18f, 1.00f);  // Base
  colors[ImGuiCol_ChildBg] = ImVec4(0.09f, 0.09f, 0.15f, 1.00f);   // Mantle
  colors[ImGuiCol_PopupBg] = ImVec4(0.07f, 0.07f, 0.11f, 0.96f);   // Crust

  // Borders
  colors[ImGuiCol_Border] = ImVec4(0.19f, 0.20f, 0.27f, 1.00f);  // Surface0
  colors[ImGuiCol_BorderShadow] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);

  // Frames
  colors[ImGuiCol_FrameBg] = ImVec4(0.19f, 0.20f, 0.27f, 1.00f);         // Surface0
  colors[ImGuiCol_FrameBgHovered] = ImVec4(0.25f, 0.26f, 0.35f, 1.00f);  // Surface1
  colors[ImGuiCol_FrameBgActive] = ImVec4(0.31f, 0.32f, 0.42f, 1.00f);   // Surface2

  // Title bars
  colors[ImGuiCol_TitleBg] = ImVec4(0.09f, 0.09f, 0.15f, 1.00f);           // Mantle
  colors[ImGuiCol_TitleBgActive] = ImVec4(0.12f, 0.12f, 0.18f, 1.00f);     // Base
  colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.07f, 0.07f, 0.11f, 1.00f);  // Crust

  // Menus
  colors[ImGuiCol_MenuBarBg] = ImVec4(0.09f, 0.09f, 0.15f, 1.00f);

  // Scrollbars
  colors[ImGuiCol_ScrollbarBg] = ImVec4(0.09f, 0.09f, 0.15f, 1.00f);
  colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.31f, 0.32f, 0.42f, 1.00f);  // Surface2
  colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.37f, 0.38f, 0.51f, 1.00f);
  colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.42f, 0.45f, 0.55f, 1.00f);

  // Interactables
  colors[ImGuiCol_CheckMark] = ImVec4(0.71f, 0.75f, 1.00f, 1.00f);   // Lavender
  colors[ImGuiCol_SliderGrab] = ImVec4(0.45f, 0.78f, 0.93f, 1.00f);  // Sapphire
  colors[ImGuiCol_SliderGrabActive] = ImVec4(0.45f, 0.78f, 0.93f, 1.00f);
  colors[ImGuiCol_Button] = ImVec4(0.19f, 0.20f, 0.27f, 1.00f);
  colors[ImGuiCol_ButtonHovered] = ImVec4(0.80f, 0.65f, 0.97f, 1.00f);  // Mauve
  colors[ImGuiCol_ButtonActive] = ImVec4(0.70f, 0.55f, 0.87f, 1.00f);
  colors[ImGuiCol_Header] = ImVec4(0.19f, 0.20f, 0.27f, 1.00f);
  colors[ImGuiCol_HeaderHovered] = ImVec4(0.25f, 0.26f, 0.35f, 1.00f);
  colors[ImGuiCol_HeaderActive] = ImVec4(0.31f, 0.32f, 0.42f, 1.00f);

  // Separators + Pane Divider
  colors[ImGuiCol_Separator] = ImVec4(0.25f, 0.26f, 0.35f, 1.00f);  // Surface1
  colors[ImGuiCol_SeparatorHovered] = ImVec4(0.71f, 0.75f, 1.00f, 1.00f);
  colors[ImGuiCol_SeparatorActive] = ImVec4(0.71f, 0.75f, 1.00f, 1.00f);

  // Tabs
  colors[ImGuiCol_Tab] = ImVec4(0.12f, 0.12f, 0.18f, 1.00f);
  colors[ImGuiCol_TabHovered] = ImVec4(0.31f, 0.32f, 0.42f, 1.00f);
  colors[ImGuiCol_TabSelected] = ImVec4(0.19f, 0.20f, 0.27f, 1.00f);
  colors[ImGuiCol_TabDimmed] = ImVec4(0.09f, 0.09f, 0.15f, 1.00f);
  colors[ImGuiCol_TabDimmedSelected] = ImVec4(0.12f, 0.12f, 0.18f, 1.00f);

  // Misc
  colors[ImGuiCol_PlotLines] = ImVec4(0.94f, 0.72f, 0.42f, 1.00f);  // Marigold
  colors[ImGuiCol_PlotHistogram] = ImVec4(0.45f, 0.78f, 0.93f, 1.00f);  // Sapphire (progress bar)
  colors[ImGuiCol_TextSelectedBg] = ImVec4(0.31f, 0.32f, 0.42f, 1.00f);
  colors[ImGuiCol_NavCursor] = ImVec4(0.71f, 0.75f, 1.00f, 1.00f);  // Lavender
}

int CALLBACK wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PWSTR lpCmdLine, int nCmdShow)
{
  {
    SharedContext::Access access = g_ctx.Acquire();
    SharedFields fields = g_shared_fields.load(std::memory_order_seq_cst);
    openntc::ContextInitInfo init_info = {};
    init_info.profile_ = openntc::Profile::Bpp_0_2;
    access.ctx_.Init(init_info);
    fields.train_in_progress_ = false;
    fields.train_complete_ = false;
    fields.train_steps_ = 0;
    fields.train_total_steps_ = 0;

    g_shared_fields.store(fields, std::memory_order_seq_cst);
  }

  ::CoInitializeEx(nullptr, COINIT_MULTITHREADED);
  SetThreadDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);

  const char* wndclass_name = "openntc-gui-window-class";

  if (!g_datapack.Load(GetExeRelativePath(L"data.bin")))
  {
    ::MessageBoxA(nullptr, "data.bin not found! Download from latest release or rebuild", "openntc-gui", MB_ICONERROR);
    return 1;
  }

  EnableDebugLayer();
  g_gsync = CheckTearingSupport();
  g_hwnd = CreateAndShowWindow(hInstance, wndclass_name, "openntc-gui");

  ::GetWindowRect(g_hwnd, &g_window_rect);

  ComPtr<IDXGIAdapter4> dxgi_adapter4 = GetAdapter();
  g_device = CreateDevice(dxgi_adapter4);

  // Descriptor Initialization

  {
    g_dalloc_srv.Init(g_device.Get(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, true, g_srv_count);
    
    for (int tex_i = 0; tex_i < openntc::kMaxSources; tex_i++)
      g_dhandle_tex[tex_i] = g_dalloc_srv.Allocate();
    for (int i = 0; i < 5; i++)
      g_dhandle_pbr_tex[i] = g_dalloc_srv.Allocate();
    g_dhandle_ntc_info = g_dalloc_srv.Allocate();
    for (int i = 0; i < 1 + 1 + 1; i++)
      g_dhandle_ntc_data[i] = g_dalloc_srv.Allocate();
    for (int i = 0; i < 2; i++)
      g_dhandle_ibl[i] = g_dalloc_srv.Allocate();
    g_dhandle_lparams = g_dalloc_srv.Allocate();
    g_dhandle_frame_srv = g_dalloc_srv.Allocate();
    for (int i = 0; i < 2; i++)
      g_dhandle_taa_srv[i] = g_dalloc_srv.Allocate();
    g_dhandle_depth_srv = g_dalloc_srv.Allocate();
    g_dhandle_depth_prev_srv = g_dalloc_srv.Allocate();
  }

  for (int i = 0; i < g_imgui_srv_count; i++)
  {
    UINT slot = g_nonimgui_srv_count + i;
    (void) g_dalloc_srv.Allocate();
    g_imgui_available_srv_slots.push_back(slot);
  }

  {
    g_dalloc_rtv.Init(g_device.Get(), D3D12_DESCRIPTOR_HEAP_TYPE_RTV, false, g_numframes + 3);
    for (int frame_i = 0; frame_i < g_numframes; frame_i++)
      g_dhandle_rtv[frame_i] = g_dalloc_rtv.Allocate();
    g_dhandle_frame_rtv = g_dalloc_rtv.Allocate();
    for (int i = 0; i < 2; i++)
      g_dhandle_taa_rtv[i] = g_dalloc_rtv.Allocate();
  }

  {
    g_dalloc_dsv.Init(g_device.Get(), D3D12_DESCRIPTOR_HEAP_TYPE_DSV, false, 1);
    g_dhandle_dsv = g_dalloc_dsv.Allocate();
  }

  g_queue = CreateCommandQueue(g_device, D3D12_COMMAND_LIST_TYPE_DIRECT);
  g_swapchain = CreateSwapChain(g_hwnd, g_queue, g_width, g_height, g_numframes);
  g_frame_i = g_swapchain->GetCurrentBackBufferIndex();
  g_descriptorsize = g_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
  UpdateRenderTargetViews(g_device, g_swapchain, g_dalloc_rtv.GetHeapUnsafe());

  for (int i = 0; i < g_numframes; i++)
  {
    g_commandallocators[i] = CreateCommandAllocator(g_device, D3D12_COMMAND_LIST_TYPE_DIRECT);
  }
  g_commandlist = CreateCommandList(g_device, g_commandallocators[g_frame_i], D3D12_COMMAND_LIST_TYPE_DIRECT);

  g_fence = CreateFence(g_device);
  g_fence_event = CreateEventHandle();

  g_initialized = true;

  LoadContent();

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  SetupImGuiCatppuccinMochaStyle();
  ImGui_ImplWin32_Init(g_hwnd);

  ImGui_ImplDX12_InitInfo imgui_info = {};
  imgui_info.Device = g_device.Get();
  imgui_info.CommandQueue = g_queue.Get();
  imgui_info.NumFramesInFlight = g_numframes;
  imgui_info.DSVFormat = DXGI_FORMAT_D32_FLOAT;
  imgui_info.RTVFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
  imgui_info.SrvDescriptorHeap = g_dalloc_srv.GetHeapUnsafe();
  imgui_info.SrvDescriptorAllocFn = ImguiDescriptorSrvAlloc;
  imgui_info.SrvDescriptorFreeFn = ImguiDescriptorSrvFree;
  ImGui_ImplDX12_Init(&imgui_info);

  ::ShowWindow(g_hwnd, SW_SHOW);
  MSG msg = {};
  while (msg.message != WM_QUIT)
  {
    if (::PeekMessageA(&msg, NULL, 0, 0, PM_REMOVE))
    {
      ::TranslateMessage(&msg);
      ::DispatchMessage(&msg);
    }
    Update();
    Render();
  }

  Flush(g_queue, g_fence, &g_fenceval, g_fence_event);
  ImGui_ImplDX12_Shutdown();
  ImGui_ImplWin32_Shutdown();
  ImGui::DestroyContext();
  ::CloseHandle(g_fence_event);
  g_stop_training.store(true, std::memory_order_seq_cst);
  if (g_train_job.valid())
    g_train_job.wait();

  return 0;
}