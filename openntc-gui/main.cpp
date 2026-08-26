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
#include <DirectXTex/DirectXTex.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <thread>
#include <mutex>
#include <future>

#include <imgui.h>
#include <imgui_impl_win32.h>
#include <imgui_impl_dx12.h>

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
ComPtr<ID3D12PipelineState> g_pipelinestate_ggx;
ComPtr<ID3D12RootSignature> g_rootsignature_ggx;
ComPtr<ID3D12PipelineState> g_pipelinestate_pbr_ntc;
ComPtr<ID3D12RootSignature> g_rootsignature_pbr_ntc;
ComPtr<ID3D12PipelineState> g_pipelinestate_pbr_ntc_coop;
ComPtr<ID3D12RootSignature> g_rootsignature_pbr_ntc_coop;
ComPtr<ID3D12PipelineState> g_pipelinestate_cubemap;
ComPtr<ID3D12RootSignature> g_rootsignature_cubemap;
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
bool g_compressed_data_loaded = false;
std::atomic<SharedFields> g_shared_fields;
openntc::FileData g_fil_data;

DescriptorAllocator g_dalloc_rtv;
DescriptorHandle g_dhandle_rtv[g_numframes];

DescriptorAllocator g_dalloc_dsv;
DescriptorHandle g_dhandle_dsv;

DescriptorAllocator g_dalloc_srv;
DescriptorHandle g_dhandle_tex[openntc::kMaxSources];
DescriptorHandle g_dhandle_ntc_info;
DescriptorHandle g_dhandle_ntc_data[1 + 1 + 3 + 3];
DescriptorHandle g_dhandle_lparams;
DescriptorHandle g_dhandle_ibl[2];

ComPtr<ID3D12Resource> g_buffer_ntc_info;
ComPtr<ID3D12Resource> g_buffer_g0;
ComPtr<ID3D12Resource> g_buffer_g1;
ComPtr<ID3D12Resource> g_buffer_W0;
ComPtr<ID3D12Resource> g_buffer_W1;
ComPtr<ID3D12Resource> g_buffer_Wout;
ComPtr<ID3D12Resource> g_buffer_W0_scale;
ComPtr<ID3D12Resource> g_buffer_W1_scale;
ComPtr<ID3D12Resource> g_buffer_Wout_scale;
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
  XMMATRIX model_to_world;
  XMMATRIX world_to_view;
  XMMATRIX view_to_proj;
};

struct NTCInfo
{
  int g0_grid_dim[8];
  int g1_grid_dim[8];
  uint32_t g0_offset[8];
  uint32_t g1_offset[8];
  int g0_bytes_per_channel;
  int g1_bytes_per_channel;
  int g0_channels;
  int g1_channels;
  int dim;
  int mip_count;
  float rcp_s_a1;
  float rcp_s_a2;
};

struct CubemapTransforms
{
  XMMATRIX proj_to_view;
  XMMATRIX view_to_world;
};

struct LightingParams
{
  float exposure;
  float displacement_scale;
  float normal_scale;
  float pad0;
  XMFLOAT3A diffuse_sh[9];
};

D3D12_VERTEX_BUFFER_VIEW g_vbv[g_kModelCount];
D3D12_INDEX_BUFFER_VIEW g_ibv[g_kModelCount];

uint32_t g_map_model_to_index_count[g_kModelCount];

XMMATRIX g_model_mat, g_view_mat, g_proj_mat;
float g_pitch = 0.0f;
float g_yaw = 0.0f;
float y_roll = 0.0f;
float g_fov_y = 65.0f;

constexpr int32_t g_nonimgui_srv_count = openntc::kMaxSources + 1 + (1 + 1 + 3 + 3) + 3;
constexpr int32_t g_imgui_srv_count = 64;
constexpr int32_t g_srv_count = g_nonimgui_srv_count + g_imgui_srv_count;

void PerformTrainingJob()
{
  openntc::TrainInfo train_info = {};
  train_info.grids_per_batch_ = 1;
  train_info.batch_count_ = 30000;
  SharedFields fields = g_shared_fields.load(std::memory_order_seq_cst);
  fields.train_complete = false;
  fields.train_in_progress = true;
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
      fields.eval_psnr = eval_results.psnr;
      fields.eval_mse = eval_results.mse;
      fields.train_in_progress = false;
      fields.train_complete = true;
      g_shared_fields.store(fields);
      break;
    }
    else
    {
      fields.train_steps = tprogress.batches_complete_;
      fields.train_total_steps = tprogress.total_batches_;
      g_shared_fields.store(fields);
    }
  }
}

std::vector<UINT> g_imgui_available_srv_slots;

enum class Shader: int32_t
{
  Flat,
  GGX,
  PBR_NTC,
  PBR_NTC_COOP,

  Count,
};

const char* g_map_shader_to_name[] = {
  "Flat",
  "GGX",
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

int32_t g_gui_shader_left = static_cast<int32_t>(Shader::GGX);
int32_t g_gui_shader_right = static_cast<int32_t>(Shader::GGX);
int32_t g_gui_texture = 0;
int32_t g_gui_camera_mode = static_cast<int32_t>(CameraMode::Orbit);
float g_gui_displacement_scale = 0.01f;
float g_gui_normal_scale = 1.0f;
float g_gui_exposure = 1.0f;
bool g_gui_spin = true;
int32_t g_gui_model = 0;

static inline UINT64 RoundUpTo(UINT64 a, UINT64 b)
{
  return ((a + b - 1) / b) * b;
}

void ImguiDescriptorSrvAlloc(ImGui_ImplDX12_InitInfo* init_info, D3D12_CPU_DESCRIPTOR_HANDLE* cpu_handle, D3D12_GPU_DESCRIPTOR_HANDLE* gpu_handle)
{
  UINT slot = g_imgui_available_srv_slots.back();
  g_imgui_available_srv_slots.pop_back();
  DescriptorHandle dhandle = g_dalloc_srv.At(slot);
  *cpu_handle = dhandle.cpu;
  *gpu_handle = dhandle.gpu;
}

void ImguiDescriptorSrvFree(ImGui_ImplDX12_InitInfo* init_info, D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle, D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle)
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

void WaitForFenceValue(ComPtr<ID3D12Fence> fence, uint64_t fenceval, HANDLE fenceevent, std::chrono::milliseconds duration_ms = std::chrono::milliseconds::max())
{
  if (fence->GetCompletedValue() < fenceval)
  {
    VERIFY(fence->SetEventOnCompletion(fenceval, fenceevent));
    ::WaitForSingleObject(fenceevent, static_cast<DWORD>(duration_ms.count()));
  }
}

void TransitionResource(ComPtr<ID3D12GraphicsCommandList10> command_list, ComPtr<ID3D12Resource> res, D3D12_RESOURCE_STATES initial_state, D3D12_RESOURCE_STATES final_state)
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

  VERIFY(g_device->CreateCommittedResource(&heap_props, D3D12_HEAP_FLAG_NONE, &res_desc, D3D12_RESOURCE_STATE_DEPTH_WRITE, &optimized_clear_val, IID_PPV_ARGS(&g_depthbuffer)));

  D3D12_DEPTH_STENCIL_VIEW_DESC dsv_desc = {};
  dsv_desc.Format = DXGI_FORMAT_D32_FLOAT;
  dsv_desc.Flags = D3D12_DSV_FLAG_NONE;
  dsv_desc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
  dsv_desc.Texture2D.MipSlice = 0;

  g_device->CreateDepthStencilView(g_depthbuffer.Get(), &dsv_desc, g_dhandle_dsv.cpu);
}

static void RebuildTextureResources(SharedContext::Access& access)
{
  Flush(g_queue, g_fence, &g_fenceval, g_fence_event);
  openntc::TextureData tex_data = access.ctx_.GetTextureData();

  for (int tex_i = 0; tex_i < tex_data.tex_count_; tex_i++)
  {
    g_formats[tex_i] = (tex_data.channels_[tex_i] == 1) ?
                        DXGI_FORMAT_R8_UNORM :
                        DXGI_FORMAT_R8G8B8A8_UNORM;

    g_tex[tex_i] = CreateTexture2D(g_device.Get(), D3D12_HEAP_TYPE_GPU_UPLOAD, access.ctx_.GetMipDim(0), 1, tex_data.mip_count_, g_formats[tex_i]);

    D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
    srv_desc.Format = g_formats[tex_i];
    srv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv_desc.Texture2D.MipLevels = tex_data.mip_count_;
    srv_desc.Texture2D.MostDetailedMip = 0;
    srv_desc.Texture2D.PlaneSlice = 0;
    srv_desc.Texture2D.ResourceMinLODClamp = 0.0f;
    srv_desc.Shader4ComponentMapping = D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_0, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_1, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_2, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_3);

    g_device->CreateShaderResourceView(g_tex[tex_i].Get(), &srv_desc, g_dhandle_tex[tex_i].cpu);
    g_compressed_data_loaded = false;
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
          memcpy(static_cast<char*>(mapped) + y * mip_row_padded + mip_off, tex_data.mips_[tex_i][mip_i] + y * mip_dim * tex_data.channels_[tex_i], mip_row);
      }
      else
      {
        for (int y = 0; y < mip_dim; y++)
        {
          for (int x = 0; x < mip_dim; x++)
          {
            memcpy(
              static_cast<char*>(mapped) + y * mip_row_padded + mip_off + x * channels_padded,
              tex_data.mips_[tex_i][mip_i] + y * mip_dim * tex_data.channels_[tex_i] + x * tex_data.channels_[tex_i], 3 * sizeof(uint8_t));
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
    rbar.Transition.StateAfter = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    rbar.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    g_commandlist->ResourceBarrier(1, &rbar);

    g_buffer_scratch->Unmap(0, nullptr);
    g_commandlist->Close();
    ID3D12CommandList* lists[] = { g_commandlist.Get() };
    g_queue->ExecuteCommandLists(1, lists);
    Flush(g_queue, g_fence, &g_fenceval, g_fence_event);
  }
}

static void LoadDiffuseSH(const char* path)
{
  FILE* f = nullptr;
  fopen_s(&f, path, "r");
  VERIFY(f != nullptr);
  if (f == nullptr) return;
  int count = 0;
  char line[256];
  while (count < 9 && fgets(line, sizeof(line), f))
  {
    float x, y, z;
    if (sscanf_s(line, " ( %f , %f , %f", &x, &y, &z) == 3)
    {
      g_diffuse_sh[count] = XMFLOAT3A(x, y, z);
      count++;
    }
  }
  fclose(f);
  VERIFY(count == 9);
}

static void LoadIBL()
{
  LoadDiffuseSH("C:/Code/openntc/img/ibl/baked/sh.txt");
  {
    ScratchImage img;
    LoadFromDDSFile(L"C:/Code/openntc/img/ibl/baked/specular_cube.dds", DDS_FLAGS_NONE, nullptr, img);

    g_tex_specular_ibl = CreateTexture2D(g_device.Get(), D3D12_HEAP_TYPE_GPU_UPLOAD, 256, 6, 5, DXGI_FORMAT_R16G16B16A16_FLOAT);

    D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
    srv_desc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    srv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
    srv_desc.TextureCube.MipLevels = 5;
    srv_desc.TextureCube.MostDetailedMip = 0;
    srv_desc.TextureCube.ResourceMinLODClamp = 0.0f;
    srv_desc.Shader4ComponentMapping = D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_0, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_1, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_2, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_3);

    g_device->CreateShaderResourceView(g_tex_specular_ibl.Get(), &srv_desc, g_dhandle_ibl[0].cpu);

    for (int face_i = 0; face_i < 6; face_i++)
    {
      for (int mip_i = 0; mip_i < 5; mip_i++)
      {
        const Image* subimg = img.GetImage(mip_i, face_i, 0);
        VERIFY(g_tex_specular_ibl->WriteToSubresource(face_i * 5 + mip_i, nullptr, subimg->pixels, (UINT)subimg->rowPitch, (UINT)subimg->slicePitch));
      }
    }
  }

  {
    ScratchImage img;
    LoadFromDDSFile(L"C:/Code/openntc/img/ibl/baked/dfg.dds", DDS_FLAGS_NONE, nullptr, img);

    g_tex_specular_dfg = CreateTexture2D(g_device.Get(), D3D12_HEAP_TYPE_GPU_UPLOAD, 256, 1, 1, DXGI_FORMAT_R16G16B16A16_FLOAT);

    D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
    srv_desc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    srv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv_desc.Texture2D.MipLevels = 1;
    srv_desc.Texture2D.MostDetailedMip = 0;
    srv_desc.Texture2D.ResourceMinLODClamp = 0.0f;
    srv_desc.Texture2D.PlaneSlice = 0;
    srv_desc.Shader4ComponentMapping = D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_0, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_1, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_2, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_3);

    g_device->CreateShaderResourceView(g_tex_specular_dfg.Get(), &srv_desc, g_dhandle_ibl[1].cpu);

    const Image* pimg = img.GetImage(0, 0, 0);
    VERIFY(g_tex_specular_dfg->WriteToSubresource(0, nullptr, pimg->pixels, pimg->rowPitch, pimg->slicePitch));
  }

  {
    uint64_t cbv_size = RoundUpTo(sizeof(LightingParams), 256);

    g_buffer_lighting_params = CreateBuffer(g_device.Get(), D3D12_HEAP_TYPE_GPU_UPLOAD, cbv_size);

    D3D12_CONSTANT_BUFFER_VIEW_DESC cbv_desc = {};
    cbv_desc.BufferLocation = g_buffer_lighting_params->GetGPUVirtualAddress();
    cbv_desc.SizeInBytes = cbv_size;

    g_device->CreateConstantBufferView(&cbv_desc, g_dhandle_lparams.cpu);
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
    InitModel(static_cast<ModelType>(i), 200, model);
    g_map_model_to_index_count[i] = model.GetIndexCount();
    g_vertex_buffer[i] = CreateBufferWithData(g_device.Get(), D3D12_HEAP_TYPE_GPU_UPLOAD, sizeof(VertexDescriptor) * model.GetVertexCount(), sizeof(VertexDescriptor) * model.GetVertexCount(), model.GetVertices());
    g_index_buffer[i] = CreateBufferWithData(g_device.Get(), D3D12_HEAP_TYPE_GPU_UPLOAD, sizeof(uint32_t) * model.GetIndexCount(), sizeof(uint32_t) * model.GetIndexCount(), model.GetIndices());
    g_vbv[i].BufferLocation = g_vertex_buffer[i]->GetGPUVirtualAddress();
    g_vbv[i].SizeInBytes = sizeof(VertexDescriptor) * model.GetVertexCount();
    g_vbv[i].StrideInBytes = sizeof(VertexDescriptor);
    g_ibv[i].BufferLocation = g_index_buffer[i]->GetGPUVirtualAddress();
    g_ibv[i].SizeInBytes = sizeof(uint32_t) * model.GetIndexCount();
    g_ibv[i].Format = DXGI_FORMAT_R32_UINT;
  }

  // Flat

  {
    ComPtr<ID3DBlob> vertex_shader_blob;
    VERIFY(D3DReadFileToBlob(L"C:/Code/openntc/openntc-gui/flat_vs.cso", &vertex_shader_blob));

    ComPtr<ID3DBlob> pixel_shader_blob;
    VERIFY(D3DReadFileToBlob(L"C:/Code/openntc/openntc-gui/flat_ps.cso", &pixel_shader_blob));

    D3D12_INPUT_ELEMENT_DESC input_layout[] = {
      { "SV_Position", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
      { "NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
      { "TANGENT", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
      { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
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

    D3D12_DESCRIPTOR_RANGE1 drange = {};
    drange.RegisterSpace = 0;
    drange.BaseShaderRegister = 1;
    drange.NumDescriptors = 1;
    drange.OffsetInDescriptorsFromTableStart = 0;
    drange.Flags = D3D12_DESCRIPTOR_RANGE_FLAG_DATA_STATIC;
    drange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;

    D3D12_ROOT_PARAMETER1 root_parameters[2];
    root_parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    root_parameters[0].Constants.Num32BitValues = sizeof(XMMATRIX) / 4;
    root_parameters[0].Constants.RegisterSpace = 0;
    root_parameters[0].Constants.ShaderRegister = 0;
    root_parameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;
    root_parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    root_parameters[1].DescriptorTable.NumDescriptorRanges = 1;
    root_parameters[1].DescriptorTable.pDescriptorRanges = &drange;
    root_parameters[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

    D3D12_STATIC_SAMPLER_DESC sampler_desc = {};
    sampler_desc.AddressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    sampler_desc.AddressV = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    sampler_desc.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    sampler_desc.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    sampler_desc.MinLOD = 0.0f;
    sampler_desc.MaxLOD = D3D12_FLOAT32_MAX;
    sampler_desc.BorderColor = D3D12_STATIC_BORDER_COLOR_OPAQUE_BLACK;
    sampler_desc.ComparisonFunc = D3D12_COMPARISON_FUNC_NONE;
    sampler_desc.ShaderRegister = 0;
    sampler_desc.RegisterSpace = 0;
    sampler_desc.MaxAnisotropy = 16;
    sampler_desc.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

    D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_signature_desc = {};
    root_signature_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
    root_signature_desc.Desc_1_1.NumParameters = 2;
    root_signature_desc.Desc_1_1.pParameters = root_parameters;
    root_signature_desc.Desc_1_1.NumStaticSamplers = 1;
    root_signature_desc.Desc_1_1.pStaticSamplers = &sampler_desc;
    root_signature_desc.Desc_1_1.Flags = root_signature_flags;

    ComPtr<ID3DBlob> root_signature_blob;
    ComPtr<ID3DBlob> error_blob;
    VERIFY(D3D12SerializeVersionedRootSignature(&root_signature_desc, &root_signature_blob, &error_blob));

    VERIFY(g_device->CreateRootSignature(0, root_signature_blob->GetBufferPointer(), root_signature_blob->GetBufferSize(), IID_PPV_ARGS(&g_rootsignature_flat)));

    struct PipelineStream
    {
      D3D12_PIPELINE_STATE_SUBOBJECT_TYPE root_signature_type;
      ID3D12RootSignature* root_signature;
      D3D12_PIPELINE_STATE_SUBOBJECT_TYPE input_layout_type;
      D3D12_INPUT_LAYOUT_DESC input_layout;
      D3D12_PIPELINE_STATE_SUBOBJECT_TYPE primitive_topology_type;
      D3D12_PRIMITIVE_TOPOLOGY_TYPE primitive_topology;
      D3D12_PIPELINE_STATE_SUBOBJECT_TYPE vs_type;
      D3D12_SHADER_BYTECODE vs;
      D3D12_PIPELINE_STATE_SUBOBJECT_TYPE ps_type;
      D3D12_SHADER_BYTECODE ps;
      D3D12_PIPELINE_STATE_SUBOBJECT_TYPE dsv_type;
      DXGI_FORMAT dsv;
      D3D12_PIPELINE_STATE_SUBOBJECT_TYPE rtv_type;
      D3D12_RT_FORMAT_ARRAY rtv;
    } pipeline_stream;

    pipeline_stream.root_signature_type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE;
    pipeline_stream.root_signature = g_rootsignature_flat.Get();
    pipeline_stream.input_layout_type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_INPUT_LAYOUT;
    pipeline_stream.input_layout.NumElements = _countof(input_layout);
    pipeline_stream.input_layout.pInputElementDescs = input_layout;
    pipeline_stream.primitive_topology_type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY;
    pipeline_stream.primitive_topology = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pipeline_stream.vs_type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS;
    pipeline_stream.vs.BytecodeLength = vertex_shader_blob->GetBufferSize();
    pipeline_stream.vs.pShaderBytecode = vertex_shader_blob->GetBufferPointer();
    pipeline_stream.ps_type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS;
    pipeline_stream.ps.BytecodeLength = pixel_shader_blob->GetBufferSize();
    pipeline_stream.ps.pShaderBytecode = pixel_shader_blob->GetBufferPointer();
    pipeline_stream.dsv_type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT;
    pipeline_stream.dsv = DXGI_FORMAT_D32_FLOAT;
    pipeline_stream.rtv_type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS;
    pipeline_stream.rtv.NumRenderTargets = 1;
    pipeline_stream.rtv.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    for(int i = 1; i < 8; i++) pipeline_stream.rtv.RTFormats[i] = DXGI_FORMAT_UNKNOWN;

    D3D12_PIPELINE_STATE_STREAM_DESC pdesc = {};
    pdesc.SizeInBytes = sizeof(PipelineStream);
    pdesc.pPipelineStateSubobjectStream = &pipeline_stream;
    VERIFY(g_device->CreatePipelineState(&pdesc, IID_PPV_ARGS(&g_pipelinestate_flat)));
  }

  // GGX

  {
    ComPtr<ID3DBlob> vertex_shader_blob;
    VERIFY(D3DReadFileToBlob(L"C:/Code/openntc/openntc-gui/ggx_vs.cso", &vertex_shader_blob));

    ComPtr<ID3DBlob> pixel_shader_blob;
    VERIFY(D3DReadFileToBlob(L"C:/Code/openntc/openntc-gui/ggx_ps.cso", &pixel_shader_blob));

    D3D12_INPUT_ELEMENT_DESC input_layout[] = {
      { "SV_Position", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
      { "NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
      { "TANGENT", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
      { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
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

    D3D12_DESCRIPTOR_RANGE1 drange = {};
    drange.RegisterSpace = 0;
    drange.BaseShaderRegister = 1;
    drange.NumDescriptors = 5;
    drange.OffsetInDescriptorsFromTableStart = 0;
    drange.Flags = D3D12_DESCRIPTOR_RANGE_FLAG_DATA_STATIC;
    drange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;

    D3D12_DESCRIPTOR_RANGE1 drange_srv = {};
    drange_srv.RegisterSpace = 0;
    drange_srv.BaseShaderRegister = 6;
    drange_srv.NumDescriptors = 2;
    drange_srv.OffsetInDescriptorsFromTableStart = 0;
    drange_srv.Flags = D3D12_DESCRIPTOR_RANGE_FLAG_DATA_STATIC;
    drange_srv.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;

    D3D12_DESCRIPTOR_RANGE1 drange_cbv_light = {};
    drange_cbv_light.RegisterSpace = 0;
    drange_cbv_light.BaseShaderRegister = 1;
    drange_cbv_light.NumDescriptors = 1;
    drange_cbv_light.OffsetInDescriptorsFromTableStart = 0;
    drange_cbv_light.Flags = D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE;
    drange_cbv_light.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_CBV;

    D3D12_ROOT_PARAMETER1 root_parameters[4];
    root_parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    root_parameters[0].Constants.Num32BitValues = sizeof(ModelViewProjection) / 4;
    root_parameters[0].Constants.RegisterSpace = 0;
    root_parameters[0].Constants.ShaderRegister = 0;
    root_parameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    root_parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    root_parameters[1].DescriptorTable.NumDescriptorRanges = 1;
    root_parameters[1].DescriptorTable.pDescriptorRanges = &drange;
    root_parameters[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    root_parameters[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    root_parameters[2].DescriptorTable.NumDescriptorRanges = 1;
    root_parameters[2].DescriptorTable.pDescriptorRanges = &drange_srv;
    root_parameters[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    root_parameters[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    root_parameters[3].DescriptorTable.NumDescriptorRanges = 1;
    root_parameters[3].DescriptorTable.pDescriptorRanges = &drange_cbv_light;
    root_parameters[3].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_STATIC_SAMPLER_DESC sampler_desc[1];
    sampler_desc[0].AddressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    sampler_desc[0].AddressV = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    sampler_desc[0].AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    sampler_desc[0].Filter = D3D12_FILTER_ANISOTROPIC;
    sampler_desc[0].MinLOD = 0.0f;
    sampler_desc[0].MaxLOD = D3D12_FLOAT32_MAX;
    sampler_desc[0].BorderColor = D3D12_STATIC_BORDER_COLOR_OPAQUE_BLACK;
    sampler_desc[0].ComparisonFunc = D3D12_COMPARISON_FUNC_NONE;
    sampler_desc[0].ShaderRegister = 0;
    sampler_desc[0].RegisterSpace = 0;
    sampler_desc[0].MaxAnisotropy = 16;
    sampler_desc[0].MipLODBias = 0.0f;
    sampler_desc[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_signature_desc = {};
    root_signature_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
    root_signature_desc.Desc_1_1.NumParameters = 4;
    root_signature_desc.Desc_1_1.pParameters = root_parameters;
    root_signature_desc.Desc_1_1.NumStaticSamplers = 1;
    root_signature_desc.Desc_1_1.pStaticSamplers = sampler_desc;
    root_signature_desc.Desc_1_1.Flags = root_signature_flags;

    ComPtr<ID3DBlob> root_signature_blob;
    ComPtr<ID3DBlob> error_blob;
    VERIFY(D3D12SerializeVersionedRootSignature(&root_signature_desc, &root_signature_blob, &error_blob));

    VERIFY(g_device->CreateRootSignature(0, root_signature_blob->GetBufferPointer(), root_signature_blob->GetBufferSize(), IID_PPV_ARGS(&g_rootsignature_ggx)));

    struct PipelineStream
    {
      D3D12_PIPELINE_STATE_SUBOBJECT_TYPE root_signature_type;
      ID3D12RootSignature* root_signature;
      D3D12_PIPELINE_STATE_SUBOBJECT_TYPE input_layout_type;
      D3D12_INPUT_LAYOUT_DESC input_layout;
      D3D12_PIPELINE_STATE_SUBOBJECT_TYPE primitive_topology_type;
      D3D12_PRIMITIVE_TOPOLOGY_TYPE primitive_topology;
      D3D12_PIPELINE_STATE_SUBOBJECT_TYPE vs_type;
      D3D12_SHADER_BYTECODE vs;
      D3D12_PIPELINE_STATE_SUBOBJECT_TYPE ps_type;
      D3D12_SHADER_BYTECODE ps;
      D3D12_PIPELINE_STATE_SUBOBJECT_TYPE dsv_type;
      DXGI_FORMAT dsv;
      D3D12_PIPELINE_STATE_SUBOBJECT_TYPE rtv_type;
      D3D12_RT_FORMAT_ARRAY rtv;
    } pipeline_stream;

    pipeline_stream.root_signature_type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE;
    pipeline_stream.root_signature = g_rootsignature_ggx.Get();
    pipeline_stream.input_layout_type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_INPUT_LAYOUT;
    pipeline_stream.input_layout.NumElements = _countof(input_layout);
    pipeline_stream.input_layout.pInputElementDescs = input_layout;
    pipeline_stream.primitive_topology_type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY;
    pipeline_stream.primitive_topology = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pipeline_stream.vs_type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS;
    pipeline_stream.vs.BytecodeLength = vertex_shader_blob->GetBufferSize();
    pipeline_stream.vs.pShaderBytecode = vertex_shader_blob->GetBufferPointer();
    pipeline_stream.ps_type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS;
    pipeline_stream.ps.BytecodeLength = pixel_shader_blob->GetBufferSize();
    pipeline_stream.ps.pShaderBytecode = pixel_shader_blob->GetBufferPointer();
    pipeline_stream.dsv_type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT;
    pipeline_stream.dsv = DXGI_FORMAT_D32_FLOAT;
    pipeline_stream.rtv_type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS;
    pipeline_stream.rtv.NumRenderTargets = 1;
    pipeline_stream.rtv.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    for(int i = 1; i < 8; i++) pipeline_stream.rtv.RTFormats[i] = DXGI_FORMAT_UNKNOWN;

    D3D12_PIPELINE_STATE_STREAM_DESC pdesc = {};
    pdesc.SizeInBytes = sizeof(PipelineStream);
    pdesc.pPipelineStateSubobjectStream = &pipeline_stream;
    VERIFY(g_device->CreatePipelineState(&pdesc, IID_PPV_ARGS(&g_pipelinestate_ggx)));
  }

  // PBR NTC

  {
    ComPtr<ID3DBlob> vertex_shader_blob;
    VERIFY(D3DReadFileToBlob(L"C:/Code/openntc/openntc-gui/pbr_ntc_vs.cso", &vertex_shader_blob));

    ComPtr<ID3DBlob> pixel_shader_blob;
    VERIFY(D3DReadFileToBlob(L"C:/Code/openntc/openntc-gui/pbr_ntc_ps.cso", &pixel_shader_blob));

    D3D12_INPUT_ELEMENT_DESC input_layout[] = {
      { "SV_Position", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
      { "NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
      { "TANGENT", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
      { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
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

    D3D12_DESCRIPTOR_RANGE1 drange_cbv = {};
    drange_cbv.RegisterSpace = 0;
    drange_cbv.BaseShaderRegister = 1;
    drange_cbv.NumDescriptors = 1;
    drange_cbv.OffsetInDescriptorsFromTableStart = 0;
    drange_cbv.Flags = D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE;
    drange_cbv.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_CBV;

    D3D12_DESCRIPTOR_RANGE1 drange_srv = {};
    drange_srv.RegisterSpace = 0;
    drange_srv.BaseShaderRegister = 0;
    drange_srv.NumDescriptors = 1 + 1 + 3 + 3;
    drange_srv.OffsetInDescriptorsFromTableStart = 0;
    drange_srv.Flags = D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE;
    drange_srv.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;

    D3D12_DESCRIPTOR_RANGE1 drange_tex = {};
    drange_tex.RegisterSpace = 0;
    drange_tex.BaseShaderRegister = 8;
    drange_tex.NumDescriptors = 2;
    drange_tex.OffsetInDescriptorsFromTableStart = 8;
    drange_tex.Flags = D3D12_DESCRIPTOR_RANGE_FLAG_DATA_STATIC;
    drange_tex.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;

    D3D12_DESCRIPTOR_RANGE1 drange_cbv_light = {};
    drange_cbv_light.RegisterSpace = 0;
    drange_cbv_light.RegisterSpace = 0;
    drange_cbv_light.BaseShaderRegister = 2;
    drange_cbv_light.NumDescriptors = 1;
    drange_cbv_light.OffsetInDescriptorsFromTableStart = 0;
    drange_cbv_light.Flags = D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE;
    drange_cbv_light.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_CBV;

    D3D12_DESCRIPTOR_RANGE1 drange_pixel[2] = {drange_srv, drange_tex};

    D3D12_ROOT_PARAMETER1 root_parameters[4];
    root_parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    root_parameters[0].Constants.Num32BitValues = sizeof(ModelViewProjection) / 4;
    root_parameters[0].Constants.RegisterSpace = 0;
    root_parameters[0].Constants.ShaderRegister = 0;
    root_parameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    root_parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    root_parameters[1].DescriptorTable.NumDescriptorRanges = 1;
    root_parameters[1].DescriptorTable.pDescriptorRanges = &drange_cbv;
    root_parameters[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    root_parameters[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    root_parameters[2].DescriptorTable.NumDescriptorRanges = 2;
    root_parameters[2].DescriptorTable.pDescriptorRanges = drange_pixel;
    root_parameters[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    root_parameters[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    root_parameters[3].DescriptorTable.NumDescriptorRanges = 1;
    root_parameters[3].DescriptorTable.pDescriptorRanges = &drange_cbv_light;
    root_parameters[3].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_STATIC_SAMPLER_DESC sampler_desc[1];
    sampler_desc[0].AddressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    sampler_desc[0].AddressV = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    sampler_desc[0].AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    sampler_desc[0].Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    sampler_desc[0].MinLOD = 0.0f;
    sampler_desc[0].MaxLOD = D3D12_FLOAT32_MAX;
    sampler_desc[0].BorderColor = D3D12_STATIC_BORDER_COLOR_OPAQUE_BLACK;
    sampler_desc[0].ComparisonFunc = D3D12_COMPARISON_FUNC_NONE;
    sampler_desc[0].ShaderRegister = 0;
    sampler_desc[0].RegisterSpace = 0;
    sampler_desc[0].MaxAnisotropy = 16;
    sampler_desc[0].MipLODBias = 0.0f;
    sampler_desc[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

    D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_signature_desc = {};
    root_signature_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
    root_signature_desc.Desc_1_1.NumParameters = 4;
    root_signature_desc.Desc_1_1.pParameters = root_parameters;
    root_signature_desc.Desc_1_1.NumStaticSamplers = 1;
    root_signature_desc.Desc_1_1.pStaticSamplers = sampler_desc;
    root_signature_desc.Desc_1_1.Flags = root_signature_flags;

    ComPtr<ID3DBlob> root_signature_blob;
    ComPtr<ID3DBlob> error_blob;
    VERIFY(D3D12SerializeVersionedRootSignature(&root_signature_desc, &root_signature_blob, &error_blob));

    VERIFY(g_device->CreateRootSignature(0, root_signature_blob->GetBufferPointer(), root_signature_blob->GetBufferSize(), IID_PPV_ARGS(&g_rootsignature_pbr_ntc)));

    struct PipelineStream
    {
      D3D12_PIPELINE_STATE_SUBOBJECT_TYPE root_signature_type;
      ID3D12RootSignature* root_signature;
      D3D12_PIPELINE_STATE_SUBOBJECT_TYPE input_layout_type;
      D3D12_INPUT_LAYOUT_DESC input_layout;
      D3D12_PIPELINE_STATE_SUBOBJECT_TYPE primitive_topology_type;
      D3D12_PRIMITIVE_TOPOLOGY_TYPE primitive_topology;
      D3D12_PIPELINE_STATE_SUBOBJECT_TYPE vs_type;
      D3D12_SHADER_BYTECODE vs;
      D3D12_PIPELINE_STATE_SUBOBJECT_TYPE ps_type;
      D3D12_SHADER_BYTECODE ps;
      D3D12_PIPELINE_STATE_SUBOBJECT_TYPE dsv_type;
      DXGI_FORMAT dsv;
      D3D12_PIPELINE_STATE_SUBOBJECT_TYPE rtv_type;
      D3D12_RT_FORMAT_ARRAY rtv;
    } pipeline_stream;

    pipeline_stream.root_signature_type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE;
    pipeline_stream.root_signature = g_rootsignature_pbr_ntc.Get();
    pipeline_stream.input_layout_type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_INPUT_LAYOUT;
    pipeline_stream.input_layout.NumElements = _countof(input_layout);
    pipeline_stream.input_layout.pInputElementDescs = input_layout;
    pipeline_stream.primitive_topology_type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY;
    pipeline_stream.primitive_topology = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pipeline_stream.vs_type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS;
    pipeline_stream.vs.BytecodeLength = vertex_shader_blob->GetBufferSize();
    pipeline_stream.vs.pShaderBytecode = vertex_shader_blob->GetBufferPointer();
    pipeline_stream.ps_type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS;
    pipeline_stream.ps.BytecodeLength = pixel_shader_blob->GetBufferSize();
    pipeline_stream.ps.pShaderBytecode = pixel_shader_blob->GetBufferPointer();
    pipeline_stream.dsv_type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT;
    pipeline_stream.dsv = DXGI_FORMAT_D32_FLOAT;
    pipeline_stream.rtv_type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS;
    pipeline_stream.rtv.NumRenderTargets = 1;
    pipeline_stream.rtv.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    for(int i = 1; i < 8; i++) pipeline_stream.rtv.RTFormats[i] = DXGI_FORMAT_UNKNOWN;

    D3D12_PIPELINE_STATE_STREAM_DESC pdesc = {};
    pdesc.SizeInBytes = sizeof(PipelineStream);
    pdesc.pPipelineStateSubobjectStream = &pipeline_stream;
    VERIFY(g_device->CreatePipelineState(&pdesc, IID_PPV_ARGS(&g_pipelinestate_pbr_ntc)));
  }

  // PBR NTC COOP

  {
    ComPtr<ID3DBlob> vertex_shader_blob;
    VERIFY(D3DReadFileToBlob(L"C:/Code/openntc/openntc-gui/pbr_ntc_coop_vs.cso", &vertex_shader_blob));

    ComPtr<ID3DBlob> pixel_shader_blob;
    VERIFY(D3DReadFileToBlob(L"C:/Code/openntc/openntc-gui/pbr_ntc_coop_ps.cso", &pixel_shader_blob));

    D3D12_INPUT_ELEMENT_DESC input_layout[] = {
      { "SV_Position", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
      { "NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
      { "TANGENT", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
      { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
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

    D3D12_DESCRIPTOR_RANGE1 drange_cbv = {};
    drange_cbv.RegisterSpace = 0;
    drange_cbv.BaseShaderRegister = 1;
    drange_cbv.NumDescriptors = 1;
    drange_cbv.OffsetInDescriptorsFromTableStart = 0;
    drange_cbv.Flags = D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE;
    drange_cbv.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_CBV;

    D3D12_DESCRIPTOR_RANGE1 drange_srv = {};
    drange_srv.RegisterSpace = 0;
    drange_srv.BaseShaderRegister = 0;
    drange_srv.NumDescriptors = 1 + 1 + 3 + 3;
    drange_srv.OffsetInDescriptorsFromTableStart = 0;
    drange_srv.Flags = D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE;
    drange_srv.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;

    D3D12_DESCRIPTOR_RANGE1 drange_tex = {};
    drange_tex.RegisterSpace = 0;
    drange_tex.BaseShaderRegister = 8;
    drange_tex.NumDescriptors = 2;
    drange_tex.OffsetInDescriptorsFromTableStart = 8;
    drange_tex.Flags = D3D12_DESCRIPTOR_RANGE_FLAG_DATA_STATIC;
    drange_tex.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;

    D3D12_DESCRIPTOR_RANGE1 drange_cbv_light = {};
    drange_cbv_light.RegisterSpace = 0;
    drange_cbv_light.RegisterSpace = 0;
    drange_cbv_light.BaseShaderRegister = 2;
    drange_cbv_light.NumDescriptors = 1;
    drange_cbv_light.OffsetInDescriptorsFromTableStart = 0;
    drange_cbv_light.Flags = D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE;
    drange_cbv_light.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_CBV;

    D3D12_DESCRIPTOR_RANGE1 drange_pixel[2] = {drange_srv, drange_tex};

    D3D12_ROOT_PARAMETER1 root_parameters[4];
    root_parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    root_parameters[0].Constants.Num32BitValues = sizeof(ModelViewProjection) / 4;
    root_parameters[0].Constants.RegisterSpace = 0;
    root_parameters[0].Constants.ShaderRegister = 0;
    root_parameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    root_parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    root_parameters[1].DescriptorTable.NumDescriptorRanges = 1;
    root_parameters[1].DescriptorTable.pDescriptorRanges = &drange_cbv;
    root_parameters[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    root_parameters[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    root_parameters[2].DescriptorTable.NumDescriptorRanges = 2;
    root_parameters[2].DescriptorTable.pDescriptorRanges = drange_pixel;
    root_parameters[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    root_parameters[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    root_parameters[3].DescriptorTable.NumDescriptorRanges = 1;
    root_parameters[3].DescriptorTable.pDescriptorRanges = &drange_cbv_light;
    root_parameters[3].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_STATIC_SAMPLER_DESC sampler_desc[1];
    sampler_desc[0].AddressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    sampler_desc[0].AddressV = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    sampler_desc[0].AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    sampler_desc[0].Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    sampler_desc[0].MinLOD = 0.0f;
    sampler_desc[0].MaxLOD = D3D12_FLOAT32_MAX;
    sampler_desc[0].BorderColor = D3D12_STATIC_BORDER_COLOR_OPAQUE_BLACK;
    sampler_desc[0].ComparisonFunc = D3D12_COMPARISON_FUNC_NONE;
    sampler_desc[0].ShaderRegister = 0;
    sampler_desc[0].RegisterSpace = 0;
    sampler_desc[0].MaxAnisotropy = 16;
    sampler_desc[0].MipLODBias = 0.0f;
    sampler_desc[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

    D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_signature_desc = {};
    root_signature_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
    root_signature_desc.Desc_1_1.NumParameters = 4;
    root_signature_desc.Desc_1_1.pParameters = root_parameters;
    root_signature_desc.Desc_1_1.NumStaticSamplers = 1;
    root_signature_desc.Desc_1_1.pStaticSamplers = sampler_desc;
    root_signature_desc.Desc_1_1.Flags = root_signature_flags;

    ComPtr<ID3DBlob> root_signature_blob;
    ComPtr<ID3DBlob> error_blob;
    VERIFY(D3D12SerializeVersionedRootSignature(&root_signature_desc, &root_signature_blob, &error_blob));

    VERIFY(g_device->CreateRootSignature(0, root_signature_blob->GetBufferPointer(), root_signature_blob->GetBufferSize(), IID_PPV_ARGS(&g_rootsignature_pbr_ntc_coop)));

    struct PipelineStream
    {
      D3D12_PIPELINE_STATE_SUBOBJECT_TYPE root_signature_type;
      ID3D12RootSignature* root_signature;
      D3D12_PIPELINE_STATE_SUBOBJECT_TYPE input_layout_type;
      D3D12_INPUT_LAYOUT_DESC input_layout;
      D3D12_PIPELINE_STATE_SUBOBJECT_TYPE primitive_topology_type;
      D3D12_PRIMITIVE_TOPOLOGY_TYPE primitive_topology;
      D3D12_PIPELINE_STATE_SUBOBJECT_TYPE vs_type;
      D3D12_SHADER_BYTECODE vs;
      D3D12_PIPELINE_STATE_SUBOBJECT_TYPE ps_type;
      D3D12_SHADER_BYTECODE ps;
      D3D12_PIPELINE_STATE_SUBOBJECT_TYPE dsv_type;
      DXGI_FORMAT dsv;
      D3D12_PIPELINE_STATE_SUBOBJECT_TYPE rtv_type;
      D3D12_RT_FORMAT_ARRAY rtv;
    } pipeline_stream;

    pipeline_stream.root_signature_type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE;
    pipeline_stream.root_signature = g_rootsignature_pbr_ntc_coop.Get();
    pipeline_stream.input_layout_type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_INPUT_LAYOUT;
    pipeline_stream.input_layout.NumElements = _countof(input_layout);
    pipeline_stream.input_layout.pInputElementDescs = input_layout;
    pipeline_stream.primitive_topology_type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY;
    pipeline_stream.primitive_topology = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pipeline_stream.vs_type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS;
    pipeline_stream.vs.BytecodeLength = vertex_shader_blob->GetBufferSize();
    pipeline_stream.vs.pShaderBytecode = vertex_shader_blob->GetBufferPointer();
    pipeline_stream.ps_type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS;
    pipeline_stream.ps.BytecodeLength = pixel_shader_blob->GetBufferSize();
    pipeline_stream.ps.pShaderBytecode = pixel_shader_blob->GetBufferPointer();
    pipeline_stream.dsv_type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT;
    pipeline_stream.dsv = DXGI_FORMAT_D32_FLOAT;
    pipeline_stream.rtv_type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS;
    pipeline_stream.rtv.NumRenderTargets = 1;
    pipeline_stream.rtv.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    for(int i = 1; i < 8; i++) pipeline_stream.rtv.RTFormats[i] = DXGI_FORMAT_UNKNOWN;

    D3D12_PIPELINE_STATE_STREAM_DESC pdesc = {};
    pdesc.SizeInBytes = sizeof(PipelineStream);
    pdesc.pPipelineStateSubobjectStream = &pipeline_stream;
    VERIFY(g_device->CreatePipelineState(&pdesc, IID_PPV_ARGS(&g_pipelinestate_pbr_ntc_coop)));
  }

  // Cubemap

  {
    ComPtr<ID3DBlob> vertex_shader_blob;
    VERIFY(D3DReadFileToBlob(L"C:/Code/openntc/openntc-gui/cubemap_vs.cso", &vertex_shader_blob));

    ComPtr<ID3DBlob> pixel_shader_blob;
    VERIFY(D3DReadFileToBlob(L"C:/Code/openntc/openntc-gui/cubemap_ps.cso", &pixel_shader_blob));

    D3D12_INPUT_ELEMENT_DESC input_layout[] = {
      { "SV_Position", 0, DXGI_FORMAT_R32G32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
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

    D3D12_DESCRIPTOR_RANGE1 drange = {};
    drange.RegisterSpace = 0;
    drange.BaseShaderRegister = 1;
    drange.NumDescriptors = 1;
    drange.OffsetInDescriptorsFromTableStart = 0;
    drange.Flags = D3D12_DESCRIPTOR_RANGE_FLAG_DATA_STATIC;
    drange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;

    D3D12_DESCRIPTOR_RANGE1 drange_cbv_light = {};
    drange_cbv_light.RegisterSpace = 0;
    drange_cbv_light.BaseShaderRegister = 1;
    drange_cbv_light.NumDescriptors = 1;
    drange_cbv_light.OffsetInDescriptorsFromTableStart = 0;
    drange_cbv_light.Flags = D3D12_DESCRIPTOR_RANGE_FLAG_DATA_STATIC;
    drange_cbv_light.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_CBV;

    D3D12_ROOT_PARAMETER1 root_parameters[3];
    root_parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    root_parameters[0].Constants.Num32BitValues = sizeof(CubemapTransforms) / 4;
    root_parameters[0].Constants.RegisterSpace = 0;
    root_parameters[0].Constants.ShaderRegister = 0;
    root_parameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    root_parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    root_parameters[1].DescriptorTable.NumDescriptorRanges = 1;
    root_parameters[1].DescriptorTable.pDescriptorRanges = &drange;
    root_parameters[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    root_parameters[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    root_parameters[2].DescriptorTable.NumDescriptorRanges = 1;
    root_parameters[2].DescriptorTable.pDescriptorRanges = &drange_cbv_light;
    root_parameters[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

    D3D12_STATIC_SAMPLER_DESC sampler_desc = {};
    sampler_desc.AddressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    sampler_desc.AddressV = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    sampler_desc.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    sampler_desc.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    sampler_desc.MinLOD = 0.0f;
    sampler_desc.MaxLOD = D3D12_FLOAT32_MAX;
    sampler_desc.BorderColor = D3D12_STATIC_BORDER_COLOR_OPAQUE_BLACK;
    sampler_desc.ComparisonFunc = D3D12_COMPARISON_FUNC_NONE;
    sampler_desc.ShaderRegister = 0;
    sampler_desc.RegisterSpace = 0;
    sampler_desc.MaxAnisotropy = 16;
    sampler_desc.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

    D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_signature_desc = {};
    root_signature_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
    root_signature_desc.Desc_1_1.NumParameters = 3;
    root_signature_desc.Desc_1_1.pParameters = root_parameters;
    root_signature_desc.Desc_1_1.NumStaticSamplers = 1;
    root_signature_desc.Desc_1_1.pStaticSamplers = &sampler_desc;
    root_signature_desc.Desc_1_1.Flags = root_signature_flags;

    ComPtr<ID3DBlob> root_signature_blob;
    ComPtr<ID3DBlob> error_blob;
    VERIFY(D3D12SerializeVersionedRootSignature(&root_signature_desc, &root_signature_blob, &error_blob));

    VERIFY(g_device->CreateRootSignature(0, root_signature_blob->GetBufferPointer(), root_signature_blob->GetBufferSize(), IID_PPV_ARGS(&g_rootsignature_cubemap)));

    D3D12_DEPTH_STENCIL_DESC depth_stencil = {};
    depth_stencil.DepthEnable = FALSE;

    struct PipelineStream
    {
      D3D12_PIPELINE_STATE_SUBOBJECT_TYPE root_signature_type;
      ID3D12RootSignature* root_signature;
      D3D12_PIPELINE_STATE_SUBOBJECT_TYPE input_layout_type;
      D3D12_INPUT_LAYOUT_DESC input_layout;
      D3D12_PIPELINE_STATE_SUBOBJECT_TYPE primitive_topology_type;
      D3D12_PRIMITIVE_TOPOLOGY_TYPE primitive_topology;
      D3D12_PIPELINE_STATE_SUBOBJECT_TYPE vs_type;
      D3D12_SHADER_BYTECODE vs;
      D3D12_PIPELINE_STATE_SUBOBJECT_TYPE ps_type;
      D3D12_SHADER_BYTECODE ps;
      D3D12_PIPELINE_STATE_SUBOBJECT_TYPE dsv_type;
      DXGI_FORMAT dsv;
      D3D12_PIPELINE_STATE_SUBOBJECT_TYPE ds_type;
      D3D12_DEPTH_STENCIL_DESC ds;
      D3D12_PIPELINE_STATE_SUBOBJECT_TYPE rtv_type;
      D3D12_RT_FORMAT_ARRAY rtv;
    } pipeline_stream;

    pipeline_stream.root_signature_type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE;
    pipeline_stream.root_signature = g_rootsignature_cubemap.Get();
    pipeline_stream.input_layout_type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_INPUT_LAYOUT;
    pipeline_stream.input_layout.NumElements = _countof(input_layout);
    pipeline_stream.input_layout.pInputElementDescs = input_layout;
    pipeline_stream.primitive_topology_type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY;
    pipeline_stream.primitive_topology = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pipeline_stream.vs_type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS;
    pipeline_stream.vs.BytecodeLength = vertex_shader_blob->GetBufferSize();
    pipeline_stream.vs.pShaderBytecode = vertex_shader_blob->GetBufferPointer();
    pipeline_stream.ps_type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS;
    pipeline_stream.ps.BytecodeLength = pixel_shader_blob->GetBufferSize();
    pipeline_stream.ps.pShaderBytecode = pixel_shader_blob->GetBufferPointer();
    pipeline_stream.dsv_type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT;
    pipeline_stream.dsv = DXGI_FORMAT_D32_FLOAT;
    pipeline_stream.ds_type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL;
    pipeline_stream.ds = depth_stencil;
    pipeline_stream.rtv_type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS;
    pipeline_stream.rtv.NumRenderTargets = 1;
    pipeline_stream.rtv.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    for(int i = 1; i < 8; i++) pipeline_stream.rtv.RTFormats[i] = DXGI_FORMAT_UNKNOWN;

    D3D12_PIPELINE_STATE_STREAM_DESC pdesc = {};
    pdesc.SizeInBytes = sizeof(PipelineStream);
    pdesc.pPipelineStateSubobjectStream = &pipeline_stream;
    VERIFY(g_device->CreatePipelineState(&pdesc, IID_PPV_ARGS(&g_pipelinestate_cubemap)));
  }

  // Texture

  {
    SharedContext::Access access = g_ctx.Acquire();

    openntc::Result load_res = access.ctx_.LoadManifest("C:/Code/openntc/img/Bricks101_2K-JPG/manifest.json");
    VERIFY(load_res == openntc::Result::Success);
    
    RebuildTextureResources(access);
  }

  LoadIBL();

  g_contentloaded = true;

  ResizeDepthBuffer(g_width, g_height);
}

void UploadCompressedData(openntc::CompressedData& cdata)
{
  {
    NTCInfo ntc_info = {};
    for (int i = 0; i < cdata.level_count_; i++)
    {
      ntc_info.g0_grid_dim[i] = cdata.g0_grid_dim_[i];
      ntc_info.g1_grid_dim[i] = cdata.g1_grid_dim_[i];
      ntc_info.g0_offset[i] = static_cast<uint32_t>(cdata.g0_offset_[i]);
      ntc_info.g1_offset[i] = static_cast<uint32_t>(cdata.g1_offset_[i]);
    }

    ntc_info.g0_bytes_per_channel = cdata.g0_bytes_per_channel_;
    ntc_info.g1_bytes_per_channel = cdata.g1_bytes_per_channel_;
    ntc_info.g0_channels = cdata.g0_channels_;
    ntc_info.g1_channels = cdata.g1_channels_;
    ntc_info.dim = cdata.dim_;
    ntc_info.mip_count = cdata.mip_count_;
    ntc_info.rcp_s_a1 = 1.0f / cdata.caldata_.s_a1;
    ntc_info.rcp_s_a2 = 1.0f / cdata.caldata_.s_a2;

    uint64_t cbv_size = RoundUpTo(sizeof(NTCInfo), 256);

    g_buffer_ntc_info = CreateBufferWithData(g_device.Get(), D3D12_HEAP_TYPE_GPU_UPLOAD, cbv_size, sizeof(NTCInfo), &ntc_info);
    
    D3D12_CONSTANT_BUFFER_VIEW_DESC cbv_desc = {};
    cbv_desc.BufferLocation = g_buffer_ntc_info->GetGPUVirtualAddress();
    cbv_desc.SizeInBytes = cbv_size;
    g_device->CreateConstantBufferView(&cbv_desc, g_dhandle_ntc_info.cpu);
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
    srv_desc.Shader4ComponentMapping = D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_0, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_1, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_2, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_3);

    g_device->CreateShaderResourceView(g_buffer_g0.Get(), &srv_desc, g_dhandle_ntc_data[0].cpu);
    
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
    srv_desc.Shader4ComponentMapping = D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_0, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_1, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_2, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_3);

    g_device->CreateShaderResourceView(g_buffer_g1.Get(), &srv_desc, g_dhandle_ntc_data[1].cpu);
    
    void* mapped = nullptr;
    D3D12_RANGE read_range = {0, 0};
    g_buffer_g1->Map(0, &read_range, &mapped);
    for (int level_i = 0; level_i < cdata.level_count_; level_i++)
      memcpy(static_cast<char*>(mapped) + cdata.g1_offset_[level_i], cdata.g1_[level_i], cdata.g1_size_[level_i]);
    g_buffer_g1->Unmap(0, nullptr);
  }

  {
    g_buffer_W0 = CreateBufferWithData(g_device.Get(), D3D12_HEAP_TYPE_GPU_UPLOAD, cdata.W0_size_, cdata.W0_size_, cdata.W0_);
    
    D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
    srv_desc.Format = DXGI_FORMAT_R32_TYPELESS;
    srv_desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srv_desc.Buffer.FirstElement = 0;
    srv_desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
    srv_desc.Buffer.NumElements = cdata.W0_size_ / (sizeof(uint32_t));
    srv_desc.Buffer.StructureByteStride = 0;
    srv_desc.Shader4ComponentMapping = D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_0, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_1, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_2, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_3);

    g_device->CreateShaderResourceView(g_buffer_W0.Get(), &srv_desc, g_dhandle_ntc_data[2].cpu);
  }

  {
    g_buffer_W1 = CreateBufferWithData(g_device.Get(), D3D12_HEAP_TYPE_GPU_UPLOAD, cdata.W1_size_, cdata.W1_size_, cdata.W1_);

    D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
    srv_desc.Format = DXGI_FORMAT_R32_TYPELESS;
    srv_desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srv_desc.Buffer.FirstElement = 0;
    srv_desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
    srv_desc.Buffer.NumElements = cdata.W1_size_ / (sizeof(uint32_t));
    srv_desc.Buffer.StructureByteStride = 0;
    srv_desc.Shader4ComponentMapping = D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_0, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_1, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_2, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_3);

    g_device->CreateShaderResourceView(g_buffer_W1.Get(), &srv_desc, g_dhandle_ntc_data[3].cpu);
  }

  {
    g_buffer_Wout = CreateBufferWithData(g_device.Get(), D3D12_HEAP_TYPE_GPU_UPLOAD, cdata.Wout_size_, cdata.Wout_size_, cdata.Wout_);

    D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
    srv_desc.Format = DXGI_FORMAT_R32_TYPELESS;
    srv_desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srv_desc.Buffer.FirstElement = 0;
    srv_desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
    srv_desc.Buffer.NumElements = cdata.Wout_size_ / (sizeof(uint32_t));
    srv_desc.Buffer.StructureByteStride = 0;
    srv_desc.Shader4ComponentMapping = D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_0, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_1, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_2, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_3);

    g_device->CreateShaderResourceView(g_buffer_Wout.Get(), &srv_desc, g_dhandle_ntc_data[4].cpu);
  }

  {
    g_buffer_W0_scale = CreateBufferWithData(g_device.Get(), D3D12_HEAP_TYPE_GPU_UPLOAD, cdata.W0_scale_size_, cdata.W0_scale_size_, cdata.W0_scale_);

    D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
    srv_desc.Format = DXGI_FORMAT_R32_TYPELESS;
    srv_desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srv_desc.Buffer.FirstElement = 0;
    srv_desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
    srv_desc.Buffer.NumElements = cdata.W0_scale_size_ / (sizeof(float));
    srv_desc.Buffer.StructureByteStride = 0;
    srv_desc.Shader4ComponentMapping = D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_0, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_1, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_2, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_3);

    g_device->CreateShaderResourceView(g_buffer_W0_scale.Get(), &srv_desc, g_dhandle_ntc_data[5].cpu);
  }

  {
    g_buffer_W1_scale = CreateBufferWithData(g_device.Get(), D3D12_HEAP_TYPE_GPU_UPLOAD, cdata.W1_scale_size_, cdata.W1_scale_size_, cdata.W1_scale_);

    D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
    srv_desc.Format = DXGI_FORMAT_R32_TYPELESS;
    srv_desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srv_desc.Buffer.FirstElement = 0;
    srv_desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
    srv_desc.Buffer.NumElements = cdata.W1_scale_size_ / (sizeof(float));
    srv_desc.Buffer.StructureByteStride = 0;
    srv_desc.Shader4ComponentMapping = D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_0, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_1, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_2, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_3);

    g_device->CreateShaderResourceView(g_buffer_W1_scale.Get(), &srv_desc, g_dhandle_ntc_data[6].cpu);
  }

  {
    g_buffer_Wout_scale = CreateBufferWithData(g_device.Get(), D3D12_HEAP_TYPE_GPU_UPLOAD, cdata.Wout_scale_size_, cdata.Wout_scale_size_, cdata.Wout_scale_);

    D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
    srv_desc.Format = DXGI_FORMAT_R32_TYPELESS;
    srv_desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srv_desc.Buffer.FirstElement = 0;
    srv_desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
    srv_desc.Buffer.NumElements = cdata.Wout_scale_size_ / (sizeof(float));
    srv_desc.Buffer.StructureByteStride = 0;
    srv_desc.Shader4ComponentMapping = D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_0, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_1, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_2, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_3);

    g_device->CreateShaderResourceView(g_buffer_Wout_scale.Get(), &srv_desc, g_dhandle_ntc_data[7].cpu);
  }
}

void LoadCompressedDataFromContext(SharedContext::Access& access)
{
  if (g_compressed_data_loaded) return;

  openntc::CompressedData cdata = access.ctx_.GetCompressedData();
  openntc::Context::Dump("bricks101_2k.ntc", cdata);
  UploadCompressedData(cdata);
  g_compressed_data_loaded = true;
}

void LoadCompressedDataFromFile()
{
  if (g_compressed_data_loaded) return;

  openntc::Result res = openntc::Context::Load("bricks101_2k.ntc", g_fil_data);
  VERIFY(res == openntc::Result::Success);
  openntc::CompressedData cdata = g_fil_data.Data();
  UploadCompressedData(cdata);
  g_compressed_data_loaded = true;
}

static void SetPipelineStateForShader(Shader shader)
{
  if (shader == Shader::Flat)
  {
    g_commandlist->SetPipelineState(g_pipelinestate_flat.Get());
    g_commandlist->SetGraphicsRootSignature(g_rootsignature_flat.Get());
  }
  else if (shader == Shader::GGX)
  {
    g_commandlist->SetPipelineState(g_pipelinestate_ggx.Get());
    g_commandlist->SetGraphicsRootSignature(g_rootsignature_ggx.Get());
  }
  else if (shader == Shader::PBR_NTC)
  {
    g_commandlist->SetPipelineState(g_pipelinestate_pbr_ntc.Get());
    g_commandlist->SetGraphicsRootSignature(g_rootsignature_pbr_ntc.Get());
  }
  else if (shader == Shader::PBR_NTC_COOP)
  {
    g_commandlist->SetPipelineState(g_pipelinestate_pbr_ntc_coop.Get());
    g_commandlist->SetGraphicsRootSignature(g_rootsignature_pbr_ntc_coop.Get());
  }
}

static void SetDescriptorsForShader(Shader shader)
{
  if (shader == Shader::Flat)
  {
    XMMATRIX mvp_mat = XMMatrixMultiply(g_model_mat, g_view_mat);
    mvp_mat = XMMatrixMultiply(mvp_mat, g_proj_mat);
    D3D12_GPU_DESCRIPTOR_HANDLE tex_color_handle = g_dhandle_tex[g_gui_texture].gpu;
    g_commandlist->SetGraphicsRoot32BitConstants(0, sizeof(XMMATRIX) / 4, &mvp_mat, 0);
    g_commandlist->SetGraphicsRootDescriptorTable(1, tex_color_handle);
  }
  else if (shader == Shader::GGX)
  {
    ModelViewProjection mvp = {};
    mvp.model_to_world = g_model_mat;
    mvp.world_to_view = g_view_mat;
    mvp.view_to_proj = g_proj_mat;

    D3D12_GPU_DESCRIPTOR_HANDLE tex_color_handle = g_dhandle_tex[0].gpu;
    D3D12_GPU_DESCRIPTOR_HANDLE srv_handle = g_dhandle_ibl[0].gpu;
    D3D12_GPU_DESCRIPTOR_HANDLE cbv_lighing_handle = g_dhandle_lparams.gpu;
    g_commandlist->SetGraphicsRoot32BitConstants(0, sizeof(ModelViewProjection) / 4, &mvp, 0);
    g_commandlist->SetGraphicsRootDescriptorTable(1, tex_color_handle);
    g_commandlist->SetGraphicsRootDescriptorTable(2, srv_handle);
    g_commandlist->SetGraphicsRootDescriptorTable(3, cbv_lighing_handle);
  }
  else if (shader == Shader::PBR_NTC || shader == Shader::PBR_NTC_COOP)
  {
    ModelViewProjection mvp = {};
    mvp.model_to_world = g_model_mat;
    mvp.world_to_view = g_view_mat;
    mvp.view_to_proj = g_proj_mat;

    D3D12_GPU_DESCRIPTOR_HANDLE cbv_handle = g_dhandle_ntc_info.gpu;
    D3D12_GPU_DESCRIPTOR_HANDLE srv_handle = g_dhandle_ntc_data[0].gpu;
    D3D12_GPU_DESCRIPTOR_HANDLE cbv_lighing_handle = g_dhandle_lparams.gpu;
    g_commandlist->SetGraphicsRoot32BitConstants(0, sizeof(ModelViewProjection) / 4, &mvp, 0);
    g_commandlist->SetGraphicsRootDescriptorTable(1, cbv_handle);
    g_commandlist->SetGraphicsRootDescriptorTable(2, srv_handle);
    g_commandlist->SetGraphicsRootDescriptorTable(3, cbv_lighing_handle);
  }
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

void PerformLoadManifest()
{
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
        
        // TODO: Log failure in GUI.

        if (load_res == openntc::Result::Success)
        {
          RebuildTextureResources(access);
        }
      }
    }
  }
}

void PerformLoadCompressed()
{
  // TODO: Add open file dialog
  LoadCompressedDataFromFile();
}

void PerformSaveCompressed()
{
  // TODO: Currently we auto-save to a path fixed in code. Add save dialog.
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
      if (ImGui::BeginMenu("File"))
      {
        if (ImGui::MenuItem("Load Manifest...", "Ctrl+O"))
        {
          PerformLoadManifest();
        }
        if (ImGui::MenuItem("Load Compressed...", "Ctrl+Shift+O"))
        {
          PerformLoadCompressed();
        }
        if (ImGui::MenuItem("Save Compressed...", "Ctrl+S"))
        {
          PerformSaveCompressed();
        }
        ImGui::Separator();
        if (ImGui::MenuItem("Exit", "Alt+F4"))
        {
          PostQuitMessage(0);
        }
        ImGui::EndMenu();
      }
      ImGui::EndMainMenuBar();
    }
  }

  if (ImGui::Shortcut(ImGuiMod_Ctrl | ImGuiKey_O, ImGuiInputFlags_RouteGlobal))
  {
    PerformLoadManifest();
  }

  if (ImGui::Shortcut(ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_O, ImGuiInputFlags_RouteGlobal))
  {
    PerformLoadCompressed();
  }

  if (ImGui::Shortcut(ImGuiMod_Ctrl | ImGuiKey_S, ImGuiInputFlags_RouteGlobal))
  {
    PerformSaveCompressed();
  }

  const ImGuiViewport* vp = ImGui::GetMainViewport();
  const float sidebar_w = 350.0f;
  const float footer_h = ImGui::GetFrameHeight() * 1.6f;
  const ImGuiWindowFlags pinned_flags = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                                        ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar;
                                      
  ImGui::SetNextWindowPos({vp->WorkPos.x + vp->WorkSize.x - sidebar_w, vp->WorkPos.y});
  ImGui::SetNextWindowSize({sidebar_w, vp->WorkSize.y - footer_h});
  ImGui::Begin("Sidebar", nullptr, pinned_flags);
  ImGui::SeparatorText("Shading");
  ImGui::Combo("Camera", &g_gui_camera_mode, g_map_camera_mode_to_name, static_cast<int32_t>(CameraMode::Count));
  ImGui::Combo("Left", &g_gui_shader_left, g_map_shader_to_name, static_cast<int32_t>(Shader::Count));
  ImGui::Combo("Right", &g_gui_shader_right, g_map_shader_to_name, static_cast<int32_t>(Shader::Count));
  ImGui::SliderFloat("FOV", &g_fov_y, 10.0f, 180.0f);
  ImGui::SliderFloat("Displacement Scale", &g_gui_displacement_scale, 0.0f, 0.5f);
  ImGui::SliderFloat("Normal Scale", &g_gui_normal_scale, 0.0f, 10.0f);
  ImGui::SliderFloat("Exposure", &g_gui_exposure, 0.5f, 1.5f);
  ImGui::Combo("Model", &g_gui_model, g_map_model_to_name, g_kModelCount);
  Shader shader_left = static_cast<Shader>(g_gui_shader_left);
  Shader shader_right = static_cast<Shader>(g_gui_shader_right);
  if (shader_left == Shader::Flat)
  {
    ImGui::Combo("Channel", &g_gui_texture, g_map_semantic_to_name, openntc::kMaxSources);
  }
  ImGui::Checkbox("Spin", &g_gui_spin);

  ImGui::SeparatorText("Train");
  bool train_button = ImGui::Button("Train");
  {
    std::optional<SharedContext::Access> oaccess = g_ctx.TryAcquire();
    SharedFields fields = g_shared_fields.load(std::memory_order_seq_cst);
    if (oaccess.has_value())
    {
      SharedContext::Access& access = oaccess.value();
      if (train_button)
      {
        if (!fields.train_in_progress)
        {
          fields.train_in_progress = true;
          fields.train_complete = false;
          fields.train_steps = 0;
          fields.train_total_steps = 30000;
          g_shared_fields.store(fields, std::memory_order_seq_cst);
          g_stop_training.store(false, std::memory_order_seq_cst);
          g_compressed_data_loaded = false;
          g_train_job = std::async(std::launch::async, PerformTrainingJob);
        }
      }
    }
    if (fields.train_in_progress)
    {
      ImGui::ProgressBar((float)fields.train_steps / fields.train_total_steps);
    }
    if (fields.train_complete)
    {
      ImGui::LabelText("PSNR", "%f", fields.eval_psnr);
      ImGui::LabelText("MSE", "%f", fields.eval_mse);
      if (!g_compressed_data_loaded)
      {
        if (oaccess.has_value())
        {
          SharedContext::Access& access = oaccess.value();
          LoadCompressedDataFromContext(access);
        }
      }
    }
  }
  ImGui::End();

  ImGui::SetNextWindowPos({vp->WorkPos.x, vp->WorkPos.y + vp->WorkSize.y - footer_h});
  ImGui::SetNextWindowSize({vp->WorkSize.x, footer_h});
  ImGui::Begin("Footer", nullptr, pinned_flags);
  ImGui::Text("openntc-gui v0.1 pre-release");
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
  float distance_spacing_factor = 1.1f;
  float eye_distance = distance_spacing_factor * bounding_sphere_radius / std::sinf(fov / 2.0f);
  const XMVECTOR focus_pos = XMVectorSet(0.0f, 0.0f, 0.0f, 1.0f);
  const XMVECTOR up_dir = XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f);

  if (cmode == CameraMode::Orbit)
  {
    float angle = g_gui_spin ? static_cast<float>(std::fmod(g_total_seconds, std::acos(-1.0) * 2.0)) : 0.0f;
    const XMVECTOR rotation_axis = XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f);
    g_model_mat = XMMatrixRotationAxis(rotation_axis, angle);
    const XMVECTOR eye_pos = XMVectorSet(0.0f, 0.0f, -eye_distance, 1.0f);
    g_view_mat = XMMatrixLookAtLH(eye_pos, focus_pos, up_dir);
    g_proj_mat = XMMatrixPerspectiveFovLH(fov, aspect_ratio, 0.1f, 100.0f);
  }
  else if (cmode == CameraMode::Controlled)
  {
    const XMVECTOR eye_pos = focus_pos + eye_distance * XMVectorSet(std::cosf(g_pitch) * std::sinf(g_yaw), std::sinf(g_pitch), std::cosf(g_pitch) * std::cosf(g_yaw), 0.0f);
    g_model_mat = XMMatrixIdentity();
    g_view_mat = XMMatrixLookAtLH(eye_pos, focus_pos, up_dir);
    g_proj_mat = XMMatrixPerspectiveFovLH(fov_y, aspect_ratio, 0.1f, 100.0f);
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
    if (ImGui::IsItemActive())
    {
      if (static_cast<CameraMode>(g_gui_camera_mode) == CameraMode::Orbit)
        g_gui_camera_mode = static_cast<int32_t>(CameraMode::Controlled);

      const float sensitivity = 0.008f;
      ImGuiIO& io = ImGui::GetIO();
      g_yaw   += io.MouseDelta.x * sensitivity;
      g_pitch += io.MouseDelta.y * sensitivity;

      const float pitch_limit = XMConvertToRadians(89.0f);
      g_pitch = std::clamp(g_pitch, -pitch_limit, pitch_limit);
    }
    ImGui::End();
    ImGui::PopStyleVar();
  }

  auto command_allocator = g_commandallocators[g_frame_i];
  auto buffer = g_buffers[g_frame_i];
  command_allocator->Reset();
  g_commandlist->Reset(command_allocator.Get(), nullptr);
  D3D12_CPU_DESCRIPTOR_HANDLE rtv_handle = g_dhandle_rtv[g_frame_i].cpu;
  D3D12_CPU_DESCRIPTOR_HANDLE dsv_handle = g_dhandle_dsv.cpu;
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

  {
    TransitionResource(g_commandlist, buffer, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET);
    FLOAT clear_color[] = { 0.4f, 0.6f, 0.9f, 1.0f };
    g_commandlist->ClearRenderTargetView(rtv_handle, clear_color, 0, nullptr);
    g_commandlist->ClearDepthStencilView(dsv_handle, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
  }

  ID3D12DescriptorHeap* heaps[1] = {g_dalloc_srv.GetHeapUnsafe()};
  g_commandlist->SetDescriptorHeaps(_countof(heaps), heaps);
  g_commandlist->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
  g_commandlist->IASetVertexBuffers(0, 1, &g_vbv[g_gui_model]);
  g_commandlist->IASetIndexBuffer(&g_ibv[g_gui_model]);

  {
    LightingParams lp = {};
    lp.exposure = g_gui_exposure;
    lp.displacement_scale = g_gui_displacement_scale;
    lp.normal_scale = g_gui_normal_scale;
    for (int i = 0; i < 9; i++)
      lp.diffuse_sh[i] = g_diffuse_sh[i];
    void* mapped = 0;
    D3D12_RANGE map_range = {0, 0};
    g_buffer_lighting_params->Map(0, &map_range, &mapped);
    memcpy(mapped, &lp, sizeof(lp));
    g_buffer_lighting_params->Unmap(0, nullptr);
  }

  {
    g_commandlist->RSSetViewports(1, &viewport_left);
    g_commandlist->RSSetScissorRects(1, &scissor);
    g_commandlist->OMSetRenderTargets(1, &rtv_handle, FALSE, &dsv_handle);

    CubemapTransforms cubemap_transforms = {};
    cubemap_transforms.proj_to_view = XMMatrixInverse(nullptr, g_proj_mat);
    cubemap_transforms.view_to_world = XMMatrixInverse(nullptr, g_view_mat);
    g_commandlist->SetPipelineState(g_pipelinestate_cubemap.Get());
    g_commandlist->SetGraphicsRootSignature(g_rootsignature_cubemap.Get());
    g_commandlist->SetGraphicsRoot32BitConstants(0, sizeof(CubemapTransforms) / 4, &cubemap_transforms, 0);
    g_commandlist->SetGraphicsRootDescriptorTable(1, g_dhandle_ibl[0].gpu);
    g_commandlist->SetGraphicsRootDescriptorTable(2, g_dhandle_lparams.gpu);
    g_commandlist->DrawInstanced(6, 1, 0, 0);

    SetPipelineStateForShader(shader_left);
    SetDescriptorsForShader(shader_left);
    g_commandlist->DrawIndexedInstanced(g_map_model_to_index_count[g_gui_model], 1, 0, 0, 0);
  }
  {
    g_commandlist->RSSetViewports(1, &viewport_right);
    g_commandlist->RSSetScissorRects(1, &scissor);
    g_commandlist->OMSetRenderTargets(1, &rtv_handle, FALSE, &dsv_handle);

    CubemapTransforms cubemap_transforms = {};
    cubemap_transforms.proj_to_view = XMMatrixInverse(nullptr, g_proj_mat);
    cubemap_transforms.view_to_world = XMMatrixInverse(nullptr, g_view_mat);
    g_commandlist->SetPipelineState(g_pipelinestate_cubemap.Get());
    g_commandlist->SetGraphicsRootSignature(g_rootsignature_cubemap.Get());
    g_commandlist->SetGraphicsRoot32BitConstants(0, sizeof(CubemapTransforms) / 4, &cubemap_transforms, 0);
    g_commandlist->SetGraphicsRootDescriptorTable(1, g_dhandle_ibl[0].gpu);
    g_commandlist->SetGraphicsRootDescriptorTable(2, g_dhandle_lparams.gpu);
    g_commandlist->DrawInstanced(6, 1, 0, 0);

    SetPipelineStateForShader(shader_right);
    SetDescriptorsForShader(shader_right);
    g_commandlist->DrawIndexedInstanced(g_map_model_to_index_count[g_gui_model], 1, 0, 0, 0);
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

void UpdateRenderTargetViews(ComPtr<ID3D12Device2> device, ComPtr<IDXGISwapChain4> swapchain, ComPtr<ID3D12DescriptorHeap> descriptor_heap)
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
    VERIFY(g_swapchain->ResizeBuffers(g_numframes, g_width, g_height, swapchain_desc.BufferDesc.Format, swapchain_desc.Flags));
    g_frame_i = g_swapchain->GetCurrentBackBufferIndex();
    UpdateRenderTargetViews(g_device, g_swapchain, g_dalloc_rtv.GetHeapUnsafe());
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
      UINT window_style = WS_OVERLAPPEDWINDOW & ~(WS_CAPTION | WS_SYSMENU | WS_THICKFRAME | WS_MINIMIZEBOX | WS_MAXIMIZEBOX);
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

    if ((dxgi_adapter_desc1.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0 &&
        SUCCEEDED(D3D12CreateDevice(dxgi_adapter1.Get(), D3D_FEATURE_LEVEL_12_2, __uuidof(ID3D12Device), nullptr)) &&
        dxgi_adapter_desc1.DedicatedVideoMemory > max_dedicated_vram)
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

  D3D12_FEATURE_DATA_LINEAR_ALGEBRA_SUPPORT la = {};
  VERIFY(device2->CheckFeatureSupport(D3D12_FEATURE_LINEAR_ALGEBRA_SUPPORT, &la, sizeof(la)));
  assert(la.LinearAlgebraTier != D3D12_LINEAR_ALGEBRA_TIER_NOT_SUPPORTED);

  D3D12_FEATURE_DATA_LINEAR_ALGEBRA_MATRIX_OPERATION_SUPPORT op = {};
  op.OperationType = D3D12_LINEAR_ALGEBRA_OPERATION_TYPE_THREAD_VECTOR_MATRIX_MULTIPLY;
  op.ThreadVectorMatrixMultiply.VectorInputType = D3D12_LINEAR_ALGEBRA_DATATYPE_SINT8;
  op.ThreadVectorMatrixMultiply.MatrixInputType = D3D12_LINEAR_ALGEBRA_DATATYPE_SINT8;
  op.ThreadVectorMatrixMultiply.BiasInputType = D3D12_LINEAR_ALGEBRA_DATATYPE_SINT32;
  op.ThreadVectorMatrixMultiply.VectorResultType = D3D12_LINEAR_ALGEBRA_DATATYPE_SINT32;
  VERIFY(device2->CheckFeatureSupport(D3D12_FEATURE_LINEAR_ALGEBRA_LINEAR_ALGEBRA_MATRIX_OPERATION_SUPPORT, &op, sizeof(op)));

  D3D12_LINEAR_ALGEBRA_MULTIPLICATION_SUPPORT_FLAGS flags = op.ThreadVectorMatrixMultiply.SupportFlags;
  bool native_support = flags & D3D12_LINEAR_ALGEBRA_MULTIPLICATION_SUPPORT_FLAG_SUPPORTED;
  bool emu_input = flags & D3D12_LINEAR_ALGEBRA_MULTIPLICATION_SUPPORT_FLAG_EMULATED_INPUTS;
  bool emu_output = flags & D3D12_LINEAR_ALGEBRA_MULTIPLICATION_SUPPORT_FLAG_EMULATED_OUTPUTS;

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
      if (FAILED(factory5->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &allow_tearing, sizeof(allow_tearing))))
      {
        allow_tearing = FALSE;
      }
    }
  }

  return (allow_tearing == TRUE);
}

ComPtr<IDXGISwapChain4> CreateSwapChain(HWND hwnd, ComPtr<ID3D12CommandQueue> command_queue, uint32_t width, uint32_t height, uint32_t buffer_count)
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
  VERIFY(dxgi_factory4->CreateSwapChainForHwnd(command_queue.Get(), g_hwnd, &swapchain_desc, nullptr, nullptr, &swapchain1));
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

ComPtr<ID3D12GraphicsCommandList10> CreateCommandList(ComPtr<ID3D12Device2> device, ComPtr<ID3D12CommandAllocator> command_allocator, D3D12_COMMAND_LIST_TYPE type)
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

int CALLBACK wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PWSTR lpCmdLine, int nCmdShow)
{
  {
    SharedContext::Access access = g_ctx.Acquire();
    SharedFields fields = g_shared_fields.load(std::memory_order_seq_cst);
    openntc::ContextInitInfo init_info = {};
    init_info.profile = openntc::Profile::Bpp_0_2;
    access.ctx_.Init(init_info);
    fields.train_in_progress = false;
    fields.train_complete = false;
    fields.train_steps = 0;
    fields.train_total_steps = 0;

    g_shared_fields.store(fields, std::memory_order_seq_cst);
  }

  ::CoInitializeEx(nullptr, COINIT_MULTITHREADED);
  SetThreadDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);

  const char* wndclass_name = "openntc-gui-window-class";

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
    g_dhandle_ntc_info = g_dalloc_srv.Allocate();
    for (int i = 0; i < 1 + 1 + 3 + 3; i++)
      g_dhandle_ntc_data[i] = g_dalloc_srv.Allocate();
    for (int i = 0; i < 2; i++)
      g_dhandle_ibl[i] = g_dalloc_srv.Allocate();
    g_dhandle_lparams = g_dalloc_srv.Allocate();
  }

  for (int i = 0; i < g_imgui_srv_count; i++)
  {
    UINT slot = g_nonimgui_srv_count + i;
    (void) g_dalloc_srv.Allocate();
    g_imgui_available_srv_slots.push_back(slot);
  }

  {
    g_dalloc_rtv.Init(g_device.Get(), D3D12_DESCRIPTOR_HEAP_TYPE_RTV, false, g_numframes);
    for (int frame_i = 0; frame_i < g_numframes; frame_i++)
      g_dhandle_rtv[frame_i] = g_dalloc_rtv.Allocate();
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
  ImGui::StyleColorsDark();
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