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

SharedContext g_ctx;
std::future<void> g_train_job;
std::atomic<bool> g_stop_training;

const uint8_t g_numframes = 2;
uint32_t g_width = 1280;
uint32_t g_height = 720;

HWND g_hwnd;
RECT g_window_rect;

ComPtr<ID3D12Device2> g_device;
ComPtr<ID3D12CommandQueue> g_queue;
ComPtr<IDXGISwapChain4> g_swapchain;
ComPtr<ID3D12Resource> g_buffers[g_numframes];
ComPtr<ID3D12Resource> g_depthbuffer;
ComPtr<ID3D12GraphicsCommandList10> g_commandlist;
ComPtr<ID3D12CommandAllocator> g_commandallocators[g_numframes];
ComPtr<ID3D12DescriptorHeap> g_descriptorheap;
ComPtr<ID3D12PipelineState> g_pipelinestate_flat;
ComPtr<ID3D12RootSignature> g_rootsignature_flat;
ComPtr<ID3D12PipelineState> g_pipelinestate_ggx;
ComPtr<ID3D12RootSignature> g_rootsignature_ggx;
ComPtr<ID3D12PipelineState> g_pipelinestate_pbr_ntc;
ComPtr<ID3D12RootSignature> g_rootsignature_pbr_ntc;
ComPtr<ID3D12PipelineState> g_pipelinestate_pbr_ntc_coop;
ComPtr<ID3D12RootSignature> g_rootsignature_pbr_ntc_coop;
ComPtr<ID3D12Resource> g_vertex_buffer;
ComPtr<ID3D12Resource> g_index_buffer;
ComPtr<ID3D12DescriptorHeap> g_descriptorheap_dsv;
ComPtr<ID3D12DescriptorHeap> g_descriptorheap_srv;
ComPtr<ID3D12Resource> g_tex[openntc::kMaxSources];
ComPtr<ID3D12Resource> g_buffer_scratch;
DXGI_FORMAT g_formats[openntc::kMaxSources];
UINT g_descriptorsize;
UINT g_frame_i;
bool g_initialized;
bool g_contentloaded = false;
bool g_compressed_data_loaded = false;
std::atomic<SharedFields> g_shared_fields;
openntc::FileData g_fil_data;

ComPtr<ID3D12Resource> g_buffer_ntc_info;
ComPtr<ID3D12Resource> g_buffer_g0;
ComPtr<ID3D12Resource> g_buffer_g1;
ComPtr<ID3D12Resource> g_buffer_W0;
ComPtr<ID3D12Resource> g_buffer_W1;
ComPtr<ID3D12Resource> g_buffer_Wout;
ComPtr<ID3D12Resource> g_buffer_W0_scale;
ComPtr<ID3D12Resource> g_buffer_W1_scale;
ComPtr<ID3D12Resource> g_buffer_Wout_scale;

ComPtr<ID3D12Fence> g_fence;
uint64_t g_fenceval = 0;
uint64_t g_framefenceval[g_numframes] = {};
HANDLE g_fence_event;

bool g_vsync = true;
bool g_gsync = false; // Tearing enabled?
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

static const int g_index_count = 36;
WORD g_cube_indices[g_index_count] =
{
     0,  1,  2,  0,  2,  3, // front
     4,  5,  6,  4,  6,  7, // back
     8,  9, 10,  8, 10, 11, // left
    12, 13, 14, 12, 14, 15, // right
    16, 17, 18, 16, 18, 19, // top
    20, 21, 22, 20, 22, 23  // bottom
};

D3D12_VERTEX_BUFFER_VIEW g_vbv;
D3D12_INDEX_BUFFER_VIEW g_ibv;

XMMATRIX g_model_mat, g_view_mat, g_proj_mat;
float g_fov = 110.0f;

constexpr int32_t g_nonimgui_srv_count = openntc::kMaxSources + 1 + (1 + 1 + 3 + 3);
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

template <typename T>
T SrvDescriptorHead()
{
  if constexpr (std::is_same_v<T, D3D12_CPU_DESCRIPTOR_HANDLE>)
    return g_descriptorheap_srv->GetCPUDescriptorHandleForHeapStart();
  else
    return g_descriptorheap_srv->GetGPUDescriptorHandleForHeapStart();
}

template <typename T>
T SrvDescriptorForTex(int32_t tex_i)
{
  UINT inc = g_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
  T head = SrvDescriptorHead<T>();
  head.ptr += inc * tex_i;
  return head;
}

template <typename T>
T CbvDescriptorForNTCInfo()
{
  UINT inc = g_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
  T head = SrvDescriptorHead<T>();
  head.ptr += inc * openntc::kMaxSources;
  return head;
}

template <typename T>
T SrvDescriptorForNTCInfo(int offset)
{
  UINT inc = g_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
  T head = SrvDescriptorHead<T>();
  head.ptr += inc * (openntc::kMaxSources + 1 + offset);
  return head;
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

int32_t g_gui_shader = static_cast<int32_t>(Shader::GGX);
int32_t g_gui_window_shading = true;
int32_t g_gui_texture = 0;
bool g_gui_spin = true;

static inline UINT64 RoundUpTo(UINT64 a, UINT64 b)
{
  return ((a + b - 1) / b) * b;
}

void ImguiDescriptorSrvAlloc(ImGui_ImplDX12_InitInfo* init_info, D3D12_CPU_DESCRIPTOR_HANDLE* cpu_handle, D3D12_GPU_DESCRIPTOR_HANDLE* gpu_handle)
{
  UINT slot = g_imgui_available_srv_slots.back();
  g_imgui_available_srv_slots.pop_back();
  UINT size_inc = g_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
  cpu_handle->ptr = g_descriptorheap_srv->GetCPUDescriptorHandleForHeapStart().ptr + size_inc * slot;
  gpu_handle->ptr = g_descriptorheap_srv->GetGPUDescriptorHandleForHeapStart().ptr + size_inc * slot;
}

void ImguiDescriptorSrvFree(ImGui_ImplDX12_InitInfo* init_info, D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle, D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle)
{
  UINT size_inc = g_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
  UINT slot = (cpu_handle.ptr - g_descriptorheap_srv->GetCPUDescriptorHandleForHeapStart().ptr) / size_inc;
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

struct VertexDescriptor
{
  DirectX::XMFLOAT3 pos;
  DirectX::XMFLOAT3 normal;
  DirectX::XMFLOAT3 tangent;
  DirectX::XMFLOAT2 uv;
};

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

  g_device->CreateDepthStencilView(g_depthbuffer.Get(), &dsv_desc, g_descriptorheap_dsv->GetCPUDescriptorHandleForHeapStart());
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
    D3D12_RESOURCE_DESC tex_desc = {};
    tex_desc.Format = g_formats[tex_i];
    tex_desc.Alignment = 0;
    tex_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    tex_desc.Flags = D3D12_RESOURCE_FLAG_NONE;
    tex_desc.Width = access.ctx_.GetMipDim(0);
    tex_desc.Height = access.ctx_.GetMipDim(0);
    tex_desc.DepthOrArraySize = 1;
    tex_desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    tex_desc.MipLevels = tex_data.mip_count_;
    tex_desc.SampleDesc.Count = 1;
    tex_desc.SampleDesc.Quality = 0;

    D3D12_HEAP_PROPERTIES heap_props;
    heap_props.Type = D3D12_HEAP_TYPE_GPU_UPLOAD;
    heap_props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    heap_props.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heap_props.CreationNodeMask = 1;
    heap_props.VisibleNodeMask = 1;

    VERIFY(g_device->CreateCommittedResource(&heap_props, D3D12_HEAP_FLAG_NONE, &tex_desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&g_tex[tex_i])));

    D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
    srv_desc.Format = g_formats[tex_i];
    srv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv_desc.Texture2D.MipLevels = tex_data.mip_count_;
    srv_desc.Texture2D.MostDetailedMip = 0;
    srv_desc.Texture2D.PlaneSlice = 0;
    srv_desc.Texture2D.ResourceMinLODClamp = 0.0f;
    srv_desc.Shader4ComponentMapping = D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_0, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_1, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_2, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_3);

    g_device->CreateShaderResourceView(g_tex[tex_i].Get(), &srv_desc, SrvDescriptorForTex<D3D12_CPU_DESCRIPTOR_HANDLE>(tex_i));
    g_compressed_data_loaded = false;
  }

  UINT64 scratch_size = 0llu;
  for (int mip_i = 0; mip_i < tex_data.mip_count_; mip_i++)
  {
    int mip_dim = access.ctx_.GetMipDim(mip_i);
    UINT64 mip_row = RoundUpTo(sizeof(uint8_t) * mip_dim * 4, D3D12_TEXTURE_DATA_PITCH_ALIGNMENT);
    scratch_size += mip_row * mip_dim;
  }

  {
    D3D12_RESOURCE_DESC buf_desc = {};
    buf_desc.Format = DXGI_FORMAT_UNKNOWN;
    buf_desc.Alignment = 0;
    buf_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buf_desc.Flags = D3D12_RESOURCE_FLAG_NONE;
    buf_desc.Width = scratch_size;
    buf_desc.Height = 1;
    buf_desc.DepthOrArraySize = 1;
    buf_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    buf_desc.MipLevels = 1;
    buf_desc.SampleDesc.Count = 1;
    buf_desc.SampleDesc.Quality = 0;

    D3D12_HEAP_PROPERTIES heap_props;
    heap_props.Type = D3D12_HEAP_TYPE_GPU_UPLOAD;
    heap_props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    heap_props.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heap_props.CreationNodeMask = 1;
    heap_props.VisibleNodeMask = 1;

    VERIFY(g_device->CreateCommittedResource(&heap_props, D3D12_HEAP_FLAG_NONE, &buf_desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&g_buffer_scratch)));
  }

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
    rbar.Transition.StateAfter = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    rbar.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    g_commandlist->ResourceBarrier(1, &rbar);

    g_buffer_scratch->Unmap(0, nullptr);
    g_commandlist->Close();
    ID3D12CommandList* lists[] = { g_commandlist.Get() };
    g_queue->ExecuteCommandLists(1, lists);
    Flush(g_queue, g_fence, &g_fenceval, g_fence_event);
  }
}

void LoadContent()
{
  static const uint32_t vertex_count = 24;
  VertexDescriptor cube[vertex_count] = {
    // front (z = -1), N = -z, T = +x
    { XMFLOAT3(-1.0f, -1.0f, -1.0f), XMFLOAT3( 0.0f,  0.0f, -1.0f), XMFLOAT3( 1.0f,  0.0f,  0.0f), XMFLOAT2(0.0f, 1.0f) }, // 0
    { XMFLOAT3(-1.0f,  1.0f, -1.0f), XMFLOAT3( 0.0f,  0.0f, -1.0f), XMFLOAT3( 1.0f,  0.0f,  0.0f), XMFLOAT2(0.0f, 0.0f) }, // 1
    { XMFLOAT3( 1.0f,  1.0f, -1.0f), XMFLOAT3( 0.0f,  0.0f, -1.0f), XMFLOAT3( 1.0f,  0.0f,  0.0f), XMFLOAT2(1.0f, 0.0f) }, // 2
    { XMFLOAT3( 1.0f, -1.0f, -1.0f), XMFLOAT3( 0.0f,  0.0f, -1.0f), XMFLOAT3( 1.0f,  0.0f,  0.0f), XMFLOAT2(1.0f, 1.0f) }, // 3
    // back (z = +1), N = +z, T = -x
    { XMFLOAT3( 1.0f, -1.0f,  1.0f), XMFLOAT3( 0.0f,  0.0f,  1.0f), XMFLOAT3(-1.0f,  0.0f,  0.0f), XMFLOAT2(0.0f, 1.0f) }, // 4
    { XMFLOAT3( 1.0f,  1.0f,  1.0f), XMFLOAT3( 0.0f,  0.0f,  1.0f), XMFLOAT3(-1.0f,  0.0f,  0.0f), XMFLOAT2(0.0f, 0.0f) }, // 5
    { XMFLOAT3(-1.0f,  1.0f,  1.0f), XMFLOAT3( 0.0f,  0.0f,  1.0f), XMFLOAT3(-1.0f,  0.0f,  0.0f), XMFLOAT2(1.0f, 0.0f) }, // 6
    { XMFLOAT3(-1.0f, -1.0f,  1.0f), XMFLOAT3( 0.0f,  0.0f,  1.0f), XMFLOAT3(-1.0f,  0.0f,  0.0f), XMFLOAT2(1.0f, 1.0f) }, // 7
    // left (x = -1), N = -x, T = -z
    { XMFLOAT3(-1.0f, -1.0f,  1.0f), XMFLOAT3(-1.0f,  0.0f,  0.0f), XMFLOAT3( 0.0f,  0.0f, -1.0f), XMFLOAT2(0.0f, 1.0f) }, // 8
    { XMFLOAT3(-1.0f,  1.0f,  1.0f), XMFLOAT3(-1.0f,  0.0f,  0.0f), XMFLOAT3( 0.0f,  0.0f, -1.0f), XMFLOAT2(0.0f, 0.0f) }, // 9
    { XMFLOAT3(-1.0f,  1.0f, -1.0f), XMFLOAT3(-1.0f,  0.0f,  0.0f), XMFLOAT3( 0.0f,  0.0f, -1.0f), XMFLOAT2(1.0f, 0.0f) }, // 10
    { XMFLOAT3(-1.0f, -1.0f, -1.0f), XMFLOAT3(-1.0f,  0.0f,  0.0f), XMFLOAT3( 0.0f,  0.0f, -1.0f), XMFLOAT2(1.0f, 1.0f) }, // 11
    // right (x = +1), N = +x, T = +z
    { XMFLOAT3( 1.0f, -1.0f, -1.0f), XMFLOAT3( 1.0f,  0.0f,  0.0f), XMFLOAT3( 0.0f,  0.0f,  1.0f), XMFLOAT2(0.0f, 1.0f) }, // 12
    { XMFLOAT3( 1.0f,  1.0f, -1.0f), XMFLOAT3( 1.0f,  0.0f,  0.0f), XMFLOAT3( 0.0f,  0.0f,  1.0f), XMFLOAT2(0.0f, 0.0f) }, // 13
    { XMFLOAT3( 1.0f,  1.0f,  1.0f), XMFLOAT3( 1.0f,  0.0f,  0.0f), XMFLOAT3( 0.0f,  0.0f,  1.0f), XMFLOAT2(1.0f, 0.0f) }, // 14
    { XMFLOAT3( 1.0f, -1.0f,  1.0f), XMFLOAT3( 1.0f,  0.0f,  0.0f), XMFLOAT3( 0.0f,  0.0f,  1.0f), XMFLOAT2(1.0f, 1.0f) }, // 15
    // top (y = +1), N = +y, T = +x
    { XMFLOAT3(-1.0f,  1.0f, -1.0f), XMFLOAT3( 0.0f,  1.0f,  0.0f), XMFLOAT3( 1.0f,  0.0f,  0.0f), XMFLOAT2(0.0f, 1.0f) }, // 16
    { XMFLOAT3(-1.0f,  1.0f,  1.0f), XMFLOAT3( 0.0f,  1.0f,  0.0f), XMFLOAT3( 1.0f,  0.0f,  0.0f), XMFLOAT2(0.0f, 0.0f) }, // 17
    { XMFLOAT3( 1.0f,  1.0f,  1.0f), XMFLOAT3( 0.0f,  1.0f,  0.0f), XMFLOAT3( 1.0f,  0.0f,  0.0f), XMFLOAT2(1.0f, 0.0f) }, // 18
    { XMFLOAT3( 1.0f,  1.0f, -1.0f), XMFLOAT3( 0.0f,  1.0f,  0.0f), XMFLOAT3( 1.0f,  0.0f,  0.0f), XMFLOAT2(1.0f, 1.0f) }, // 19
    // bottom (y = -1), N = -y, T = +x
    { XMFLOAT3(-1.0f, -1.0f,  1.0f), XMFLOAT3( 0.0f, -1.0f,  0.0f), XMFLOAT3( 1.0f,  0.0f,  0.0f), XMFLOAT2(0.0f, 1.0f) }, // 20
    { XMFLOAT3(-1.0f, -1.0f, -1.0f), XMFLOAT3( 0.0f, -1.0f,  0.0f), XMFLOAT3( 1.0f,  0.0f,  0.0f), XMFLOAT2(0.0f, 0.0f) }, // 21
    { XMFLOAT3( 1.0f, -1.0f, -1.0f), XMFLOAT3( 0.0f, -1.0f,  0.0f), XMFLOAT3( 1.0f,  0.0f,  0.0f), XMFLOAT2(1.0f, 0.0f) }, // 22
    { XMFLOAT3( 1.0f, -1.0f,  1.0f), XMFLOAT3( 0.0f, -1.0f,  0.0f), XMFLOAT3( 1.0f,  0.0f,  0.0f), XMFLOAT2(1.0f, 1.0f) }  // 23
  };

  {
    D3D12_FEATURE_DATA_D3D12_OPTIONS16 options16 = {};
    VERIFY(g_device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS16, &options16, sizeof(options16)));
    assert(options16.GPUUploadHeapSupported);

    D3D12_HEAP_PROPERTIES hprops = {};
    hprops.Type = D3D12_HEAP_TYPE_GPU_UPLOAD;
    hprops.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    hprops.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    hprops.CreationNodeMask = 1;
    hprops.VisibleNodeMask = 1;

    D3D12_RESOURCE_DESC rdesc = {};
    rdesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rdesc.Alignment = 0;
    rdesc.Width = sizeof(VertexDescriptor) * vertex_count;
    rdesc.Height = 1;
    rdesc.DepthOrArraySize = 1;
    rdesc.MipLevels = 1;
    rdesc.Format = DXGI_FORMAT_UNKNOWN;
    rdesc.SampleDesc.Count = 1;
    rdesc.SampleDesc.Quality = 0;
    rdesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rdesc.Flags = D3D12_RESOURCE_FLAG_NONE;
    VERIFY(g_device->CreateCommittedResource(&hprops, D3D12_HEAP_FLAG_NONE, &rdesc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&g_vertex_buffer)));
  }

  void* mapped = nullptr;
  D3D12_RANGE read_range = {0, 0};
  VERIFY(g_vertex_buffer->Map(0, &read_range, &mapped));
  memcpy(mapped, cube, sizeof(VertexDescriptor) * vertex_count);
  g_vertex_buffer->Unmap(0, nullptr);

  {
    D3D12_HEAP_PROPERTIES hprops = {};
    hprops.Type = D3D12_HEAP_TYPE_GPU_UPLOAD;
    hprops.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    hprops.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    hprops.CreationNodeMask = 1;
    hprops.VisibleNodeMask = 1;

    D3D12_RESOURCE_DESC rdesc = {};
    rdesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rdesc.Alignment = 0;
    rdesc.Width = sizeof(WORD) * g_index_count;
    rdesc.Height = 1;
    rdesc.DepthOrArraySize = 1;
    rdesc.MipLevels = 1;
    rdesc.Format = DXGI_FORMAT_UNKNOWN;
    rdesc.SampleDesc.Count = 1;
    rdesc.SampleDesc.Quality = 0;
    rdesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rdesc.Flags = D3D12_RESOURCE_FLAG_NONE;
    g_device->CreateCommittedResource(&hprops, D3D12_HEAP_FLAG_NONE, &rdesc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&g_index_buffer));
    
    void* mapped = nullptr;
    D3D12_RANGE read_range = {0, 0};
    VERIFY(g_index_buffer->Map(0, &read_range, &mapped));
    memcpy(mapped, g_cube_indices, sizeof(WORD) * g_index_count);
    g_index_buffer->Unmap(0, nullptr);
  }

  g_vbv.BufferLocation = g_vertex_buffer->GetGPUVirtualAddress();
  g_vbv.SizeInBytes = sizeof(VertexDescriptor) * vertex_count;
  g_vbv.StrideInBytes = sizeof(VertexDescriptor);

  g_ibv.BufferLocation = g_index_buffer->GetGPUVirtualAddress();
  g_ibv.SizeInBytes = sizeof(WORD) * g_index_count;
  g_ibv.Format = DXGI_FORMAT_R16_UINT;

  D3D12_DESCRIPTOR_HEAP_DESC dsv_heap_desc = {};
  dsv_heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
  dsv_heap_desc.NumDescriptors = 1;
  dsv_heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
  dsv_heap_desc.NodeMask = 0;
  VERIFY(g_device->CreateDescriptorHeap(&dsv_heap_desc, IID_PPV_ARGS(&g_descriptorheap_dsv)));

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
    sampler_desc.Filter = D3D12_FILTER_MIN_MAG_LINEAR_MIP_POINT;
    sampler_desc.MinLOD = 0.0f;
    sampler_desc.MaxLOD = 1.0f;
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

    D3D12_ROOT_PARAMETER1 root_parameters[2];
    root_parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    root_parameters[0].Constants.Num32BitValues = sizeof(ModelViewProjection) / 4;
    root_parameters[0].Constants.RegisterSpace = 0;
    root_parameters[0].Constants.ShaderRegister = 0;
    root_parameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    root_parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    root_parameters[1].DescriptorTable.NumDescriptorRanges = 1;
    root_parameters[1].DescriptorTable.pDescriptorRanges = &drange;
    root_parameters[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

    D3D12_STATIC_SAMPLER_DESC sampler_desc = {};
    sampler_desc.AddressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    sampler_desc.AddressV = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    sampler_desc.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    sampler_desc.Filter = D3D12_FILTER_MIN_MAG_LINEAR_MIP_POINT;
    sampler_desc.MinLOD = 0.0f;
    sampler_desc.MaxLOD = 1.0f;
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

    D3D12_ROOT_PARAMETER1 root_parameters[3];
    root_parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    root_parameters[0].Constants.Num32BitValues = sizeof(ModelViewProjection) / 4;
    root_parameters[0].Constants.RegisterSpace = 0;
    root_parameters[0].Constants.ShaderRegister = 0;
    root_parameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;
    root_parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    root_parameters[1].DescriptorTable.NumDescriptorRanges = 1;
    root_parameters[1].DescriptorTable.pDescriptorRanges = &drange_cbv;
    root_parameters[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    root_parameters[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    root_parameters[2].DescriptorTable.NumDescriptorRanges = 1;
    root_parameters[2].DescriptorTable.pDescriptorRanges = &drange_srv;
    root_parameters[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

    D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_signature_desc = {};
    root_signature_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
    root_signature_desc.Desc_1_1.NumParameters = 3;
    root_signature_desc.Desc_1_1.pParameters = root_parameters;
    root_signature_desc.Desc_1_1.NumStaticSamplers = 0;
    root_signature_desc.Desc_1_1.pStaticSamplers = nullptr;
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

    D3D12_ROOT_PARAMETER1 root_parameters[3];
    root_parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    root_parameters[0].Constants.Num32BitValues = sizeof(ModelViewProjection) / 4;
    root_parameters[0].Constants.RegisterSpace = 0;
    root_parameters[0].Constants.ShaderRegister = 0;
    root_parameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;
    root_parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    root_parameters[1].DescriptorTable.NumDescriptorRanges = 1;
    root_parameters[1].DescriptorTable.pDescriptorRanges = &drange_cbv;
    root_parameters[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    root_parameters[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    root_parameters[2].DescriptorTable.NumDescriptorRanges = 1;
    root_parameters[2].DescriptorTable.pDescriptorRanges = &drange_srv;
    root_parameters[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

    D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_signature_desc = {};
    root_signature_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
    root_signature_desc.Desc_1_1.NumParameters = 3;
    root_signature_desc.Desc_1_1.pParameters = root_parameters;
    root_signature_desc.Desc_1_1.NumStaticSamplers = 0;
    root_signature_desc.Desc_1_1.pStaticSamplers = nullptr;
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

  // Texture

  D3D12_DESCRIPTOR_HEAP_DESC srv_heap_desc = {};
  srv_heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
  srv_heap_desc.NumDescriptors = g_srv_count;
  srv_heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
  srv_heap_desc.NodeMask = 0;
  VERIFY(g_device->CreateDescriptorHeap(&srv_heap_desc, IID_PPV_ARGS(&g_descriptorheap_srv)));
  D3D12_CPU_DESCRIPTOR_HANDLE srv_handle_head = g_descriptorheap_srv->GetCPUDescriptorHandleForHeapStart();
  UINT srv_descriptor_size = g_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

  {
    SharedContext::Access access = g_ctx.Acquire();

    openntc::Result load_res = access.ctx_.LoadManifest("C:/Code/openntc/img/Bricks101_2K-JPG/manifest.json");
    VERIFY(load_res == openntc::Result::Success);
    
    RebuildTextureResources(access);
  }

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

    D3D12_RESOURCE_DESC buf_desc = {};
    buf_desc.Format = DXGI_FORMAT_UNKNOWN;
    buf_desc.Alignment = 0;
    buf_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buf_desc.Flags = D3D12_RESOURCE_FLAG_NONE;
    buf_desc.Width = (sizeof(NTCInfo) + 255) & (~255);
    buf_desc.Height = 1;
    buf_desc.DepthOrArraySize = 1;
    buf_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    buf_desc.MipLevels = 1;
    buf_desc.SampleDesc.Count = 1;
    buf_desc.SampleDesc.Quality = 0;

    D3D12_HEAP_PROPERTIES heap_props;
    heap_props.Type = D3D12_HEAP_TYPE_GPU_UPLOAD;
    heap_props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    heap_props.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heap_props.CreationNodeMask = 1;
    heap_props.VisibleNodeMask = 1;

    VERIFY(g_device->CreateCommittedResource(&heap_props, D3D12_HEAP_FLAG_NONE, &buf_desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&g_buffer_ntc_info)));

    D3D12_CONSTANT_BUFFER_VIEW_DESC cbv_desc = {};
    cbv_desc.BufferLocation = g_buffer_ntc_info->GetGPUVirtualAddress();
    cbv_desc.SizeInBytes = (sizeof(NTCInfo) + 255) & (~255);

    g_device->CreateConstantBufferView(&cbv_desc, CbvDescriptorForNTCInfo<D3D12_CPU_DESCRIPTOR_HANDLE>());
    
    void* mapped = nullptr;
    D3D12_RANGE read_range = {0, 0};
    g_buffer_ntc_info->Map(0, &read_range, &mapped);
    memcpy(mapped, &ntc_info, sizeof(NTCInfo));
    g_buffer_ntc_info->Unmap(0, &read_range);
  }

  uint64_t g0_size = 0;
  uint64_t g1_size = 0;
  for (int level_i = 0; level_i < cdata.level_count_; level_i++)
  {
    g0_size += cdata.g0_size_[level_i];
    g1_size += cdata.g1_size_[level_i];
  }

  {
    D3D12_RESOURCE_DESC buf_desc = {};
    buf_desc.Format = DXGI_FORMAT_UNKNOWN;
    buf_desc.Alignment = 0;
    buf_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buf_desc.Flags = D3D12_RESOURCE_FLAG_NONE;
    buf_desc.Width = g0_size;
    buf_desc.Height = 1;
    buf_desc.DepthOrArraySize = 1;
    buf_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    buf_desc.MipLevels = 1;
    buf_desc.SampleDesc.Count = 1;
    buf_desc.SampleDesc.Quality = 0;

    D3D12_HEAP_PROPERTIES heap_props;
    heap_props.Type = D3D12_HEAP_TYPE_GPU_UPLOAD;
    heap_props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    heap_props.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heap_props.CreationNodeMask = 1;
    heap_props.VisibleNodeMask = 1;

    VERIFY(g_device->CreateCommittedResource(&heap_props, D3D12_HEAP_FLAG_NONE, &buf_desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&g_buffer_g0)));
    
    D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
    srv_desc.Format = DXGI_FORMAT_R32_UINT;
    srv_desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srv_desc.Buffer.FirstElement = 0;
    srv_desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
    srv_desc.Buffer.NumElements = g0_size / sizeof(uint32_t);
    srv_desc.Buffer.StructureByteStride = 0;
    srv_desc.Shader4ComponentMapping = D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_0, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_1, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_2, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_3);

    g_device->CreateShaderResourceView(g_buffer_g0.Get(), &srv_desc, SrvDescriptorForNTCInfo<D3D12_CPU_DESCRIPTOR_HANDLE>(0));
    
    void* mapped = nullptr;
    D3D12_RANGE read_range = {0, 0};
    g_buffer_g0->Map(0, &read_range, &mapped);
    for (int level_i = 0; level_i < cdata.level_count_; level_i++)
      memcpy(static_cast<char*>(mapped) + cdata.g0_offset_[level_i], cdata.g0_[level_i], cdata.g0_size_[level_i]);
    g_buffer_g0->Unmap(0, nullptr);
  }

  {
    D3D12_RESOURCE_DESC buf_desc = {};
    buf_desc.Format = DXGI_FORMAT_UNKNOWN;
    buf_desc.Alignment = 0;
    buf_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buf_desc.Flags = D3D12_RESOURCE_FLAG_NONE;
    buf_desc.Width = g1_size;
    buf_desc.Height = 1;
    buf_desc.DepthOrArraySize = 1;
    buf_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    buf_desc.MipLevels = 1;
    buf_desc.SampleDesc.Count = 1;
    buf_desc.SampleDesc.Quality = 0;

    D3D12_HEAP_PROPERTIES heap_props;
    heap_props.Type = D3D12_HEAP_TYPE_GPU_UPLOAD;
    heap_props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    heap_props.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heap_props.CreationNodeMask = 1;
    heap_props.VisibleNodeMask = 1;

    VERIFY(g_device->CreateCommittedResource(&heap_props, D3D12_HEAP_FLAG_NONE, &buf_desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&g_buffer_g1)));
    
    D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
    srv_desc.Format = DXGI_FORMAT_R32_UINT;
    srv_desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srv_desc.Buffer.FirstElement = 0;
    srv_desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
    srv_desc.Buffer.NumElements = g1_size / sizeof(uint32_t);
    srv_desc.Buffer.StructureByteStride = 0;
    srv_desc.Shader4ComponentMapping = D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_0, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_1, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_2, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_3);

    g_device->CreateShaderResourceView(g_buffer_g1.Get(), &srv_desc, SrvDescriptorForNTCInfo<D3D12_CPU_DESCRIPTOR_HANDLE>(1));
    
    void* mapped = nullptr;
    D3D12_RANGE read_range = {0, 0};
    g_buffer_g1->Map(0, &read_range, &mapped);
    for (int level_i = 0; level_i < cdata.level_count_; level_i++)
      memcpy(static_cast<char*>(mapped) + cdata.g1_offset_[level_i], cdata.g1_[level_i], cdata.g1_size_[level_i]);
    g_buffer_g1->Unmap(0, nullptr);
  }

  {
    D3D12_RESOURCE_DESC buf_desc = {};
    buf_desc.Format = DXGI_FORMAT_UNKNOWN;
    buf_desc.Alignment = 0;
    buf_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buf_desc.Flags = D3D12_RESOURCE_FLAG_NONE;
    buf_desc.Width = cdata.W0_size_;
    buf_desc.Height = 1;
    buf_desc.DepthOrArraySize = 1;
    buf_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    buf_desc.MipLevels = 1;
    buf_desc.SampleDesc.Count = 1;
    buf_desc.SampleDesc.Quality = 0;

    D3D12_HEAP_PROPERTIES heap_props;
    heap_props.Type = D3D12_HEAP_TYPE_GPU_UPLOAD;
    heap_props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    heap_props.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heap_props.CreationNodeMask = 1;
    heap_props.VisibleNodeMask = 1;

    VERIFY(g_device->CreateCommittedResource(&heap_props, D3D12_HEAP_FLAG_NONE, &buf_desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&g_buffer_W0)));
    
    D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
    srv_desc.Format = DXGI_FORMAT_R32_TYPELESS;
    srv_desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srv_desc.Buffer.FirstElement = 0;
    srv_desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
    srv_desc.Buffer.NumElements = cdata.W0_size_ / (sizeof(uint32_t));
    srv_desc.Buffer.StructureByteStride = 0;
    srv_desc.Shader4ComponentMapping = D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_0, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_1, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_2, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_3);

    g_device->CreateShaderResourceView(g_buffer_W0.Get(), &srv_desc, SrvDescriptorForNTCInfo<D3D12_CPU_DESCRIPTOR_HANDLE>(2));

    void* mapped = nullptr;
    D3D12_RANGE read_range = {0, 0};
    g_buffer_W0->Map(0, &read_range, &mapped);
    memcpy(mapped, cdata.W0_, cdata.W0_size_);
    g_buffer_W0->Unmap(0, &read_range);
  }

  {
    D3D12_RESOURCE_DESC buf_desc = {};
    buf_desc.Format = DXGI_FORMAT_UNKNOWN;
    buf_desc.Alignment = 0;
    buf_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buf_desc.Flags = D3D12_RESOURCE_FLAG_NONE;
    buf_desc.Width = cdata.W1_size_;
    buf_desc.Height = 1;
    buf_desc.DepthOrArraySize = 1;
    buf_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    buf_desc.MipLevels = 1;
    buf_desc.SampleDesc.Count = 1;
    buf_desc.SampleDesc.Quality = 0;

    D3D12_HEAP_PROPERTIES heap_props;
    heap_props.Type = D3D12_HEAP_TYPE_GPU_UPLOAD;
    heap_props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    heap_props.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heap_props.CreationNodeMask = 1;
    heap_props.VisibleNodeMask = 1;

    VERIFY(g_device->CreateCommittedResource(&heap_props, D3D12_HEAP_FLAG_NONE, &buf_desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&g_buffer_W1)));
    
    D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
    srv_desc.Format = DXGI_FORMAT_R32_TYPELESS;
    srv_desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srv_desc.Buffer.FirstElement = 0;
    srv_desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
    srv_desc.Buffer.NumElements = cdata.W1_size_ / (sizeof(uint32_t));
    srv_desc.Buffer.StructureByteStride = 0;
    srv_desc.Shader4ComponentMapping = D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_0, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_1, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_2, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_3);


    g_device->CreateShaderResourceView(g_buffer_W1.Get(), &srv_desc, SrvDescriptorForNTCInfo<D3D12_CPU_DESCRIPTOR_HANDLE>(3));

    void* mapped = nullptr;
    D3D12_RANGE read_range = {0, 0};
    g_buffer_W1->Map(0, &read_range, &mapped);
    memcpy(mapped, cdata.W1_, cdata.W1_size_);
    g_buffer_W1->Unmap(0, &read_range);
  }

  {
    D3D12_RESOURCE_DESC buf_desc = {};
    buf_desc.Format = DXGI_FORMAT_UNKNOWN;
    buf_desc.Alignment = 0;
    buf_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buf_desc.Flags = D3D12_RESOURCE_FLAG_NONE;
    buf_desc.Width = cdata.Wout_size_;
    buf_desc.Height = 1;
    buf_desc.DepthOrArraySize = 1;
    buf_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    buf_desc.MipLevels = 1;
    buf_desc.SampleDesc.Count = 1;
    buf_desc.SampleDesc.Quality = 0;

    D3D12_HEAP_PROPERTIES heap_props;
    heap_props.Type = D3D12_HEAP_TYPE_GPU_UPLOAD;
    heap_props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    heap_props.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heap_props.CreationNodeMask = 1;
    heap_props.VisibleNodeMask = 1;

    VERIFY(g_device->CreateCommittedResource(&heap_props, D3D12_HEAP_FLAG_NONE, &buf_desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&g_buffer_Wout)));
    
    D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
    srv_desc.Format = DXGI_FORMAT_R32_TYPELESS;
    srv_desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srv_desc.Buffer.FirstElement = 0;
    srv_desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
    srv_desc.Buffer.NumElements = cdata.Wout_size_ / (sizeof(uint32_t));
    srv_desc.Buffer.StructureByteStride = 0;
    srv_desc.Shader4ComponentMapping = D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_0, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_1, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_2, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_3);

    g_device->CreateShaderResourceView(g_buffer_Wout.Get(), &srv_desc, SrvDescriptorForNTCInfo<D3D12_CPU_DESCRIPTOR_HANDLE>(4));
    
    void* mapped = nullptr;
    D3D12_RANGE read_range = {0, 0};
    g_buffer_Wout->Map(0, &read_range, &mapped);
    memcpy(mapped, cdata.Wout_, cdata.Wout_size_);
    g_buffer_Wout->Unmap(0, &read_range);
  }

  {
    D3D12_RESOURCE_DESC buf_desc = {};
    buf_desc.Format = DXGI_FORMAT_UNKNOWN;
    buf_desc.Alignment = 0;
    buf_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buf_desc.Flags = D3D12_RESOURCE_FLAG_NONE;
    buf_desc.Width = cdata.W0_scale_size_;
    buf_desc.Height = 1;
    buf_desc.DepthOrArraySize = 1;
    buf_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    buf_desc.MipLevels = 1;
    buf_desc.SampleDesc.Count = 1;
    buf_desc.SampleDesc.Quality = 0;

    D3D12_HEAP_PROPERTIES heap_props;
    heap_props.Type = D3D12_HEAP_TYPE_GPU_UPLOAD;
    heap_props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    heap_props.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heap_props.CreationNodeMask = 1;
    heap_props.VisibleNodeMask = 1;

    VERIFY(g_device->CreateCommittedResource(&heap_props, D3D12_HEAP_FLAG_NONE, &buf_desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&g_buffer_W0_scale)));
    
    D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
    srv_desc.Format = DXGI_FORMAT_R32_TYPELESS;
    srv_desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srv_desc.Buffer.FirstElement = 0;
    srv_desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
    srv_desc.Buffer.NumElements = cdata.W0_scale_size_ / (sizeof(float));
    srv_desc.Buffer.StructureByteStride = 0;
    srv_desc.Shader4ComponentMapping = D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_0, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_1, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_2, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_3);

    g_device->CreateShaderResourceView(g_buffer_W0_scale.Get(), &srv_desc, SrvDescriptorForNTCInfo<D3D12_CPU_DESCRIPTOR_HANDLE>(5));
    
    void* mapped = nullptr;
    D3D12_RANGE read_range = {0, 0};
    g_buffer_W0_scale->Map(0, &read_range, &mapped);
    memcpy(mapped, cdata.W0_scale_, cdata.W0_scale_size_);
    g_buffer_W0_scale->Unmap(0, &read_range);
  }

  {
    D3D12_RESOURCE_DESC buf_desc = {};
    buf_desc.Format = DXGI_FORMAT_UNKNOWN;
    buf_desc.Alignment = 0;
    buf_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buf_desc.Flags = D3D12_RESOURCE_FLAG_NONE;
    buf_desc.Width = cdata.W1_scale_size_;
    buf_desc.Height = 1;
    buf_desc.DepthOrArraySize = 1;
    buf_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    buf_desc.MipLevels = 1;
    buf_desc.SampleDesc.Count = 1;
    buf_desc.SampleDesc.Quality = 0;

    D3D12_HEAP_PROPERTIES heap_props;
    heap_props.Type = D3D12_HEAP_TYPE_GPU_UPLOAD;
    heap_props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    heap_props.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heap_props.CreationNodeMask = 1;
    heap_props.VisibleNodeMask = 1;

    VERIFY(g_device->CreateCommittedResource(&heap_props, D3D12_HEAP_FLAG_NONE, &buf_desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&g_buffer_W1_scale)));
    
    D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
    srv_desc.Format = DXGI_FORMAT_R32_TYPELESS;
    srv_desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srv_desc.Buffer.FirstElement = 0;
    srv_desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
    srv_desc.Buffer.NumElements = cdata.W1_scale_size_ / (sizeof(float));
    srv_desc.Buffer.StructureByteStride = 0;
    srv_desc.Shader4ComponentMapping = D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_0, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_1, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_2, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_3);

    g_device->CreateShaderResourceView(g_buffer_W1_scale.Get(), &srv_desc, SrvDescriptorForNTCInfo<D3D12_CPU_DESCRIPTOR_HANDLE>(6));
    
    void* mapped = nullptr;
    D3D12_RANGE read_range = {0, 0};
    g_buffer_W1_scale->Map(0, &read_range, &mapped);
    memcpy(mapped, cdata.W1_scale_, cdata.W1_scale_size_);
    g_buffer_W1_scale->Unmap(0, &read_range);
  }

  {
    D3D12_RESOURCE_DESC buf_desc = {};
    buf_desc.Format = DXGI_FORMAT_UNKNOWN;
    buf_desc.Alignment = 0;
    buf_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buf_desc.Flags = D3D12_RESOURCE_FLAG_NONE;
    buf_desc.Width = cdata.Wout_scale_size_;
    buf_desc.Height = 1;
    buf_desc.DepthOrArraySize = 1;
    buf_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    buf_desc.MipLevels = 1;
    buf_desc.SampleDesc.Count = 1;
    buf_desc.SampleDesc.Quality = 0;

    D3D12_HEAP_PROPERTIES heap_props;
    heap_props.Type = D3D12_HEAP_TYPE_GPU_UPLOAD;
    heap_props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    heap_props.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heap_props.CreationNodeMask = 1;
    heap_props.VisibleNodeMask = 1;

    VERIFY(g_device->CreateCommittedResource(&heap_props, D3D12_HEAP_FLAG_NONE, &buf_desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&g_buffer_Wout_scale)));
    
    D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
    srv_desc.Format = DXGI_FORMAT_R32_TYPELESS;
    srv_desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srv_desc.Buffer.FirstElement = 0;
    srv_desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
    srv_desc.Buffer.NumElements = cdata.Wout_scale_size_ / (sizeof(float));
    srv_desc.Buffer.StructureByteStride = 0;
    srv_desc.Shader4ComponentMapping = D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_0, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_1, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_2, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_3);

    g_device->CreateShaderResourceView(g_buffer_Wout_scale.Get(), &srv_desc, SrvDescriptorForNTCInfo<D3D12_CPU_DESCRIPTOR_HANDLE>(7));
    
    void* mapped = nullptr;
    D3D12_RANGE read_range = {0, 0};
    g_buffer_Wout_scale->Map(0, &read_range, &mapped);
    memcpy(mapped, cdata.Wout_scale_, cdata.Wout_scale_size_);
    g_buffer_Wout_scale->Unmap(0, &read_range);
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

void Update()
{
  static uint64_t framecounter = 0;
  static double elapsed_seconds = 0.0;
  static double total_seconds = 0.0;
  static std::chrono::high_resolution_clock clock;
  static auto t0 = clock.now();

  framecounter++;
  auto t1 = clock.now();
  std::chrono::duration<double> dT = t1 - t0;
  t0 = t1;

  elapsed_seconds += dT.count();
  total_seconds += dT.count();
  if (elapsed_seconds > 1.0)
  {
    char buffer[500];
    auto fps = framecounter / elapsed_seconds;
    sprintf_s(buffer, 500, "FPS: %f\n", fps);
    OutputDebugStringA(buffer);
    framecounter = 0;
    elapsed_seconds = 0.0;
  }

  float angle = g_gui_spin ? static_cast<float>(std::fmod(total_seconds, std::acos(-1.0) * 2.0)) : 0.0f;
  const XMVECTOR rotation_axis = XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f);
  g_model_mat = XMMatrixRotationAxis(rotation_axis, angle);

  const XMVECTOR eye_pos = XMVectorSet(0.0f, 0.0f, -2.0f, 1.0f);
  const XMVECTOR focus_pos = XMVectorSet(0.0f, 0.0f, 0.0f, 1.0f);
  const XMVECTOR up_dir = XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f);
  g_view_mat = XMMatrixLookAtLH(eye_pos, focus_pos, up_dir);

  float aspect_ratio = g_width / (2.0f * static_cast<float>(g_height));
  g_proj_mat = XMMatrixPerspectiveFovLH(XMConvertToRadians(g_fov), aspect_ratio, 0.1f, 100.0f);
}

void Render()
{
  ImGui_ImplDX12_NewFrame();
  ImGui_ImplWin32_NewFrame();
  ImGui::NewFrame();
  g_gui_window_shading = ImGui::Begin("Shading");
  ImGui::Combo("Shader", &g_gui_shader, g_map_shader_to_name, static_cast<int32_t>(Shader::Count));
  Shader shader = static_cast<Shader>(g_gui_shader);
  if (shader == Shader::Flat)
  {
    ImGui::Combo("Channel", &g_gui_texture, g_map_semantic_to_name, openntc::kMaxSources);
  }
  ImGui::Checkbox("Spin", &g_gui_spin);
  ImGui::End();

  ImGui::Begin("Train");
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
  
  if (ImGui::Button("Load"))
  {
    LoadCompressedDataFromFile();
  }
  ImGui::End();
  ImGui::Begin("Load");
  if (ImGui::Button("Load Manifest"))
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
  ImGui::End();

  auto command_allocator = g_commandallocators[g_frame_i];
  auto buffer = g_buffers[g_frame_i];
  command_allocator->Reset();
  g_commandlist->Reset(command_allocator.Get(), nullptr);
  D3D12_CPU_DESCRIPTOR_HANDLE rtv_handle = g_descriptorheap->GetCPUDescriptorHandleForHeapStart();
  rtv_handle.ptr += g_frame_i * g_descriptorsize;
  D3D12_CPU_DESCRIPTOR_HANDLE dsv_handle = g_descriptorheap_dsv->GetCPUDescriptorHandleForHeapStart();
  UINT tex_color_size = g_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

  D3D12_VIEWPORT viewport = {};
  viewport.TopLeftX = 0.0f;
  viewport.TopLeftY = 0.0f;
  viewport.Width = static_cast<float>(g_width) / 2.0f;
  viewport.Height = static_cast<float>(g_height);
  viewport.MinDepth = 0.0f;
  viewport.MaxDepth = 1.0f;

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
  ID3D12DescriptorHeap* heaps[1] = {g_descriptorheap_srv.Get()};
  g_commandlist->SetDescriptorHeaps(_countof(heaps), heaps);

  g_commandlist->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
  g_commandlist->IASetVertexBuffers(0, 1, &g_vbv);
  g_commandlist->IASetIndexBuffer(&g_ibv);

  g_commandlist->RSSetViewports(1, &viewport);
  g_commandlist->RSSetScissorRects(1, &scissor);
  g_commandlist->OMSetRenderTargets(1, &rtv_handle, FALSE, &dsv_handle);

  if (shader == Shader::Flat)
  {
    XMMATRIX mvp_mat = XMMatrixMultiply(g_model_mat, g_view_mat);
    mvp_mat = XMMatrixMultiply(mvp_mat, g_proj_mat);
    D3D12_GPU_DESCRIPTOR_HANDLE tex_color_handle = SrvDescriptorForTex<D3D12_GPU_DESCRIPTOR_HANDLE>(g_gui_texture);
    g_commandlist->SetGraphicsRoot32BitConstants(0, sizeof(XMMATRIX) / 4, &mvp_mat, 0);
    g_commandlist->SetGraphicsRootDescriptorTable(1, tex_color_handle);
  }
  else if (shader == Shader::GGX)
  {
    ModelViewProjection mvp = {};
    mvp.model_to_world = g_model_mat;
    mvp.world_to_view = g_view_mat;
    mvp.view_to_proj = g_proj_mat;
    D3D12_GPU_DESCRIPTOR_HANDLE tex_color_handle = SrvDescriptorForTex<D3D12_GPU_DESCRIPTOR_HANDLE>(0);
    g_commandlist->SetGraphicsRoot32BitConstants(0, sizeof(ModelViewProjection) / 4, &mvp, 0);
    g_commandlist->SetGraphicsRootDescriptorTable(1, tex_color_handle);
  }
  else if (shader == Shader::PBR_NTC || shader == Shader::PBR_NTC_COOP)
  {
    ModelViewProjection mvp = {};
    mvp.model_to_world = g_model_mat;
    mvp.world_to_view = g_view_mat;
    mvp.view_to_proj = g_proj_mat;
    D3D12_GPU_DESCRIPTOR_HANDLE cbv_handle = CbvDescriptorForNTCInfo<D3D12_GPU_DESCRIPTOR_HANDLE>();
    D3D12_GPU_DESCRIPTOR_HANDLE srv_handle = SrvDescriptorForNTCInfo<D3D12_GPU_DESCRIPTOR_HANDLE>(0);
    g_commandlist->SetGraphicsRoot32BitConstants(0, sizeof(ModelViewProjection) / 4, &mvp, 0);
    g_commandlist->SetGraphicsRootDescriptorTable(1, cbv_handle);
    g_commandlist->SetGraphicsRootDescriptorTable(2, srv_handle);
  }
  g_commandlist->DrawIndexedInstanced(_countof(g_cube_indices), 1, 0, 0, 0);

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
    UpdateRenderTargetViews(g_device, g_swapchain, g_descriptorheap);
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

ComPtr<ID3D12DescriptorHeap> CreateDescriptorHeap(ComPtr<ID3D12Device2> device, D3D12_DESCRIPTOR_HEAP_TYPE type, uint32_t num_descriptors)
{
  ComPtr<ID3D12DescriptorHeap> descriptor_heap;

  D3D12_DESCRIPTOR_HEAP_DESC desc = {};
  desc.NumDescriptors = num_descriptors;
  desc.Type = type;

  VERIFY(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&descriptor_heap)));
  return descriptor_heap;
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
  g_queue = CreateCommandQueue(g_device, D3D12_COMMAND_LIST_TYPE_DIRECT);
  g_swapchain = CreateSwapChain(g_hwnd, g_queue, g_width, g_height, g_numframes);
  g_frame_i = g_swapchain->GetCurrentBackBufferIndex();
  g_descriptorheap = CreateDescriptorHeap(g_device, D3D12_DESCRIPTOR_HEAP_TYPE_RTV, g_numframes);
  g_descriptorsize = g_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
  UpdateRenderTargetViews(g_device, g_swapchain, g_descriptorheap);

  for (int i = 0; i < g_numframes; i++)
  {
    g_commandallocators[i] = CreateCommandAllocator(g_device, D3D12_COMMAND_LIST_TYPE_DIRECT);
  }
  g_commandlist = CreateCommandList(g_device, g_commandallocators[g_frame_i], D3D12_COMMAND_LIST_TYPE_DIRECT);

  g_fence = CreateFence(g_device);
  g_fence_event = CreateEventHandle();

  g_initialized = true;

  LoadContent();

  for (int i = 0; i < g_imgui_srv_count; i++)
  {
    UINT slot = g_nonimgui_srv_count + i;
    g_imgui_available_srv_slots.push_back(slot);
  }

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
  imgui_info.SrvDescriptorHeap = g_descriptorheap_srv.Get();
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