#include <openntc-gui/rhi.h>

#include <cassert>
#include <d3dcompiler.h>

// NOTE: This is far away from a full RHI, currently just a list of helper functions.
//  Need to split out further functionality from main before Vulkan implementation.

using namespace Microsoft::WRL;

#if defined(_DEBUG)
#define VERIFY(hr) do { assert(!FAILED(hr)); } while(0)
#else
#define VERIFY(hr) do { (hr); } while(0)
#endif

void DescriptorAllocator::Init(
  ID3D12Device2* device,
  D3D12_DESCRIPTOR_HEAP_TYPE heap_type,
  bool shader_visible,
  uint32_t size)
{
  D3D12_DESCRIPTOR_HEAP_DESC desc = {};
  desc.Flags = shader_visible ? 
                D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE :
                D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
  desc.NodeMask = 0;
  desc.NumDescriptors = size;
  desc.Type = heap_type;
  VERIFY(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&heap_)));

  handle_i_ = 0;

  size_ = size;
  cpu_start_ = heap_->GetCPUDescriptorHandleForHeapStart();
  if (shader_visible)
    gpu_start_ = heap_->GetGPUDescriptorHandleForHeapStart();
  else
    gpu_start_ = { 0 };
  handle_increment_ = device->GetDescriptorHandleIncrementSize(heap_type);
}

DescriptorHandle DescriptorAllocator::Allocate()
{
  assert(handle_i_ < size_);

  DescriptorHandle ret = {};
  ret.cpu_ = { cpu_start_.ptr + handle_increment_ * handle_i_ };
  ret.gpu_ = { gpu_start_.ptr + handle_increment_ * handle_i_ };
  handle_i_++;

  return ret;
}

DescriptorHandle DescriptorAllocator::At(uint32_t handle_i) const
{
  assert(handle_i < handle_i_);

  DescriptorHandle ret = {};
  ret.cpu_ = { cpu_start_.ptr + handle_increment_ * handle_i };
  ret.gpu_ = { gpu_start_.ptr + handle_increment_ * handle_i };

  return ret;
}

uint32_t DescriptorAllocator::SlotForHandle(D3D12_CPU_DESCRIPTOR_HANDLE cpu) const
{
  assert(cpu.ptr >= cpu_start_.ptr);
  uint32_t slot = (cpu.ptr - cpu_start_.ptr) / handle_increment_;
  assert(slot < handle_i_);
  return slot;
}

ID3D12DescriptorHeap* DescriptorAllocator::GetHeapUnsafe()
{
  return heap_.Get();
}

ComPtr<ID3D12Resource> CreateBuffer(ID3D12Device2* device, D3D12_HEAP_TYPE heap_type, uint64_t size)
{
  ComPtr<ID3D12Resource> buffer;

  D3D12_HEAP_PROPERTIES props = {};
  props.Type = heap_type;
  props.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
  props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
  props.CreationNodeMask = 1;
  props.VisibleNodeMask = 1;

  D3D12_RESOURCE_DESC desc = {};
  desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  desc.Alignment = 0;
  desc.Width = size;
  desc.Height = 1;
  desc.DepthOrArraySize = 1;
  desc.MipLevels = 1;
  desc.Format = DXGI_FORMAT_UNKNOWN;
  desc.SampleDesc.Count = 1;
  desc.SampleDesc.Quality = 0;
  desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  desc.Flags = D3D12_RESOURCE_FLAG_NONE;

  D3D12_RESOURCE_STATES initial_state = heap_type == D3D12_HEAP_TYPE_READBACK ? D3D12_RESOURCE_STATE_COPY_DEST
    : heap_type == D3D12_HEAP_TYPE_UPLOAD ? D3D12_RESOURCE_STATE_GENERIC_READ
    : D3D12_RESOURCE_STATE_COMMON;
  VERIFY(device->CreateCommittedResource(
    &props,
    D3D12_HEAP_FLAG_NONE,
    &desc,
    initial_state,
    nullptr,
    IID_PPV_ARGS(&buffer)));

  return buffer;
}

ComPtr<ID3D12Resource> CreateBufferWithData(
  ID3D12Device2* device,
  D3D12_HEAP_TYPE heap_type,
  uint64_t buffer_size,
  uint64_t data_size,
  const void* data)
{
  ComPtr<ID3D12Resource> buffer = CreateBuffer(device, heap_type, buffer_size);

  void* mapped = nullptr;
  D3D12_RANGE range = {0, 0};
  buffer->Map(0, &range, &mapped);
  memcpy(mapped, data, data_size);
  buffer->Unmap(0, nullptr);

  return buffer;
}

ComPtr<ID3D12Resource> CreateTexture2D(
  ID3D12Device2* device,
  D3D12_HEAP_TYPE heap_type,
  uint32_t width,
  uint32_t height,
  uint32_t depth,
  uint32_t levels,
  DXGI_FORMAT format,
  D3D12_RESOURCE_FLAGS flags,
  const float* opt_clear_color)
{
  ComPtr<ID3D12Resource> tex;

  D3D12_CLEAR_VALUE clear_value = {};
  clear_value.Format = format;
  if (opt_clear_color != nullptr)
    memcpy(clear_value.Color, opt_clear_color, sizeof(clear_value.Color));

  D3D12_RESOURCE_DESC tex_desc = {};
  tex_desc.Format = format;
  tex_desc.Alignment = 0;
  tex_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
  tex_desc.Flags = flags;
  tex_desc.Width = width;
  tex_desc.Height = height;
  tex_desc.DepthOrArraySize = depth;
  tex_desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
  tex_desc.MipLevels = levels;
  tex_desc.SampleDesc.Count = 1;
  tex_desc.SampleDesc.Quality = 0;

  D3D12_HEAP_PROPERTIES heap_props;
  heap_props.Type = heap_type;
  heap_props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
  heap_props.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
  heap_props.CreationNodeMask = 1;
  heap_props.VisibleNodeMask = 1;

  VERIFY(device->CreateCommittedResource(
    &heap_props,
    D3D12_HEAP_FLAG_NONE,
    &tex_desc,
    D3D12_RESOURCE_STATE_COMMON,
    opt_clear_color != nullptr ? &clear_value : nullptr,
    IID_PPV_ARGS(&tex)));

  return tex;
}

RootSignatureBuilder& RootSignatureBuilder::RootConstants(uint32_t num_32bit_values, D3D12_SHADER_VISIBILITY vis)
{
  D3D12_ROOT_PARAMETER1 param = {};
  param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
  param.Constants.Num32BitValues = num_32bit_values;
  param.Constants.RegisterSpace = 0;
  param.Constants.ShaderRegister = cbv_register_i_;
  param.ShaderVisibility = vis;
  cbv_register_i_++;
  params_.push_back(param);

  return *this;
}

RootSignatureBuilder& RootSignatureBuilder::Range(
  uint32_t num_descriptors,
  D3D12_DESCRIPTOR_RANGE_TYPE range_type,
  D3D12_DESCRIPTOR_RANGE_FLAGS flags,
  D3D12_SHADER_VISIBILITY vis)
{
  assert(range_type == D3D12_DESCRIPTOR_RANGE_TYPE_CBV || range_type == D3D12_DESCRIPTOR_RANGE_TYPE_SRV);

  D3D12_DESCRIPTOR_RANGE1 drange = {};
  drange.RegisterSpace = 0;
  drange.BaseShaderRegister = (range_type == D3D12_DESCRIPTOR_RANGE_TYPE_CBV) ?
                                            cbv_register_i_ :
                                            srv_register_i_;
  drange.NumDescriptors = num_descriptors;
  drange.OffsetInDescriptorsFromTableStart = 0;
  drange.Flags = flags;
  drange.RangeType = range_type;
  ranges_.push_back(drange);

  D3D12_ROOT_PARAMETER1 rparam = {};
  rparam.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
  rparam.DescriptorTable.NumDescriptorRanges = 1;
  rparam.DescriptorTable.pDescriptorRanges = nullptr;
  rparam.ShaderVisibility = vis;
  param_range_index_.push_back(static_cast<uint32_t>(ranges_.size() - 1));
  params_.push_back(rparam);
  if (range_type == D3D12_DESCRIPTOR_RANGE_TYPE_CBV)
    cbv_register_i_ += num_descriptors;
  else
    srv_register_i_ += num_descriptors;

  return *this;
}

RootSignatureBuilder& RootSignatureBuilder::StaticSampler(
  D3D12_FILTER filter,
  D3D12_TEXTURE_ADDRESS_MODE address_mode,
  D3D12_SHADER_VISIBILITY vis)
{
  D3D12_STATIC_SAMPLER_DESC desc = {};
  desc.Filter = filter;
  desc.AddressU = address_mode;
  desc.AddressV = address_mode;
  desc.AddressW = address_mode;
  desc.MipLODBias = 0.0f;
  desc.MaxAnisotropy = 16;
  desc.ComparisonFunc = D3D12_COMPARISON_FUNC_NONE;
  desc.BorderColor = D3D12_STATIC_BORDER_COLOR_OPAQUE_BLACK;
  desc.MinLOD = 0.0f;
  desc.MaxLOD = D3D12_FLOAT32_MAX;
  desc.ShaderRegister = smp_register_i_;
  desc.RegisterSpace = 0;
  desc.ShaderVisibility = vis;
  smp_register_i_++;
  samplers_.push_back(desc);

  return *this;
}

ComPtr<ID3D12RootSignature> RootSignatureBuilder::Build(ID3D12Device2* device)
{
  assert(built_ == false);
  if (built_)
    return nullptr;

  uint32_t table_i = 0;
  for (D3D12_ROOT_PARAMETER1& param : params_)
  {
    if (param.ParameterType == D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE)
      param.DescriptorTable.pDescriptorRanges = &ranges_[param_range_index_[table_i++]];
  }

  D3D12_ROOT_SIGNATURE_FLAGS root_signature_flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT |
                                                  D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS |
                                                  D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS |
                                                  D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS;

  D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_signature_desc = {};
  root_signature_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
  root_signature_desc.Desc_1_1.NumParameters = static_cast<UINT>(params_.size());
  root_signature_desc.Desc_1_1.pParameters = params_.data();
  root_signature_desc.Desc_1_1.NumStaticSamplers = static_cast<UINT>(samplers_.size());
  root_signature_desc.Desc_1_1.pStaticSamplers = samplers_.empty() ? nullptr : samplers_.data();
  root_signature_desc.Desc_1_1.Flags = root_signature_flags;

  ComPtr<ID3D12RootSignature> root_signature;
  ComPtr<ID3DBlob> root_signature_blob;
  ComPtr<ID3DBlob> error_blob;
  VERIFY(D3D12SerializeVersionedRootSignature(&root_signature_desc, &root_signature_blob, &error_blob));

  VERIFY(device->CreateRootSignature(
    0,
    root_signature_blob->GetBufferPointer(),
    root_signature_blob->GetBufferSize(),
    IID_PPV_ARGS(&root_signature)));
  built_ = true;

  return root_signature;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::RootSignature(ID3D12RootSignature* root_signature)
{
  root_signature_ = root_signature;
  return *this;
}

static ComPtr<ID3DBlob> BlobFromMemory(const void* data, size_t size)
{
  ComPtr<ID3DBlob> blob;
  VERIFY(D3DCreateBlob(size, &blob));
  memcpy(blob->GetBufferPointer(), data, size);
  return blob;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::VS(const void* data, size_t size)
{
  vs_blob_ = BlobFromMemory(data, size);
  return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::PS(const void* data, size_t size)
{
  ps_blob_ = BlobFromMemory(data, size);
  return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::Input(const D3D12_INPUT_ELEMENT_DESC* elements, uint32_t count)
{
  input_elements_ = elements;
  input_count_ = count;
  return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::DepthEnable(bool enable)
{
  depth_enable_ = enable;
  return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::RtvFormat(DXGI_FORMAT format)
{
  rtv_format_ = format;
  return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::DsvFormat(DXGI_FORMAT format)
{
  dsv_format_ = format;
  return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::CullMode(D3D12_CULL_MODE mode)
{
  cull_mode_ = mode;
  return *this;
}

ComPtr<ID3D12PipelineState> GraphicsPipelineBuilder::Build(ID3D12Device2* device)
{
  assert(built_ == false);
  assert(root_signature_ != nullptr && vs_blob_ && ps_blob_);
  if (built_)
    return nullptr;

  D3D12_GRAPHICS_PIPELINE_STATE_DESC desc = {};
  desc.pRootSignature = root_signature_;
  desc.VS = { vs_blob_->GetBufferPointer(), vs_blob_->GetBufferSize() };
  desc.PS = { ps_blob_->GetBufferPointer(), ps_blob_->GetBufferSize() };
  desc.InputLayout = { input_elements_, input_count_ };

  desc.BlendState.AlphaToCoverageEnable = FALSE;
  desc.BlendState.IndependentBlendEnable = FALSE;
  desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
  desc.SampleMask = UINT_MAX;

  desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
  desc.RasterizerState.CullMode = cull_mode_;
  desc.RasterizerState.FrontCounterClockwise = FALSE;
  desc.RasterizerState.DepthClipEnable = TRUE;

  desc.DepthStencilState.DepthEnable = depth_enable_ ? TRUE : FALSE;
  desc.DepthStencilState.DepthWriteMask = depth_enable_ ? D3D12_DEPTH_WRITE_MASK_ALL : D3D12_DEPTH_WRITE_MASK_ZERO;
  desc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
  desc.DSVFormat = dsv_format_;

  desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
  desc.NumRenderTargets = 1;
  desc.RTVFormats[0] = rtv_format_;
  desc.SampleDesc.Count = 1;

  ComPtr<ID3D12PipelineState> pso;
  VERIFY(device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&pso)));
  built_ = true;

  return pso;
}
