#include <openntc-gui/rhi.h>

#include <cassert>

using namespace Microsoft::WRL;

#if defined(_DEBUG)
#define VERIFY(hr) do { assert(!FAILED(hr)); } while(0)
#else
#define VERIFY(hr) do { (hr); } while(0)
#endif

void DescriptorAllocator::Init(ID3D12Device2* device, D3D12_DESCRIPTOR_HEAP_TYPE heap_type, bool shader_visible, uint32_t size)
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
  ret.cpu = { cpu_start_.ptr + handle_increment_ * handle_i_ };
  ret.gpu = { gpu_start_.ptr + handle_increment_ * handle_i_ };
  handle_i_++;

  return ret;
}

DescriptorHandle DescriptorAllocator::At(uint32_t handle_i) const
{
  assert(handle_i < handle_i_);

  DescriptorHandle ret = {};
  ret.cpu = { cpu_start_.ptr + handle_increment_ * handle_i };
  ret.gpu = { gpu_start_.ptr + handle_increment_ * handle_i };

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

  VERIFY(device->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&buffer)));

  return buffer;
}

ComPtr<ID3D12Resource> CreateBufferWithData(ID3D12Device2* device, D3D12_HEAP_TYPE heap_type, uint64_t buffer_size, uint64_t data_size, const void* data)
{
  ComPtr<ID3D12Resource> buffer = CreateBuffer(device, heap_type, buffer_size);

  void* mapped = nullptr;
  D3D12_RANGE range = {0, 0};
  buffer->Map(0, &range, &mapped);
  memcpy(mapped, data, data_size);
  buffer->Unmap(0, nullptr);

  return buffer;
}

ComPtr<ID3D12Resource> CreateTexture2D(ID3D12Device2* device, D3D12_HEAP_TYPE heap_type, uint32_t dim, uint32_t depth, uint32_t levels, DXGI_FORMAT format)
{
  ComPtr<ID3D12Resource> tex;

  D3D12_RESOURCE_DESC tex_desc = {};
  tex_desc.Format = format;
  tex_desc.Alignment = 0;
  tex_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
  tex_desc.Flags = D3D12_RESOURCE_FLAG_NONE;
  tex_desc.Width = dim;
  tex_desc.Height = dim;
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

  VERIFY(device->CreateCommittedResource(&heap_props, D3D12_HEAP_FLAG_NONE, &tex_desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&tex)));
  
  return tex;
}