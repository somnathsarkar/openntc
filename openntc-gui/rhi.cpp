#include <openntc-gui/rhi.h>

#include <cassert>

using namespace Microsoft::WRL;

void DescriptorAllocator::Init(ID3D12Device2* device, D3D12_DESCRIPTOR_HEAP_TYPE heap_type, bool shader_visible, uint32_t size)
{
  D3D12_DESCRIPTOR_HEAP_DESC desc = {};
  desc.Flags = shader_visible ? 
                D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE :
                D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
  desc.NodeMask = 0;
  desc.NumDescriptors = size;
  desc.Type = heap_type;
  device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&heap_));

  handle_i_ = 0;

  size_ = size;
  cpu_start_ = heap_->GetCPUDescriptorHandleForHeapStart();
  gpu_start_ = heap_->GetGPUDescriptorHandleForHeapStart();
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