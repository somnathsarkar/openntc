#include <wrl.h>

#include <d3d12.h>
#include <dxgi1_6.h>

#include <cstdint>

struct DescriptorHandle
{
  D3D12_CPU_DESCRIPTOR_HANDLE cpu;
  D3D12_GPU_DESCRIPTOR_HANDLE gpu;
};

// Simple linear allocator for descriptor heap management

class DescriptorAllocator
{
  public:
    void Init(ID3D12Device2* device, D3D12_DESCRIPTOR_HEAP_TYPE heap_type, bool shader_visible, uint32_t size);

    DescriptorHandle Allocate();
    DescriptorHandle At(uint32_t handle_i) const;
    uint32_t SlotForHandle(D3D12_CPU_DESCRIPTOR_HANDLE cpu) const;
    ID3D12DescriptorHeap* GetHeapUnsafe();

  private:
    uint32_t size_;
    uint32_t handle_i_;
    UINT handle_increment_;
    D3D12_CPU_DESCRIPTOR_HANDLE cpu_start_;
    D3D12_GPU_DESCRIPTOR_HANDLE gpu_start_;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> heap_;
};

Microsoft::WRL::ComPtr<ID3D12Resource> CreateBuffer(ID3D12Device2* device, D3D12_HEAP_TYPE heap_type, uint64_t size);
Microsoft::WRL::ComPtr<ID3D12Resource> CreateBufferWithData(ID3D12Device2* device, D3D12_HEAP_TYPE heap_type, uint64_t buffer_size, uint64_t data_size, const void* data);
Microsoft::WRL::ComPtr<ID3D12Resource> CreateTexture2D(ID3D12Device2* device, D3D12_HEAP_TYPE heap_type, uint32_t dim, uint32_t depth, uint32_t levels, DXGI_FORMAT format);