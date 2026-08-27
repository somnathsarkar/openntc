#include <wrl.h>

#include <d3d12.h>
#include <dxgi1_6.h>

#include <cstdint>
#include <vector>

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
Microsoft::WRL::ComPtr<ID3D12Resource> CreateTexture2D(ID3D12Device2* device, D3D12_HEAP_TYPE heap_type, uint32_t width, uint32_t height, uint32_t depth, uint32_t levels, DXGI_FORMAT format, D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE, const float* opt_clear_color = nullptr);

class RootSignatureBuilder
{
  public:
    RootSignatureBuilder& RootConstants(uint32_t num_32bit_values, D3D12_SHADER_VISIBILITY vis);
    RootSignatureBuilder& Range(uint32_t num_descriptors, D3D12_DESCRIPTOR_RANGE_TYPE range_type, D3D12_DESCRIPTOR_RANGE_FLAGS flags, D3D12_SHADER_VISIBILITY vis);
    RootSignatureBuilder& StaticSampler(D3D12_FILTER filter, D3D12_TEXTURE_ADDRESS_MODE address_mode, D3D12_SHADER_VISIBILITY vis);
    Microsoft::WRL::ComPtr<ID3D12RootSignature> Build(ID3D12Device2* device);

  private:
    std::vector<D3D12_DESCRIPTOR_RANGE1> ranges_;
    std::vector<D3D12_ROOT_PARAMETER1> params_;
    std::vector<D3D12_STATIC_SAMPLER_DESC> samplers_;
    std::vector<uint32_t> param_range_index_;
    bool built_ = false;
    uint32_t cbv_register_i_ = 0;
    uint32_t srv_register_i_ = 0;
    uint32_t smp_register_i_ = 0;
};

class GraphicsPipelineBuilder
{
  public:
    GraphicsPipelineBuilder& RootSignature(ID3D12RootSignature* root_signature);
    GraphicsPipelineBuilder& VS(const wchar_t* cso_path);
    GraphicsPipelineBuilder& PS(const wchar_t* cso_path);
    GraphicsPipelineBuilder& Input(const D3D12_INPUT_ELEMENT_DESC* elements, uint32_t count);
    GraphicsPipelineBuilder& DepthEnable(bool enable);
    GraphicsPipelineBuilder& CullMode(D3D12_CULL_MODE mode);
    GraphicsPipelineBuilder& RtvFormat(DXGI_FORMAT format);
    GraphicsPipelineBuilder& DsvFormat(DXGI_FORMAT format);
    Microsoft::WRL::ComPtr<ID3D12PipelineState> Build(ID3D12Device2* device);

  private:
    Microsoft::WRL::ComPtr<ID3DBlob> vs_blob_;
    Microsoft::WRL::ComPtr<ID3DBlob> ps_blob_;
    ID3D12RootSignature* root_signature_ = nullptr;
    const D3D12_INPUT_ELEMENT_DESC* input_elements_ = nullptr;
    uint32_t input_count_ = 0;
    bool depth_enable_ = true;
    D3D12_CULL_MODE cull_mode_ = D3D12_CULL_MODE_BACK;
    DXGI_FORMAT rtv_format_ = DXGI_FORMAT_R8G8B8A8_UNORM;
    DXGI_FORMAT dsv_format_ = DXGI_FORMAT_D32_FLOAT;
    bool built_ = false;
};
