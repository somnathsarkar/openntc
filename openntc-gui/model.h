#pragma once

#include <cstdint>
#include <DirectXMath.h>

enum class ModelType : int32_t
{
  kCube = 0,
  kSphere = 1,
  kKnob = 2,
  kCount = 3,
};

constexpr int32_t g_kModelCount = static_cast<int32_t>(ModelType::kCount);

struct VertexDescriptor
{
  DirectX::XMFLOAT3 pos_;
  DirectX::XMFLOAT3 normal_;
  DirectX::XMFLOAT3 tangent_;
  DirectX::XMFLOAT2 uv_;
};

class Model
{
  public:
    Model();
    ~Model();
    uint32_t GetVertexCount() const;
    uint32_t GetIndexCount() const;
    VertexDescriptor* GetVertices();
    uint32_t* GetIndices();
  private:
    friend void InitModel(ModelType type, uint32_t detail, Model& o_model, const void* knob_data, size_t knob_size);
    static void InitCube(uint32_t detail, Model& o_model);
    static void InitSphere(uint32_t detail, Model& o_model);
    static void InitKnob(const void* data, size_t size, Model& o_model);

    uint32_t vertex_count_;
    uint32_t index_count_;
    VertexDescriptor* vertices_;
    uint32_t* indices_;
};

void InitModel(
  ModelType type,
  uint32_t detail,
  Model& o_model,
  const void* knob_data = nullptr,
  size_t knob_size = 0);