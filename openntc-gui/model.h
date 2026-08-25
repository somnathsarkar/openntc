#pragma once

#include <cstdint>
#include <DirectXMath.h>

enum class ModelType : int32_t
{
  kCube = 0,
  kSphere = 1,
  kPlane = 2,
  kCount = 3,
};

constexpr int32_t g_kModelCount = static_cast<int32_t>(ModelType::kCount);

struct VertexDescriptor
{
  DirectX::XMFLOAT3 pos;
  DirectX::XMFLOAT3 normal;
  DirectX::XMFLOAT3 tangent;
  DirectX::XMFLOAT2 uv;
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
    friend void InitModel(ModelType type, uint32_t detail, Model& o_model);
    static void InitCube(uint32_t detail, Model& o_model);
    static void InitSphere(uint32_t detail, Model& o_model);
    static void InitPlane(uint32_t detail, Model& o_model);

    uint32_t vertex_count_;
    uint32_t index_count_;
    VertexDescriptor* vertices_;
    uint32_t* indices_;
};

void InitModel(ModelType type, uint32_t detail, Model& o_model);