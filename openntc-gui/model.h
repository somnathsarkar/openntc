#pragma once

#include <cstdint>
#include <DirectXMath.h>

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
    friend void InitModelCube(uint32_t detail, Model& o_model);

    uint32_t vertex_count_;
    uint32_t index_count_;
    VertexDescriptor* vertices_;
    uint32_t* indices_;
};

void InitModelCube(uint32_t detail, Model& o_model);