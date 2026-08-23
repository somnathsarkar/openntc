#include <openntc-gui/model.h>

#include <cassert>

using namespace DirectX;

Model::Model() : vertex_count_(0), index_count_(0), vertices_(nullptr), indices_(nullptr) {}

Model::~Model()
{
  delete[] vertices_;
  delete[] indices_;
}

uint32_t Model::GetVertexCount() const
{
  return vertex_count_;
}

uint32_t Model::GetIndexCount() const
{
  return index_count_;
}

VertexDescriptor* Model::GetVertices()
{
  return vertices_;
}

uint32_t* Model::GetIndices()
{
  return indices_;
}

struct CubeFace
{
  XMFLOAT3 origin;   // position at uv=(0, 0)
  XMFLOAT3 axis_u;   // axis along which u increases
  XMFLOAT3 axis_v;   // axis along which v increases
  XMFLOAT3 normal;
  XMFLOAT3 tangent;
};

static const CubeFace kCubeFaces[6] = {
  { {-1.0f,  1.0f, -1.0f}, { 2.0f, 0.0f,  0.0f}, {0.0f, -2.0f,  0.0f}, { 0.0f,  0.0f, -1.0f}, { 1.0f, 0.0f,  0.0f} }, // front  (z = -1)
  { { 1.0f,  1.0f,  1.0f}, {-2.0f, 0.0f,  0.0f}, {0.0f, -2.0f,  0.0f}, { 0.0f,  0.0f,  1.0f}, {-1.0f, 0.0f,  0.0f} }, // back   (z = +1)
  { {-1.0f,  1.0f,  1.0f}, { 0.0f, 0.0f, -2.0f}, {0.0f, -2.0f,  0.0f}, {-1.0f,  0.0f,  0.0f}, { 0.0f, 0.0f, -1.0f} }, // left   (x = -1)
  { { 1.0f,  1.0f, -1.0f}, { 0.0f, 0.0f,  2.0f}, {0.0f, -2.0f,  0.0f}, { 1.0f,  0.0f,  0.0f}, { 0.0f, 0.0f,  1.0f} }, // right  (x = +1)
  { {-1.0f,  1.0f,  1.0f}, { 2.0f, 0.0f,  0.0f}, {0.0f,  0.0f, -2.0f}, { 0.0f,  1.0f,  0.0f}, { 1.0f, 0.0f,  0.0f} }, // top    (y = +1)
  { {-1.0f, -1.0f, -1.0f}, { 2.0f, 0.0f,  0.0f}, {0.0f,  0.0f,  2.0f}, { 0.0f, -1.0f,  0.0f}, { 1.0f, 0.0f,  0.0f} }, // bottom (y = -1)
};

void InitModelCube(uint32_t detail, Model& o_model)
{
  assert(detail >= 1);
  const uint32_t n = detail;
  const uint32_t face_vertex_count = (n + 1) * (n + 1);
  const uint32_t face_index_count = n * n * 6;

  delete[] o_model.vertices_;
  delete[] o_model.indices_;
  o_model.vertex_count_ = 6 * face_vertex_count;
  o_model.index_count_ = 6 * face_index_count;
  o_model.vertices_ = new VertexDescriptor[o_model.vertex_count_];
  o_model.indices_ = new uint32_t[o_model.index_count_];

  uint32_t vert_i = 0;
  uint32_t idx_i = 0;
  for (uint32_t face_i = 0; face_i < 6; face_i++)
  {
    const CubeFace& face = kCubeFaces[face_i];
    const uint32_t base = face_i * face_vertex_count;

    for (uint32_t j = 0; j <= n; j++)
    {
      float tv = static_cast<float>(j) / static_cast<float>(n);
      for (uint32_t i = 0; i <= n; i++)
      {
        float tu = static_cast<float>(i) / static_cast<float>(n);
        VertexDescriptor& v = o_model.vertices_[vert_i++];
        XMVECTOR vpos = XMLoadFloat3(&face.origin) + tu * XMLoadFloat3(&face.axis_u) + tv * XMLoadFloat3(&face.axis_v);
        XMStoreFloat3(&v.pos, vpos);
        v.normal = face.normal;
        v.tangent = face.tangent;
        v.uv = XMFLOAT2(tu, tv);
      }
    }

    for (uint32_t j = 0; j < n; j++)
    {
      for (uint32_t i = 0; i < n; i++)
      {
        uint32_t a = base + j * (n + 1) + i;   // uv (i, j)
        uint32_t b = a + 1;                    // uv (i + 1, j)
        uint32_t c = a + (n + 1);              // uv (i, j + 1)
        uint32_t d = c + 1;                    // uv (i + 1, j + 1)
        
        o_model.indices_[idx_i++] = c;
        o_model.indices_[idx_i++] = a;
        o_model.indices_[idx_i++] = b;
        o_model.indices_[idx_i++] = c;
        o_model.indices_[idx_i++] = b;
        o_model.indices_[idx_i++] = d;
      }
    }
  }
}