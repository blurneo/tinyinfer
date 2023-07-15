#pragma once

#include "tinyinfer/common/check_macro.h"
#include "tinyinfer/common/tensor.h"
#include "tinyinfer/common/base_layer.h"
#include <vector>
#include "tinyinfer/reflection/serialize_macro.h"

namespace ti {

typedef struct GemmLayerParameter : public BaseLayerParameter {
  float alpha;
  float beta;
  bool trans_a;
  bool trans_b;
  std::shared_ptr<Tensor> weights;
  std::shared_ptr<Tensor> bias;
  DEFINE_SERIALIZE_MEMBER(
    ("alpha", alpha)
    ("beta", beta)
    ("trans_a", trans_a)
    ("trans_b", trans_b)
    ("weights", weights)
    ("bias", bias)
  )
} GemmLayerParameter;

class Serializer;
class Deserializer;
class Gemm : public BaseLayer {
public:
  Gemm() : BaseLayer(LAYER_GEMM) {}
  Gemm(GemmLayerParameter &&param) : param_(param), BaseLayer(LAYER_GEMM) {}

  bool forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
               std::vector<std::shared_ptr<Tensor>> output_tensors) override;
  virtual void serialize(Serializer &serializer);
  virtual bool deserialize(Deserializer& deserializer);
private:
  bool kernel(std::shared_ptr<Tensor> input_tensor1,
              std::shared_ptr<Tensor> input_tensor2,
              std::shared_ptr<Tensor> output_tensor);

private:
  GemmLayerParameter param_;
  DEFINE_SERIALIZE_MEMBER(
    ("param_", &param_)
  )
};

} // namespace ti
