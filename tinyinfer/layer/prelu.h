#pragma once
#include "tinyinfer/common/check_macro.h"
#include "tinyinfer/common/tensor.h"
#include "tinyinfer/common/base_layer.h"
#include "tinyinfer/reflection/serialize_macro.h"

namespace ti {

typedef struct PReluLayerParam : public BaseLayerParameter {
  std::shared_ptr<Tensor> slope;
  DEFINE_SERIALIZE_MEMBER(
    (slope)
  )
} PReluLayerParameter;

class Serializer;
class Deserializer;
class PRelu : public BaseLayer {
public:
  PRelu() : BaseLayer(LAYER_PRELU) {}
  PRelu(PReluLayerParameter &&param)
      : param_(std::move(param)), BaseLayer(LAYER_PRELU) {}
  bool forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
               std::vector<std::shared_ptr<Tensor>> output_tensors) override;
  virtual void serialize(Serializer &serializer);
  virtual bool deserialize(Deserializer& deserializer);
private:
  bool kernel(std::shared_ptr<Tensor> input_tensor,
              std::shared_ptr<Tensor> output_tensor);

private:
  PReluLayerParameter param_;
  DEFINE_SERIALIZE_MEMBER(
    (&param_)
  )
};

} // namespace ti
