#pragma once
#include "tinyinfer/common/check_macro.h"
#include "tinyinfer/common/tensor.h"
#include "tinyinfer/common/base_layer.h"
#include "tinyinfer/reflection/serialize_macro.h"

namespace ti {

typedef struct ReluLayerParam : public BaseLayerParameter {
} ReluLayerParameter;

class Serializer;
class Deserializer;
class Relu : public BaseLayer {
public:
  Relu() : BaseLayer(LAYER_RELU) {}
  Relu(ReluLayerParameter &&param)
      : param_(std::move(param)), BaseLayer(LAYER_RELU) {}
  bool forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
               std::vector<std::shared_ptr<Tensor>> output_tensors) override;
  virtual void serialize(Serializer &serializer);
  virtual bool deserialize(Deserializer& deserializer);
private:
  bool kernel(std::shared_ptr<Tensor> input_tensor,
              std::shared_ptr<Tensor> output_tensor);

private:
  ReluLayerParameter param_;
  DEFINE_SERIALIZE_MEMBER(
    (&param_)
  )
};

} // namespace ti
