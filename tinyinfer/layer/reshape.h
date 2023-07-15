#pragma once

#include "tinyinfer/common/check_macro.h"
#include "tinyinfer/common/tensor.h"
#include "tinyinfer/common/base_layer.h"
#include <vector>
#include "tinyinfer/reflection/serialize_macro.h"

namespace ti {

typedef struct ReshapeLayerParameter : public BaseLayerParameter {
  std::shared_ptr<Tensor> data;
  std::vector<unsigned long> shape;
  DEFINE_SERIALIZE_MEMBER(
    ("data", data)
    ("shape", shape)
  )
} ReshapeLayerParameter;

class Serializer;
class Deserializer;
class Reshape : public BaseLayer {
public:
  Reshape() : BaseLayer(LAYER_RESHAPE) {}
  Reshape(ReshapeLayerParameter &&param)
      : param_(param), BaseLayer(LAYER_RESHAPE) {}
  bool forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
               std::vector<std::shared_ptr<Tensor>> output_tensors) override;
  virtual void serialize(Serializer &serializer);
  virtual bool deserialize(Deserializer& deserializer);
private:
  bool kernel(std::shared_ptr<Tensor> input_tensor,
              std::shared_ptr<Tensor> output_tensor);
  ReshapeLayerParameter param_;
  DEFINE_SERIALIZE_MEMBER(
    ("param_", &param_)
  )
};

} // namespace ti
