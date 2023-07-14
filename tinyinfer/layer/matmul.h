#pragma once

#include "tinyinfer/common/check_macro.h"
#include "tinyinfer/common/tensor.h"
#include "tinyinfer/layer/base_layer.h"
#include <vector>
#include "tinyinfer/net/serialize_macro.h"

namespace ti {

typedef struct MatmulLayerParameter : public BaseLayerParameter {

} MatmulLayerParameter;

class Serializer;
class Deserializer;
class Matmul : public BaseLayer {
public:
  Matmul() : BaseLayer(LAYER_MATMUL) {}
  Matmul(MatmulLayerParameter &&param)
      : param_(param), BaseLayer(LAYER_MATMUL) {}

  bool forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
               std::vector<std::shared_ptr<Tensor>> output_tensors) override;
  virtual void serialize(Serializer &serializer);
  virtual bool deserialize(Deserializer& deserializer);
private:
  bool kernel(std::shared_ptr<Tensor> input_tensor1,
              std::shared_ptr<Tensor> input_tensor2,
              std::shared_ptr<Tensor> output_tensor);

private:
  MatmulLayerParameter param_;
  DEFINE_SERIALIZE_MEMBER(
    ("param_", &param_)
  )
};

} // namespace ti
