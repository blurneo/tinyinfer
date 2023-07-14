#pragma once
#include "tinyinfer/common/check_macro.h"
#include "tinyinfer/common/tensor.h"
#include "tinyinfer/layer/base_layer.h"

namespace ti {

typedef struct ReluLayerParam : public BaseLayerParameter {
} ReluLayerParameter;

class Relu : public BaseLayer {
public:
  Relu() : BaseLayer(LAYER_RELU) {}
  Relu(ReluLayerParameter &&param)
      : param_(std::move(param)), BaseLayer(LAYER_RELU) {}
  bool forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
               std::vector<std::shared_ptr<Tensor>> output_tensors) override;

private:
  bool kernel(std::shared_ptr<Tensor> input_tensor,
              std::shared_ptr<Tensor> output_tensor);

private:
  ReluLayerParameter param_;
};

} // namespace ti
