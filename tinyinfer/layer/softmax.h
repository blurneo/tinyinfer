#pragma once
#include "tinyinfer/common/check_macro.h"
#include "tinyinfer/common/tensor.h"
#include "tinyinfer/layer/base_layer.h"
#include <cmath>

namespace ti {

typedef struct SoftmaxLayerParam : public BaseLayerParameter {
  int axis = -1;
} SoftmaxLayerParameter;

class Softmax : public BaseLayer {
public:
  Softmax() : BaseLayer(LAYER_SOFTMAX) {}
  Softmax(SoftmaxLayerParameter &&param)
      : param_(std::move(param)), BaseLayer(LAYER_SOFTMAX) {}
  bool forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
               std::vector<std::shared_ptr<Tensor>> output_tensors) override;

private:
  bool kernel(std::shared_ptr<Tensor> input_tensor,
              std::shared_ptr<Tensor> output_tensor);

private:
  SoftmaxLayerParameter param_;
};

} // namespace ti
