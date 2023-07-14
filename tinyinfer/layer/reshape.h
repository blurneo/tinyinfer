#pragma once

#include "tinyinfer/common/check_macro.h"
#include "tinyinfer/common/tensor.h"
#include "tinyinfer/layer/base_layer.h"
#include <vector>

namespace ti {

typedef struct ReshapeLayerParameter : public BaseLayerParameter {
  std::shared_ptr<Tensor> data;
  std::vector<unsigned long> shape;
} ReshapeLayerParameter;

class Reshape : public BaseLayer {
public:
  Reshape() : BaseLayer(LAYER_RESHAPE) {}
  Reshape(ReshapeLayerParameter &&param)
      : param_(param), BaseLayer(LAYER_RESHAPE) {}
  bool forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
               std::vector<std::shared_ptr<Tensor>> output_tensors) override;

private:
  bool kernel(std::shared_ptr<Tensor> input_tensor,
              std::shared_ptr<Tensor> output_tensor);
  ReshapeLayerParameter param_;
};

} // namespace ti
