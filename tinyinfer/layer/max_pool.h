#pragma once

#include "tinyinfer/common/check_macro.h"
#include "tinyinfer/common/tensor.h"
#include "tinyinfer/layer/base_layer.h"

namespace ti {

typedef struct MaxPoolLayerParameter : public BaseLayerParameter {
  int kernel_shape_x;
  int kernel_shape_y;
  int stride_x;
  int stride_y;
  int pad_l;
  int pad_r;
  int pad_t;
  int pad_d;
} MaxPoolLayerParameter;

class MaxPool : public BaseLayer {
public:
  MaxPool() : BaseLayer(LAYER_MAXPOOL) {}
  MaxPool(MaxPoolLayerParameter &&param)
      : param_(std::move(param)), BaseLayer(LAYER_MAXPOOL) {}
  bool forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
               std::vector<std::shared_ptr<Tensor>> output_tensors) override;

private:
  bool kernel(std::shared_ptr<Tensor> input_tensor,
              std::shared_ptr<Tensor> output_tensor);
private:
  MaxPoolLayerParameter param_;
};

} // namespace ti
