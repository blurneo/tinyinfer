#pragma once

#include "tinyinfer/common/check_macro.h"
#include "tinyinfer/common/tensor.h"
#include "tinyinfer/layer/base_layer.h"
#include <vector>

namespace ti {

typedef struct MatmulLayerParameter : public BaseLayerParameter {

} MatmulLayerParameter;

class Matmul : public BaseLayer {
public:
  Matmul() : BaseLayer(LAYER_MATMUL) {}
  Matmul(MatmulLayerParameter &&param)
      : param_(param), BaseLayer(LAYER_MATMUL) {}

private:
  bool forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
               std::vector<std::shared_ptr<Tensor>> output_tensors) override;
  bool kernel(std::shared_ptr<Tensor> input_tensor1,
              std::shared_ptr<Tensor> input_tensor2,
              std::shared_ptr<Tensor> output_tensor);

private:
  MatmulLayerParameter param_;
};

} // namespace ti
