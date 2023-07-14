#pragma once

#include "tinyinfer/common/check_macro.h"
#include "tinyinfer/common/tensor.h"
#include "tinyinfer/layer/base_layer.h"

namespace ti {

typedef struct GlobalAveragePoolLayerParameter : public BaseLayerParameter {
} GlobalAveragePoolLayerParameter;

class GlobalAveragePool : public BaseLayer {
public:
  GlobalAveragePool() : BaseLayer(LAYER_GLOBAL_AVERAGE_POOL) {}
  GlobalAveragePool(GlobalAveragePoolLayerParameter &&param)
      : param_(std::move(param)), BaseLayer(LAYER_GLOBAL_AVERAGE_POOL) {}
  bool forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
               std::vector<std::shared_ptr<Tensor>> output_tensors) override;

private:
  bool kernel(std::shared_ptr<Tensor> input_tensor,
              std::shared_ptr<Tensor> output_tensor);

private:
  GlobalAveragePoolLayerParameter param_;
};

} // namespace ti
