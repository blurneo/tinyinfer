#pragma once

#include "tinyinfer/common/check_macro.h"
#include "tinyinfer/common/tensor.h"
#include "tinyinfer/common/base_layer.h"
#include <vector>

namespace ti {

typedef struct FlattenLayerParameter : public BaseLayerParameter {
  int axis = 1;
} FlattenLayerParameter;

class Flatten : public BaseLayer {
public:
  Flatten() : BaseLayer(LAYER_FLATTEN) {}
  Flatten(FlattenLayerParameter &&param)
      : param_(param), BaseLayer(LAYER_FLATTEN) {}
  bool forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
               std::vector<std::shared_ptr<Tensor>> output_tensors) override;

private:
  bool kernel(std::shared_ptr<Tensor> input_tensor, int axis_idx,
              std::shared_ptr<Tensor> output_tensor);
  FlattenLayerParameter param_;
};

} // namespace ti
