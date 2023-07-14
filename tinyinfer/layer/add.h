#pragma once
#include <ostream>
#include "tinyinfer/common/check_macro.h"
#include "tinyinfer/common/tensor.h"
#include "tinyinfer/layer/base_layer.h"

namespace ti {

typedef struct AddLayerParameter : public BaseLayerParameter {
  std::shared_ptr<Tensor> weights;
} AddLayerParameter;

class Add : public BaseLayer {
public:
  Add() : BaseLayer(LAYER_ADD) {}
  Add(AddLayerParameter &&param)
      : param_(std::move(param)), BaseLayer(LAYER_ADD) {}
  bool is_compatible(const std::shared_ptr<Tensor> &input_tensor,
                     std::vector<int> &broadcast_shapes);
  bool forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
               std::vector<std::shared_ptr<Tensor>> output_tensors) override;

private:
  bool kernel(const std::shared_ptr<Tensor> &input_tensor,
              std::shared_ptr<Tensor> output_tensor);

private:
  AddLayerParameter param_;
};

} // namespace ti
