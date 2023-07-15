#pragma once

#include "tinyinfer/common/check_macro.h"
#include "tinyinfer/common/tensor.h"
#include "tinyinfer/common/base_layer.h"
#include <cmath>

namespace ti {

typedef struct LrnLayerParameter : public BaseLayerParameter {
  float alpha;
  float beta;
  float bias;
  int size;
} LrnLayerParameter;

class Lrn : public BaseLayer {
public:
  Lrn() : BaseLayer(LAYER_LRN) {}
  Lrn(LrnLayerParameter &&param)
      : param_(std::move(param)), BaseLayer(LAYER_LRN) {}
  bool forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
               std::vector<std::shared_ptr<Tensor>> output_tensors) override;

private:
  bool kernel(const std::shared_ptr<Tensor> &input_tensor,
              std::shared_ptr<Tensor> &output_tensor);

private:
  LrnLayerParameter param_;
};

} // namespace ti
