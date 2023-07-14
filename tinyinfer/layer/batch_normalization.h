#pragma once
#include "tinyinfer/common/check_macro.h"
#include "tinyinfer/common/tensor.h"
#include "tinyinfer/layer/base_layer.h"
#include <cmath>

namespace ti {

typedef struct BatchNormalizationLayerParam : public BaseLayerParameter {
  float epsilon = 1e-05;
  std::vector<float> scale;
  std::vector<float> b;
  std::vector<float> mean;
  std::vector<float> var;
} BatchNormalizationLayerParameter;

class BatchNormalization : public BaseLayer {
public:
  BatchNormalization() : BaseLayer(LAYER_BATCH_NORMALIZATION) {}
  BatchNormalization(BatchNormalizationLayerParameter &&param)
      : param_(std::move(param)), BaseLayer(LAYER_BATCH_NORMALIZATION) {}
  bool forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
               std::vector<std::shared_ptr<Tensor>> output_tensors) override;

private:
  bool kernel(std::shared_ptr<Tensor> input_tensor,
              std::shared_ptr<Tensor> output_tensor);

private:
  BatchNormalizationLayerParameter param_;
};

} // namespace ti
