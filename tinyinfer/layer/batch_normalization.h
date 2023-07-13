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
  BatchNormalization(BatchNormalizationLayerParameter &&param)
      : param_(std::move(param)), BaseLayer(LAYER_BATCH_NORMALIZATION) {}
  bool forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
               std::vector<std::shared_ptr<Tensor>> output_tensors) override {
    CHECK_BOOL_RET(input_tensors.size(), 1,
                   "Maxpool input tensor number should be 1")
    std::shared_ptr<Tensor> input_tensor = input_tensors[0];
    std::shared_ptr<Tensor> output_tensor = output_tensors[0];
    CHECK_BOOL_RET(param_.scale.size() == input_tensor->get_c(), true,
                   "BN input tensor channel should be equal to scale size")
    CHECK_BOOL_RET(param_.b.size() == input_tensor->get_c(), true,
                   "BN input tensor channel should be equal to b size")
    CHECK_BOOL_RET(param_.mean.size() == input_tensor->get_c(), true,
                   "BN input tensor channel should be equal to mean size")
    CHECK_BOOL_RET(param_.var.size() == input_tensor->get_c(), true,
                   "BN input tensor channel should be equal to var size")
    CHECK_BOOL_RET(input_tensor->dims() >= 2, true,
                   "BN input tensor dims should be at least 2")
    output_tensor->reshape_like(input_tensor);
    return kernel(input_tensor, output_tensor);
  }

private:
  bool kernel(std::shared_ptr<Tensor> input_tensor,
              std::shared_ptr<Tensor> output_tensor) {
    auto &input_values = input_tensor->get_values();
    auto &output_values = output_tensor->get_values();
    auto input_dims_vec = input_tensor->dims_vector();
    int axis_idx = 1; // channel dim idx
    std::optional<int> stride = input_tensor->dim_stride(axis_idx);
    CHECK_BOOL_RET(stride.has_value(), true,
                   "BatchNormalization calculate stride failed\n");
    int dim_from_idx = input_dims_vec[axis_idx];
    int idx = 0;
    // Y = (X - input_mean) / sqrt(input_var + epsilon) * scale + B
    while (idx < output_values.size()) {
      for (int i = 0; i < dim_from_idx; i++) {
        int idx = i * stride.value();
        output_values[idx] = (input_values[idx] - param_.mean[i]) /
                                 std::sqrt(param_.var[i] + param_.epsilon) *
                                 param_.scale[i] +
                             param_.b[i];
      }
    }

    return true;
  }

private:
  BatchNormalizationLayerParameter param_;
};

} // namespace ti
