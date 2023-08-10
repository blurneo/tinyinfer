#pragma once
#include "tinyinfer/common/check_macro.h"
#include "tinyinfer/common/tensor.h"
#include "tinyinfer/common/base_layer.h"
#include "tinyinfer/layer/batch_normalization.h"
#include <cmath>

namespace ti
{

  bool BatchNormalization::forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                                   std::vector<std::shared_ptr<Tensor>> output_tensors)
  {
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

  bool BatchNormalization::kernel(std::shared_ptr<Tensor> input_tensor,
                                  std::shared_ptr<Tensor> output_tensor)
  {
    auto &input_values = input_tensor->get_values();
    auto &output_values = output_tensor->get_values();
    auto input_dims_vec = input_tensor->dims_vector();
    int axis_idx = 1; // channel dim idx
    std::optional<int> stride = input_tensor->dim_stride(axis_idx);
    CHECK_BOOL_RET(stride.has_value(), true,
                   "BatchNormalization calculate stride failed\n");
    int dim_from_idx = input_dims_vec[axis_idx];
    // Y = (X - input_mean) / sqrt(input_var + epsilon) * scale + B
    for (int batch_idx = 0; batch_idx < input_tensor->get_n(); batch_idx++) {
      int cur_batch_start = batch_idx * input_tensor->get_c();
      for (int c_idx = 0; c_idx < input_tensor->get_c(); c_idx++) {
        int cur_c_start = cur_batch_start + c_idx * stride.value();
        for (int val_idx = 0; val_idx < stride; val_idx++) {
          int cur_idx = cur_c_start + val_idx;
          output_values[cur_idx] = (input_values[cur_idx] - param_.mean[c_idx]) /
                                  std::sqrt(param_.var[c_idx] + param_.epsilon) *
                                  param_.scale[c_idx] +
                              param_.b[c_idx];
        }
      }
    }

    return true;
  }

} // namespace ti
