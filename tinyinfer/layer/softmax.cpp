#pragma once
#include "tinyinfer/common/check_macro.h"
#include "tinyinfer/common/tensor.h"
#include "tinyinfer/common/base_layer.h"
#include "tinyinfer/layer/softmax.h"
#include <cmath>

namespace ti
{

  bool Softmax::forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                        std::vector<std::shared_ptr<Tensor>> output_tensors)
  {
    CHECK_BOOL_RET(input_tensors.size(), 1,
                   "Maxpool input tensor number should be 1")
    std::shared_ptr<Tensor> input_tensor = input_tensors[0];
    std::shared_ptr<Tensor> output_tensor = output_tensors[0];
    output_tensor->reshape_like(input_tensor);
    // calculate computation and memory infomation
    flops_ = 2 * input_tensors[0]->get_count() * 10; // assume exp() costs 10 flops
    bytes_ = input_tensors[0]->get_bytes() + input_tensors[1]->get_bytes() + output_tensor->get_bytes();
    return kernel(input_tensor, output_tensor);
  }

  bool Softmax::kernel(std::shared_ptr<Tensor> input_tensor,
                       std::shared_ptr<Tensor> output_tensor)
  {
    auto &input_values = input_tensor->get_values();
    auto &output_values = output_tensor->get_values();
    auto input_dims_vec = input_tensor->dims_vector();
    int axis_idx =
        param_.axis < 0 ? param_.axis + input_dims_vec.size() : param_.axis;
    std::optional<int> stride = input_tensor->dim_stride(axis_idx);
    CHECK_BOOL_RET(stride.has_value(), true,
                   "Softmax calculate stride failed\n");
    int dim_from_idx = input_dims_vec[axis_idx];
    int sidx = 0;
    while (sidx < output_values.size())
    {
      float sum = 0.f;
      for (int i = 0; i < dim_from_idx; i++)
      {
        int idx = sidx + i * stride.value();
        output_values[idx] = std::exp(input_values[idx]);
        sum += output_values[idx];
      }
      for (int i = 0; i < dim_from_idx; i++)
      {
        int idx = sidx + i * stride.value();
        output_values[idx] = output_values[idx] / sum;
      }
      sidx += (dim_from_idx * stride.value());
    }

    return true;
  }

} // namespace ti
