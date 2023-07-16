#pragma once

#include "tinyinfer/common/check_macro.h"
#include "tinyinfer/common/tensor.h"
#include "tinyinfer/common/base_layer.h"
#include "tinyinfer/layer/flatten.h"
#include <vector>

namespace ti {

bool Flatten::forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
              std::vector<std::shared_ptr<Tensor>> output_tensors) {
    CHECK_BOOL_RET(input_tensors.size(), 1,
                   "Flatten input tensor number should be 1")
    std::shared_ptr<Tensor> input_tensor = input_tensors[0];
    std::shared_ptr<Tensor> output_tensor = output_tensors[0];
    int axis_idx =
        param_.axis < 0 ? param_.axis + input_tensor->dims() : param_.axis;
    CHECK_BOOL_RET(axis_idx < input_tensor->dims(), true,
                   "Flatten axis idx exceed the dims");
    CHECK_BOOL_RET(kernel(input_tensor, axis_idx, output_tensor), true, "flatten kernel executes failed");
    // calculate computation and memory infomation
    flops_ = 0;
    bytes_ = input_tensor->get_bytes() + output_tensor->get_bytes();
    return true;
  }

bool Flatten::kernel(std::shared_ptr<Tensor> input_tensor, int axis_idx,
              std::shared_ptr<Tensor> output_tensor) {
    if (axis_idx == 0) {
      output_tensor->reshape(0, 0, 1, input_tensor->get_count());
    } else {
      auto dims_vec = input_tensor->dims_vector();
      int i = 0;
      int H = 1, W = 1;
      for (; i < axis_idx; i++) {
        H *= dims_vec[i];
      }
      for (; i < dims_vec.size(); i++) {
        W *= dims_vec[i];
      }
      output_tensor->reshape(0, 0, H, W);
    }
    std::memcpy(output_tensor->get_values().data(),
                input_tensor->get_values().data(),
                input_tensor->get_values().size() * sizeof(float));
    return true;
}

} // namespace ti
