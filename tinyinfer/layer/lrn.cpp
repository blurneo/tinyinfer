#pragma once

#include "tinyinfer/common/check_macro.h"
#include "tinyinfer/common/tensor.h"
#include "tinyinfer/common/base_layer.h"
#include "tinyinfer/layer/lrn.h"
#include <cmath>

namespace ti {

bool Lrn::forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
               std::vector<std::shared_ptr<Tensor>> output_tensors) {
    CHECK_BOOL_RET(input_tensors.size(), 1,
                   "Lrn input tensor number should be 1")
    const std::shared_ptr<Tensor> &input_tensor = input_tensors[0];
    std::shared_ptr<Tensor> output_tensor = output_tensors[0];
    output_tensor->reshape_like(input_tensor);
    return kernel(input_tensor, output_tensor);
}

bool Lrn::kernel(const std::shared_ptr<Tensor> &input_tensor,
              std::shared_ptr<Tensor> &output_tensor) {
    int IN_T_N = input_tensor->get_n();
    int IN_T_C = input_tensor->get_c();
    int IN_T_H = input_tensor->get_h();
    int IN_T_W = input_tensor->get_w();
    const std::vector<float> &input_vals = input_tensor->get_values();
    int OUT_T_N = output_tensor->get_n();
    int OUT_T_C = output_tensor->get_c();
    int OUT_T_H = output_tensor->get_h();
    int OUT_T_W = output_tensor->get_w();
    std::vector<float> &output_vals = output_tensor->get_values();
    float alpha = param_.alpha;
    float beta = param_.beta;
    float bias = param_.bias;
    int size = param_.size;
    int SIZE_BY_2 = size / 2;
    int IN_T_HW = IN_T_H * IN_T_W;
    for (int in_c = 0; in_c < IN_T_C; in_c++) {
      int idx1 = in_c * IN_T_H * IN_T_W;
      for (int in_h = 0; in_h < IN_T_H; in_h++) {
        int idx2 = idx1 + in_h * IN_T_W;
        for (int in_w = 0; in_w < IN_T_W; in_w++) {
          int idx3 = idx2 + in_w;
          float a = input_vals[idx3];
          int c_start = in_c - SIZE_BY_2;
          int c_end = in_c + SIZE_BY_2;
          int idx_start =
              c_start > 0 ? c_start * IN_T_HW : in_h * IN_T_W + in_w;
          int idx_end = c_end <= IN_T_C
                            ? c_end * IN_T_HW
                            : IN_T_C * in_h * in_w + in_h * IN_T_W + in_w;
          int idx = idx_start;
          float sum = 0;
          while (idx <= idx_end) {
            sum += input_vals[idx] * input_vals[idx];
            idx += IN_T_HW;
          }
          sum = bias + alpha * sum;
          sum = std::pow(sum, beta);
          output_vals[idx3] = a / sum;
        }
      }
    }
    return true;
}

} // namespace ti
