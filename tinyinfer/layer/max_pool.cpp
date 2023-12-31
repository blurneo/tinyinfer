#pragma once

#include "tinyinfer/common/check_macro.h"
#include "tinyinfer/common/tensor.h"
#include "tinyinfer/common/base_layer.h"
#include "tinyinfer/layer/max_pool.h"
#include "tinyinfer/reflection/serializer.h"
#include "tinyinfer/reflection/deserializer.h"

namespace ti
{

  bool MaxPool::forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                        std::vector<std::shared_ptr<Tensor>> output_tensors)
  {
    CHECK_BOOL_RET(input_tensors.size(), 1,
                   "Maxpool input tensor number should be 1")
    const std::shared_ptr<Tensor> &input_tensor = input_tensors[0];
    std::shared_ptr<Tensor> output_tensor = output_tensors[0];
    // out x: (IN_W + PadL + PadR - K_W) / Stride_X + 1
    // out y: (IN_H + PadT + PadD - K_H) / Stride_Y + 1
    int out_w = (input_tensor->get_w() + param_.pad_l + param_.pad_r -
                 param_.kernel_shape_x) /
                    param_.stride_x +
                1;
    int out_h = (input_tensor->get_h() + param_.pad_t + param_.pad_d -
                 param_.kernel_shape_y) /
                    param_.stride_y +
                1;
    output_tensor->reshape(input_tensor->get_n(), input_tensor->get_c(), out_h,
                           out_w);
    // calculate computation and memory infomation
    flops_ = 0;
    bytes_ = input_tensor->get_bytes() + output_tensor->get_bytes();
    return kernel(input_tensor, output_tensor);
  }

  bool MaxPool::kernel(std::shared_ptr<Tensor> input_tensor,
                       std::shared_ptr<Tensor> output_tensor)
  {
    // param def
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
    int kshape_x = param_.kernel_shape_x, kshape_y = param_.kernel_shape_y;
    int stride_x = param_.stride_x, stride_y = param_.stride_x;
    int pad_l = param_.pad_l, pad_r = param_.pad_r, pad_t = param_.pad_t,
        pad_d = param_.pad_d;
    // implementation
    // int idx0 = in_n * IN_T_C * IN_T_H * IN_T_W;
    for (int in_c = 0; in_c < IN_T_C; in_c++)
    {
      int idx1 = in_c * IN_T_H * IN_T_W;
      for (int in_h = 0, oh = 0; oh < OUT_T_H; in_h += stride_y, oh++)
      {
        int idx2 = idx1 + in_h * IN_T_W;
        for (int in_w = 0, ow = 0; ow < OUT_T_W; in_w += stride_x, ow++)
        {
          int idx3 = idx2 + in_w;
          float max = input_vals[idx3];
          for (int h = 0; h < kshape_y; h++)
          {
            for (int w = 0; w < kshape_x; w++)
            {
              int t_idx = idx3 + h * IN_T_W + w;
              max = max > input_vals[t_idx] ? max : input_vals[t_idx];
            }
          }
          int o_idx = in_c * OUT_T_H * OUT_T_W + oh * OUT_T_W + ow;
          output_vals[o_idx] = max;
        }
      }
    }

    return true;
  }

  void MaxPool::serialize(Serializer &serializer)
  {
    BaseLayer::serialize(serializer);
    MaxPool::serialize_internal(serializer);
  }

  bool MaxPool::deserialize(Deserializer &deserializer)
  {
    CHECK_BOOL_RET(BaseLayer::deserialize(deserializer), true, "Maxpool baselayer deserialize failed");
    return MaxPool::deserialize_internal(deserializer);
  }

} // namespace ti
