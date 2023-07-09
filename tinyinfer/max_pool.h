#pragma once

#include "tensor.h"

namespace ti {

typedef struct MaxPoolLayerParam {
    int kernel_shape_x;
    int kernel_shape_y;
    int stride_x;
    int stride_y;
    int pad_l;
    int pad_r;
    int pad_t;
    int pad_d;
} MaxPoolLayerParam;

class MaxPool {
 public:
    MaxPool(MaxPoolLayerParam &&param) : param_(std::move(param)) {}
    bool Forward(const Tensor &input_tensor, Tensor &output_tensor) {
        // out x: (IN_W + PadL + PadR - K_W) / Stride_X + 1
        // out y: (IN_H + PadT + PadD - K_H) / Stride_Y + 1
        int out_w = (input_tensor.get_w() + param_.pad_l +
                     param_.pad_r - param_.kernel_shape_x) / param_.stride_x + 1;
        int out_h = (input_tensor.get_h() + param_.pad_t +
                     param_.pad_d - param_.kernel_shape_y) / param_.stride_y + 1;
        output_tensor.set_n(input_tensor.get_n());
        output_tensor.set_c(input_tensor.get_c());
        output_tensor.set_h(out_h);
        output_tensor.set_w(out_w);
        output_tensor.get_values().resize(input_tensor.get_n() * input_tensor.get_c() * out_h * out_w);

        return kernel(input_tensor, output_tensor);
    }

 private:
    bool kernel(const Tensor &input_tensor, Tensor &output_tensor) {
        // param def
        int IN_T_N = input_tensor.get_n();
        int IN_T_C = input_tensor.get_c();
        int IN_T_H = input_tensor.get_h();
        int IN_T_W = input_tensor.get_w();
        const std::vector<float> &input_vals = input_tensor.get_values();
        int OUT_T_N = output_tensor.get_n();
        int OUT_T_C = output_tensor.get_c();
        int OUT_T_H = output_tensor.get_h();
        int OUT_T_W = output_tensor.get_w();
        std::vector<float> &output_vals = output_tensor.get_values();
        int kshape_x = param_.kernel_shape_x, kshape_y = param_.kernel_shape_y;
        int stride_x = param_.stride_x, stride_y = param_.stride_x;
        int pad_l = param_.pad_l, pad_r = param_.pad_r, pad_t = param_.pad_t, pad_d = param_.pad_d;
        // implementation
        // int idx0 = in_n * IN_T_C * IN_T_H * IN_T_W;
        for (int in_c = 0; in_c < IN_T_C; in_c++) {
            int idx1 = in_c * IN_T_H * IN_T_W;
            for (int in_h = 0, oh = 0; oh < OUT_T_H; in_h+=stride_y, oh++) {
                int idx2 = idx1 + in_h * IN_T_W;
                for (int in_w = 0, ow = 0; ow < OUT_T_W; in_w+=stride_x, ow++) {
                    int idx3 = idx2 + in_w;
                    float max = input_vals[idx3];
                    for (int h = 0; h < kshape_y; h++) {
                        for (int w = 0; w < kshape_x; w++) {
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

 private:
    MaxPoolLayerParam param_;
};

}