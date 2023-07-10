#pragma once

#include <vector>
#include <iostream>
#include "common/tensor.h"

namespace ti {

typedef struct ConvolutionLayerParam {
    int kernel_shape_x;
    int kernel_shape_y;
    int stride_x;
    int stride_y;
    int pad_l;
    int pad_r;
    int pad_t;
    int pad_d;
    int group = 1;
    Tensor weights;
    Tensor bias;
} ConvolutionLayerParam;

class Convolution {
 public:
    Convolution(ConvolutionLayerParam &&param) : param_(std::move(param)) {}
    bool Forward(const Tensor &input_tensor, Tensor &output_tensor) {
        // out x: (IN_W + PadL + PadR - K_W) / Stride_X + 1
        // out y: (IN_H + PadT + PadD - K_H) / Stride_Y + 1
        Tensor padded_input_tensor = input_tensor;
        if (param_.pad_t == 0 || param_.pad_d == 0 ||
            param_.pad_l == 0 || param_.pad_r == 0) {
            Tensor::pad(input_tensor, padded_input_tensor,
                        param_.pad_t, param_.pad_d, param_.pad_l, param_.pad_r);
        }
        int out_w = (input_tensor.get_w() + param_.pad_l +
                     param_.pad_r - param_.kernel_shape_x) / param_.stride_x + 1;
        int out_h = (input_tensor.get_h() + param_.pad_t +
                     param_.pad_d - param_.kernel_shape_y) / param_.stride_y + 1;
        output_tensor.set_n(1);
        output_tensor.set_c(input_tensor.get_n());
        output_tensor.set_h(out_h);
        output_tensor.set_w(out_w);
        output_tensor.get_values().resize(input_tensor.get_n() * out_h * out_w);

        return kernel(input_tensor, output_tensor);
        // return true;
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
        int group = param_.group;
        std::vector<float> &weight_vals = param_.weights.get_values();
        int weight_n = param_.weights.get_n();
        std::vector<float> &bias_vals = param_.bias.get_values();
        // implementation
        for (int oc = 0; oc < weight_n; oc++) {
            // int idx0 = in_n * IN_T_C * IN_T_H * IN_T_W;
            float bias_val = bias_vals[oc];
            for (int in_h = 0, oh = 0; oh < OUT_T_H; in_h+=stride_y, oh++) {
                int idx1 = in_h * IN_T_W;
                for (int in_w = 0, ow = 0; ow < OUT_T_W; in_w+=stride_x, ow++) {
                    int idx2 = idx1 + in_w;
                    float res = 0;
                    for (int in_c = 0; in_c < IN_T_C; in_c++) {
                        int idx3 = in_c * (IN_T_H * IN_T_W) + idx2;
                        for (int h = 0; h < kshape_y; h++) {
                            for (int w = 0; w < kshape_x; w++) {
                                int t_idx = idx3 + h * IN_T_W + w;
                                int w_idx = oc * IN_T_C * kshape_y * kshape_x + in_c * kshape_y * kshape_x + h * kshape_x + w;
                                res += input_vals[t_idx] * weight_vals[w_idx];
                            }
                        }
                    }
                    res -= bias_val;
                    int o_idx = oc * OUT_T_H * OUT_T_W + oh * OUT_T_W + ow;
                    output_vals[o_idx] = res;
                }
            }
        }

        return true;
    }

 private:
    ConvolutionLayerParam param_;
};

}
