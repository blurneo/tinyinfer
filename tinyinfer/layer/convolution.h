#pragma once

#include <vector>
#include <iostream>
#include "tinyinfer/common/tensor.h"
#include "tinyinfer/common/check_macro.h"
#include "tinyinfer/layer/base_layer.h"

namespace ti {

typedef struct ConvolutionLayerParam : public BaseLayerParameter {
    int kernel_shape_x = 0;
    int kernel_shape_y = 0;
    int stride_x = 1;
    int stride_y = 1;
    int pad_t = 0; // first of 4 pads values in onnx
    int pad_d = 0; // second of 4 pads values in onnx
    int pad_l = 0; // third of 4 pads values in onnx
    int pad_r = 0; // fourth of 4 pads values in onnx
    int group = 1;
    int dilation_x = 1;
    int dilation_y = 1;
    int pad_type = 0; // 0: NotSet, 1: SameUpper, 2: SameLower, 3: Valid;
    std::shared_ptr<Tensor> weights;
    std::shared_ptr<Tensor> bias;
} ConvolutionLayerParameter;

class Convolution : public BaseLayer {
 public:
    Convolution(ConvolutionLayerParameter &&param) : param_(std::move(param)), BaseLayer(LAYER_CONVOLUTION) {}
    void get_pad(int w_h, int w_w, int s_h, int s_w, int pad_type,
                 int &pad_t, int& pad_d, int& pad_l, int &pad_r) {
        if (pad_type == 0) return;
        int pad_h = (int)(std::ceil(w_h / s_h));
        int pad_w = (int)(std::ceil(w_w / s_w));
        bool h_even = pad_h % 2 == 0;
        bool w_even = pad_w % 2 == 0;
        if (h_even) {
            pad_t = pad_d = pad_h / 2;
        } else {
            pad_t = pad_h / 2;
            pad_d = pad_h - pad_t;
        }
        if (w_even) {
            pad_l = pad_r = pad_w / 2;
        } else {
            pad_l = pad_w / 2;
            pad_r = pad_w - pad_l;
        }
        if (pad_type == 1) {
        } else if (pad_type == 2) {
            std::swap(pad_t, pad_d);
            std::swap(pad_l, pad_r);
        } else if (pad_type == 3) {
            std::cerr << "Not implemented valid type padding\n";
        }
    }
    bool forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                 std::vector<std::shared_ptr<Tensor>> output_tensors) override {
        CHECK_BOOL_RET(input_tensors.size(), 1, "Convolution input tensor number should be 1")
        const std::shared_ptr<Tensor> &input_tensor = input_tensors[0];
        std::shared_ptr<Tensor> output_tensor = output_tensors[0];
        std::shared_ptr<Tensor> padded_input_tensor = input_tensor;
        int pad_t = param_.pad_t, pad_d = param_.pad_d, pad_l = param_.pad_l, pad_r = param_.pad_r;
        get_pad(input_tensor->get_h(), input_tensor->get_w(), param_.stride_y, param_.stride_x,
            param_.pad_type, pad_t, pad_d, pad_l, pad_r);
        if (pad_t != 0 || pad_d != 0 || pad_l != 0 || pad_r != 0) {
            padded_input_tensor->reshape_like(input_tensor);
            Tensor::pad(input_tensor, padded_input_tensor, pad_t, pad_d, pad_l, pad_r);
        }
        // out x: (IN_W + PadL + PadR - K_W) / Stride_X + 1
        // out y: (IN_H + PadT + PadD - K_H) / Stride_Y + 1
        int out_w = (input_tensor->get_w() + pad_l +
                     pad_r - param_.kernel_shape_x) / param_.stride_x + 1;
        int out_h = (input_tensor->get_h() + pad_t +
                     pad_d - param_.kernel_shape_y) / param_.stride_y + 1;
        output_tensor->set_n(1);
        output_tensor->set_c(input_tensor->get_n());
        output_tensor->set_h(out_h);
        output_tensor->set_w(out_w);
        output_tensor->get_values().resize(input_tensor->get_n() * out_h * out_w);

        return kernel(input_tensor, output_tensor);
        // return true;
    }

 private:
    bool kernel(const std::shared_ptr<Tensor> &input_tensor, std::shared_ptr<Tensor> output_tensor) {
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
        // int pad_l = param_.pad_l, pad_r = param_.pad_r, pad_t = param_.pad_t, pad_d = param_.pad_d;
        int group = param_.group;
        std::vector<float> &weight_vals = param_.weights->get_values();
        int weight_n = param_.weights->get_n();
        std::vector<float> &bias_vals = param_.bias->get_values();
        bool do_bias = true;
        if (bias_vals.empty()) do_bias = false;
        // implementation
        for (int oc = 0; oc < weight_n; oc++) {
            // int idx0 = in_n * IN_T_C * IN_T_H * IN_T_W;
            float bias_val = 0;
            if (do_bias) bias_val = bias_vals[oc];
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
                    if (do_bias) res -= bias_val;
                    int o_idx = oc * OUT_T_H * OUT_T_W + oh * OUT_T_W + ow;
                    output_vals[o_idx] = res;
                }
            }
        }

        return true;
    }

 private:
    ConvolutionLayerParameter param_;
};

}
