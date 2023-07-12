#pragma once
#include "tinyinfer/common/tensor.h"
#include "tinyinfer/common/check_macro.h"
#include "tinyinfer/layer/base_layer.h"

namespace ti {

typedef struct AddnLayerParameter : public BaseLayerParameter {
    std::shared_ptr<Tensor> weights;
} AddLayerParameter;

class Add : public BaseLayer {
 public:
    Add(AddLayerParameter &&param) : param_(std::move(param)), BaseLayer(LAYER_ADD) {}
    bool is_compatible(const std::shared_ptr<Tensor>& input_tensor, std::vector<int> &broadcast_shapes) {
        if (input_tensor->is_alike(param_.weights)) {
            broadcast_shapes = input_tensor->dims_vector();
            return true;
        }
        auto input_dims_vec = input_tensor->dims_vector();
        auto output_dims_vec = param_.weights->dims_vector();
        std::vector<int> ret_dims_vec;
        int ret_dims_size = std::max(input_dims_vec.size(), output_dims_vec.size());
        for (int i = input_dims_vec.size(); i < ret_dims_size; i++) {
            input_dims_vec.insert(input_dims_vec.begin(), 1);
        }
        for (int i = output_dims_vec.size(); i < ret_dims_size; i++) {
            output_dims_vec.insert(output_dims_vec.begin(), 1);
        }
        for (int i = 0; i < ret_dims_size; i++) {
            if (input_dims_vec[i] != output_dims_vec[i]) {
                if (input_dims_vec[i] == 1 || output_dims_vec[i] == 1) {
                    ret_dims_vec.push_back(input_dims_vec[i] * output_dims_vec[i]);
                } else {
                    return false;
                }
            } else {
                ret_dims_vec.push_back(input_dims_vec[i]);
            }
        }
        broadcast_shapes = ret_dims_vec;
        return true;
    }
    bool forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                 std::vector<std::shared_ptr<Tensor>> output_tensors) override {
        CHECK_BOOL_RET(input_tensors.size(), 1, "Add input tensor number should be 1")
        const std::shared_ptr<Tensor> &input_tensor = input_tensors[0];
        std::shared_ptr<Tensor> output_tensor = output_tensors[0];
        std::vector<int> broadcast_shapes;
        bool compatible = is_compatible(input_tensor, broadcast_shapes);
        CHECK_BOOL_RET(compatible, true, "Add input tensor not compatible with weights");
        output_tensor->reshape(broadcast_shapes);
        return kernel(input_tensor, output_tensor);
    }
 private:
    bool kernel(const std::shared_ptr<Tensor> &input_tensor, std::shared_ptr<Tensor> output_tensor) {
        int IN_T_N = input_tensor->get_n();
        int IN_T_C = input_tensor->get_c();
        int IN_T_H = input_tensor->get_h();
        int IN_T_W = input_tensor->get_w();
        const std::vector<float> &input_vals = input_tensor->get_values();
        int W_T_N = param_.weights->get_n();
        int W_T_C = param_.weights->get_c();
        int W_T_H = param_.weights->get_h();
        int W_T_W = param_.weights->get_w();
        const std::vector<float> &weight_vals = param_.weights->get_values();
        int OUT_T_N = output_tensor->get_n();
        int OUT_T_C = output_tensor->get_c();
        int OUT_T_H = output_tensor->get_h();
        int OUT_T_W = output_tensor->get_w();
        std::vector<float> &output_vals = output_tensor->get_values();
        int LOOP_OUT_T_N = std::max(OUT_T_N, 1);
        int LOOP_OUT_T_C = std::max(OUT_T_C, 1);
        int LOOP_OUT_T_H = std::max(OUT_T_H, 1);
        int LOOP_OUT_T_W = std::max(OUT_T_W, 1);
        // auto input_dims_vec = input_tensor->dims_vector();
        // const std::vector<float> &input_vals = input_tensor->get_values();
        // auto weight_dims_vec = param_.weights->dims_vector();
        // const std::vector<float> &weight_vals = param_.weights->get_values();
        // auto output_dims_vec = output_tensor->dims_vector();
        // std::vector<float> &output_vals = output_tensor->get_values();
        for (int out_n = 0, in_n = 0, w_n = 0; out_n < LOOP_OUT_T_N; out_n++, in_n++, w_n++) {
            in_n = std::max(0, std::min(IN_T_N-1, in_n));
            w_n = std::max(0, std::min(W_T_N-1, w_n));
            for (int out_c = 0, in_c = 0, w_c = 0; out_c < LOOP_OUT_T_C; out_c++, in_c++, w_c++) {
                in_c = std::max(0, std::min(IN_T_C-1, in_c));
                w_c = std::max(0, std::min(W_T_C-1, w_c));
                for (int out_h = 0, in_h = 0, w_h = 0; out_h < LOOP_OUT_T_H; out_h++, in_h++, w_h++) {
                    in_h = std::max(0, std::min(IN_T_H-1, in_h));
                    w_h = std::max(0, std::min(W_T_H-1, w_h));
                    for (int out_w = 0, in_w = 0, w_w = 0; out_w < LOOP_OUT_T_W; out_w++, in_w++, w_w++) {
                        in_w = std::max(0, std::min(IN_T_W-1, in_w));
                        w_w = std::max(0, std::min(W_T_W-1, w_w));
                        int in_idx = in_n * IN_T_C * IN_T_H * IN_T_W + in_c * IN_T_H * IN_T_W + in_h * IN_T_W + in_w;
                        int w_idx = w_n * W_T_C * W_T_H * W_T_W + w_c * W_T_H * W_T_W + w_h * W_T_W + w_w;
                        int o_idx = out_n * OUT_T_C * OUT_T_H * OUT_T_W + out_c * OUT_T_H * OUT_T_W + out_h * OUT_T_W + out_w;
                        output_vals[o_idx] = input_vals[in_idx] + weight_vals[w_idx];
                    }
                }
            }
        }
        return true;
    }
 private:
    AddLayerParameter param_;
};

}
