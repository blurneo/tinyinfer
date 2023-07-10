#pragma once
#include "tinyinfer/common/tensor.h"
#include "tinyinfer/common/check_macro.h"
#include "tinyinfer/layer/base_layer.h"

namespace ti {

typedef struct ReluLayerParam : public BaseLayerParameter {
} ReluLayerParameter;

class Relu : public BaseLayer {
 public:
    Relu(ReluLayerParameter &&param) : param_(std::move(param)), BaseLayer(LAYER_RELU) {}
    bool Forward(const std::vector<Tensor> &input_tensors, Tensor &output_tensor) override {
        CHECK_BOOL_RET(input_tensors.size(), 1, "Maxpool input tensor number should be 1")
        const Tensor &input_tensor = input_tensors[0];
        output_tensor.set_n(input_tensor.get_n());
        output_tensor.set_c(input_tensor.get_c());
        output_tensor.set_h(input_tensor.get_h());
        output_tensor.set_w(input_tensor.get_w());
        output_tensor.get_values().resize(input_tensor.get_count());
        return kernel(input_tensor, output_tensor);
    }
 private:
    bool kernel(const Tensor &input_tensor, Tensor &output_tensor) {
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
        for (int idx = 0; idx < input_vals.size(); idx++) {
            float in = input_vals[idx];
            output_vals[idx] = in > 0 ? in : 0;
        }
        return true;
    }
 private:
    ReluLayerParameter param_;
};

}
