#pragma once
#include "tinyinfer/common/tensor.h"
#include "tinyinfer/common/check_macro.h"
#include "tinyinfer/layer/base_layer.h"

namespace ti {

typedef struct AddnLayerParameter : public BaseLayerParameter {
    Tensor weights;
} AddLayerParameter;

class Add : public BaseLayer {
 public:
    Add(AddLayerParameter &&param) : param_(std::move(param)), BaseLayer(LAYER_ADD) {}
    bool Forward(const std::vector<Tensor> &input_tensors, Tensor &output_tensor) override {
        CHECK_BOOL_RET(input_tensors.size(), 1, "Add input tensor number should be 1")
        const Tensor &input_tensor = input_tensors[0];
        CHECK_BOOL_RET(input_tensor.is_alike(param_.weights), true,
            "Add input tensor not alike with weights");
        output_tensor.reshape_like(input_tensor);
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
            output_vals[idx] = input_vals[idx] + param_.weights.get_values()[idx];
        }
        return true;
    }
 private:
    AddLayerParameter param_;
};

}
