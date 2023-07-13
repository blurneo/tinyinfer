#pragma once
#include <cmath>
#include "tinyinfer/common/tensor.h"
#include "tinyinfer/common/check_macro.h"
#include "tinyinfer/layer/base_layer.h"

namespace ti {

typedef struct SoftmaxLayerParam : public BaseLayerParameter {
    int axis = -1;
} SoftmaxLayerParameter;

class Softmax : public BaseLayer {
 public:
    Softmax(SoftmaxLayerParameter &&param) : param_(std::move(param)), BaseLayer(LAYER_SOFTMAX) {}
    bool forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors, std::vector<std::shared_ptr<Tensor>> output_tensors) override {
        CHECK_BOOL_RET(input_tensors.size(), 1, "Maxpool input tensor number should be 1")
        std::shared_ptr<Tensor> input_tensor = input_tensors[0];
        std::shared_ptr<Tensor> output_tensor = output_tensors[0];
        output_tensor->reshape_like(input_tensor);
        return kernel(input_tensor, output_tensor);
    }
 private:
    bool kernel(std::shared_ptr<Tensor> input_tensor, std::shared_ptr<Tensor> output_tensor) {
        auto& input_values = input_tensor->get_values();
        auto &output_values = output_tensor->get_values();
        auto input_dims_vec = input_tensor->dims_vector();
        int axis_idx = param_.axis < 0 ? param_.axis + input_dims_vec.size() : param_.axis;
        std::optional<int> stride = input_tensor->dim_stride(axis_idx);
        CHECK_BOOL_RET(stride.has_value(), true, "Softmax calculate stride failed\n");
        int dim_from_idx = input_dims_vec[axis_idx];
        int idx = 0;
        while (idx < output_values.size()) {
            float sum = 0.f;
            for (int i = 0; i < dim_from_idx; i++) {
                int idx = i * stride.value();
                output_values[idx] = std::exp(input_values[idx]);
                sum += output_values[idx];
            }
            for (int i = 0; i < dim_from_idx; i++) {
                int idx = i * stride.value();
                output_values[idx] = output_values[idx] / sum;
            }
        }

        return true;
    }
 private:
    SoftmaxLayerParameter param_;
};

}
