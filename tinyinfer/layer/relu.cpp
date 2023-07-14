#pragma once
#include "tinyinfer/common/check_macro.h"
#include "tinyinfer/common/tensor.h"
#include "tinyinfer/layer/base_layer.h"
#include "tinyinfer/layer/relu.h"
#include "tinyinfer/net/serializer.h"
#include "tinyinfer/net/deserializer.h"

namespace ti {

bool Relu::forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
               std::vector<std::shared_ptr<Tensor>> output_tensors) {
    CHECK_BOOL_RET(input_tensors.size(), 1,
                   "Maxpool input tensor number should be 1")
    std::shared_ptr<Tensor> input_tensor = input_tensors[0];
    std::shared_ptr<Tensor> output_tensor = output_tensors[0];
    output_tensor->reshape_like(input_tensor);
    return kernel(input_tensor, output_tensor);
}

bool Relu::kernel(std::shared_ptr<Tensor> input_tensor,
              std::shared_ptr<Tensor> output_tensor) {
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
    for (int idx = 0; idx < input_vals.size(); idx++) {
      float in = input_vals[idx];
      output_vals[idx] = in > 0 ? in : 0;
    }
    return true;
}

void Relu::serialize(Serializer& serializer) {
    BaseLayer::serialize(serializer);
}

bool Relu::deserialize(Deserializer& deserializer) {
    return BaseLayer::deserialize(deserializer);
}

} // namespace ti
