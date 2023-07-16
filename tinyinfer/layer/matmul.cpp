#pragma once

#include "tinyinfer/common/check_macro.h"
#include "tinyinfer/common/tensor.h"
#include "tinyinfer/common/base_layer.h"
#include "tinyinfer/layer/matmul.h"
#include <vector>
#include "tinyinfer/reflection/serializer.h"
#include "tinyinfer/reflection/deserializer.h"

namespace ti {

bool Matmul::forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
               std::vector<std::shared_ptr<Tensor>> output_tensors) {
    CHECK_BOOL_RET(input_tensors.size(), 2,
                   "Matmul input tensor number should be 2")
    CHECK_BOOL_RET(input_tensors[0]->can_multiply(input_tensors[1]), true,
                   "Two input tensors can't multiply.");
    std::shared_ptr<Tensor> output_tensor = output_tensors[0];
    output_tensor->reshape(0, 0, input_tensors[0]->get_h(),
                           input_tensors[1]->get_w());
    // calculate computation and memory infomation
    flops_ = input_tensors[0]->get_h() * input_tensors[0]->get_w() * output_tensor->get_w();
    bytes_ = input_tensors[0]->get_bytes() + input_tensors[1]->get_bytes() + output_tensor->get_bytes();
    return kernel(input_tensors[0], input_tensors[1], output_tensor);
}
bool Matmul::kernel(std::shared_ptr<Tensor> input_tensor1,
              std::shared_ptr<Tensor> input_tensor2,
              std::shared_ptr<Tensor> output_tensor) {
    int H1 = input_tensor1->get_h();
    int W1 = input_tensor1->get_w();
    const float *val_ptr1 = input_tensor1->get_values().data();
    int H2 = input_tensor2->get_h();
    int W2 = input_tensor2->get_w();
    const float *val_ptr2 = input_tensor2->get_values().data();
    float *out_ptr = output_tensor->get_values().data();
    for (int h1 = 0; h1 < H1; h1++) {
      for (int w2 = 0; w2 < W2; w2++) {
        float sum = 0.f;
        for (int w1 = 0, h2 = 0; w1 < W1; w1++, h2++) {
          sum += val_ptr1[h1 * W1 + w1] * val_ptr2[h2 * W2 + w2];
        }
        out_ptr[h1 * W2 + w2] = sum;
      }
    }
    return true;
}

void Matmul::serialize(Serializer& serializer) {
    BaseLayer::serialize(serializer);
    // serialize_internal(serializer);
}

bool Matmul::deserialize(Deserializer& deserializer) {
    return BaseLayer::deserialize(deserializer);
    // return deserialize_internal(deserializer);
}

} // namespace ti
