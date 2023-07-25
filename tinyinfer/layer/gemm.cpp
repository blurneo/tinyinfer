#pragma once

#include "tinyinfer/common/check_macro.h"
#include "tinyinfer/common/tensor.h"
#include "tinyinfer/common/base_layer.h"
#include "tinyinfer/layer/gemm.h"
#include <vector>
#include "tinyinfer/reflection/serializer.h"
#include "tinyinfer/reflection/deserializer.h"

namespace ti
{

  bool Gemm::forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                     std::vector<std::shared_ptr<Tensor>> output_tensors)
  {
    CHECK_BOOL_RET(input_tensors.size(), 1,
                   "Gemm input tensor number should be 1")
    std::shared_ptr<Tensor> input_tensor = input_tensors[0];
    std::shared_ptr<Tensor> output_tensor = output_tensors[0];
    CHECK_BOOL_RET(input_tensor->dims(), 2,
                   "Gemm input tensor should have 2 dims.");
    int A_H = param_.trans_a ? input_tensor->get_w() : input_tensor->get_h();
    int A_W = param_.trans_a ? input_tensor->get_h() : input_tensor->get_w();
    int B_H = param_.trans_b ? param_.weights->get_w() : param_.weights->get_h();
    int B_W = param_.trans_b ? param_.weights->get_h() : param_.weights->get_w();
    CHECK_BOOL_RET(A_W == B_H, true, "Gemm transposed input matrice shapes not match")
    int out_h = A_H;
    int out_w = B_W;
    output_tensor->reshape(0, 0, out_h, out_w);
    CHECK_BOOL_RET(param_.bias->can_uni_broadcast(output_tensor), true, "Gemm bias shape not match with matmul result")
    // calculate computation and memory infomation
    flops_ = 2 * input_tensor->get_h() * input_tensor->get_w() * output_tensor->get_w();
    bytes_ = input_tensor->get_bytes() + param_.weights->get_bytes() + output_tensor->get_bytes();
    return kernel(input_tensors[0], param_.weights, output_tensor);
  }

  bool Gemm::kernel(std::shared_ptr<Tensor> input_tensor1,
                    std::shared_ptr<Tensor> input_tensor2,
                    std::shared_ptr<Tensor> output_tensor)
  {
    int H1 = input_tensor1->get_h();
    int W1 = input_tensor1->get_w();
    const float *val_ptr1 = input_tensor1->get_values().data();
    int H2 = input_tensor2->get_h();
    int W2 = input_tensor2->get_w();
    const float *val_ptr2 = input_tensor2->get_values().data();
    int BH = param_.bias->get_h();
    int BW = param_.bias->get_w();
    const float *bias_ptr = param_.bias->get_values().data();
    float *out_ptr = output_tensor->get_values().data();
    for (int h1 = 0; h1 < H1; h1++)
    {
      for (int w2 = 0; w2 < W2; w2++)
      {
        float sum = 0.f;
        for (int w1 = 0, h2 = 0; w1 < W1; w1++, h2++)
        {
          sum += val_ptr1[h1 * W1 + w1] * val_ptr2[h2 * W2 + w2];
        }
        int bias_idx = std::min(BH, h1) * BH + std::min(BW, w2);
        out_ptr[h1 * W2 + w2] = param_.alpha * sum + bias_ptr[bias_idx];
      }
    }
    return true;
  }

  void Gemm::serialize(Serializer &serializer)
  {
    BaseLayer::serialize(serializer);
    Gemm::serialize_internal(serializer);
  }

  bool Gemm::deserialize(Deserializer &deserializer)
  {
    CHECK_BOOL_RET(BaseLayer::deserialize(deserializer), true, "Gemm baselayer deserialize failed");
    return Gemm::deserialize_internal(deserializer);
  }

} // namespace ti
