#pragma once

#include "tinyinfer/common/check_macro.h"
#include "tinyinfer/common/tensor.h"
#include "tinyinfer/common/base_layer.h"
#include "tinyinfer/layer/reshape.h"
#include <vector>
#include "tinyinfer/reflection/serializer.h"
#include "tinyinfer/reflection/deserializer.h"

namespace ti
{

  bool Reshape::forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                        std::vector<std::shared_ptr<Tensor>> output_tensors)
  {
    // CHECK_BOOL_RET(input_tensors.size(), 1, "Reshape input tensor number
    // should be 1")
    std::shared_ptr<Tensor> input_tensor;
    if (input_tensors.empty())
    {
      CHECK_BOOL_RET(param_.data != nullptr, true,
                     "Reshape input tensor number should be 1")
      input_tensor = param_.data;
    }
    else
    {
      input_tensor = input_tensors[0];
    }
    std::shared_ptr<Tensor> output_tensor = output_tensors[0];
    CHECK_BOOL_RET(param_.shape.size() > 0, true,
                   "Reshape layer shape param is empty");
    CHECK_BOOL_RET(param_.shape.size() <= 4, true,
                   "Reshape layer shape param is too large");
    int count = 1;
    for (auto val : param_.shape)
    {
      count *= val;
    }
    CHECK_BOOL_RET(count == input_tensor->get_count(), true,
                   "Reshape layer input count not same with param");
    CHECK_BOOL_RET(kernel(input_tensor, output_tensor), true, "reshape kernel executes failed");
    // calculate computation and memory infomation
    flops_ = 0;
    bytes_ = input_tensor->get_bytes() + output_tensor->get_bytes();
    return true;
  }

  bool Reshape::kernel(std::shared_ptr<Tensor> input_tensor,
                       std::shared_ptr<Tensor> output_tensor)
  {
    int shape_size = param_.shape.size();
    switch (shape_size)
    {
    case 1:
      output_tensor->reshape(0, 0, 0, param_.shape[0]);
      break;
    case 2:
      output_tensor->reshape(0, 0, param_.shape[0], param_.shape[1]);
      break;
    case 3:
      output_tensor->reshape(0, param_.shape[0], param_.shape[1],
                             param_.shape[2]);
      break;
    case 4:
      output_tensor->reshape(param_.shape[0], param_.shape[1], param_.shape[2],
                             param_.shape[3]);
      break;
    default:
      break;
    }
    output_tensor->copy_if_same_count(input_tensor);
    return true;
  }

  void Reshape::serialize(Serializer &serializer)
  {
    BaseLayer::serialize(serializer);
    Reshape::serialize_internal(serializer);
  }

  bool Reshape::deserialize(Deserializer &deserializer)
  {
    CHECK_BOOL_RET(BaseLayer::deserialize(deserializer), true, "Reshape baselayer deserialize failed");
    return Reshape::deserialize_internal(deserializer);
  }

} // namespace ti
