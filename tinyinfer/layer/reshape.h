#pragma once

#include "tinyinfer/common/check_macro.h"
#include "tinyinfer/common/tensor.h"
#include "tinyinfer/layer/base_layer.h"
#include <vector>

namespace ti {

typedef struct ReshapeLayerParameter : public BaseLayerParameter {
  std::shared_ptr<Tensor> data;
  std::vector<unsigned long> shape;
} ReshapeLayerParameter;

class Reshape : public BaseLayer {
public:
  Reshape(ReshapeLayerParameter &&param)
      : param_(param), BaseLayer(LAYER_RESHAPE) {}
  bool forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
               std::vector<std::shared_ptr<Tensor>> output_tensors) override {
    // CHECK_BOOL_RET(input_tensors.size(), 1, "Reshape input tensor number
    // should be 1")
    std::shared_ptr<Tensor> input_tensor;
    if (input_tensors.empty()) {
      CHECK_BOOL_RET(param_.data != nullptr, true,
                     "Reshape input tensor number should be 1")
      input_tensor = param_.data;
    } else {
      input_tensor = input_tensors[0];
    }
    std::shared_ptr<Tensor> output_tensor = output_tensors[0];
    CHECK_BOOL_RET(param_.shape.size() > 0, true,
                   "Reshape layer shape param is empty");
    CHECK_BOOL_RET(param_.shape.size() <= 4, true,
                   "Reshape layer shape param is too large");
    int count = 1;
    for (auto val : param_.shape) {
      count *= val;
    }
    CHECK_BOOL_RET(count == input_tensor->get_count(), true,
                   "Reshape layer input count not same with param");
    return kernel(input_tensor, output_tensor);
  }

private:
  bool kernel(std::shared_ptr<Tensor> input_tensor,
              std::shared_ptr<Tensor> output_tensor) {
    int shape_size = param_.shape.size();
    switch (shape_size) {
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
  ReshapeLayerParameter param_;
};

} // namespace ti
