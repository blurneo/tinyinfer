#pragma once

#include "tinyinfer/common/check_macro.h"
#include "tinyinfer/common/tensor.h"
#include "tinyinfer/layer/base_layer.h"

namespace ti {

typedef struct GlobalAveragePoolLayerParameter : public BaseLayerParameter {
} GlobalAveragePoolLayerParameter;

class GlobalAveragePool : public BaseLayer {
public:
  GlobalAveragePool(GlobalAveragePoolLayerParameter &&param)
      : param_(std::move(param)), BaseLayer(LAYER_GLOBAL_AVERAGE_POOL) {}
  bool forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
               std::vector<std::shared_ptr<Tensor>> output_tensors) override {
    CHECK_BOOL_RET(input_tensors.size(), 1,
                   "GlobalAveragePool input tensor number should be 1")
    const std::shared_ptr<Tensor> &input_tensor = input_tensors[0];
    std::shared_ptr<Tensor> output_tensor = output_tensors[0];
    output_tensor->reshape(input_tensor->get_n(), input_tensor->get_c(), 1, 1);

    return kernel(input_tensor, output_tensor);
  }

private:
  bool kernel(std::shared_ptr<Tensor> input_tensor,
              std::shared_ptr<Tensor> output_tensor) {
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
    int OUT_HxW = OUT_T_H * OUT_T_W;
    float OUT_HxW_RECIP = 1.0 / OUT_HxW;
    // implementation
    for (int in_n = 0; in_n < IN_T_N; in_n++) {
      int idx0 = in_n * IN_T_C * IN_T_H * IN_T_W;
      int oidx0 = in_n * IN_T_C;
      for (int in_c = 0; in_c < IN_T_C; in_c++) {
        int idx1 = idx0 + in_c * IN_T_H * IN_T_W;
        int oidx1 = oidx0 + in_c;
        float sum = 0.f;
        for (int in_h = 0; in_h < IN_T_H; in_h++) {
          int idx2 = idx1 + in_h * IN_T_W;
          for (int in_w = 0; in_w < IN_T_W; in_w++) {
            int idx3 = idx2 + in_w;
            sum += input_vals[idx3];
          }
        }
        output_vals[oidx1] = sum * OUT_HxW_RECIP;
      }
    }

    return true;
  }

private:
  GlobalAveragePoolLayerParameter param_;
};

} // namespace ti
