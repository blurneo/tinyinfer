#pragma once

#include "tinyinfer/common/check_macro.h"
#include "tinyinfer/common/tensor.h"
#include "tinyinfer/layer/base_layer.h"
#include <vector>

namespace ti {

typedef struct GemmLayerParameter : public BaseLayerParameter {
  float alpha;
  float beta;
  bool trans_a;
  bool trans_b;
  std::shared_ptr<Tensor> weights;
  std::shared_ptr<Tensor> bias;
} GemmLayerParameter;

class Gemm : public BaseLayer {
public:
  Gemm() : BaseLayer(LAYER_GEMM) {}
  Gemm(GemmLayerParameter &&param) : param_(param), BaseLayer(LAYER_GEMM) {}

private:
  bool forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
               std::vector<std::shared_ptr<Tensor>> output_tensors) override;
  bool kernel(std::shared_ptr<Tensor> input_tensor1,
              std::shared_ptr<Tensor> input_tensor2,
              std::shared_ptr<Tensor> output_tensor);

private:
  GemmLayerParameter param_;
};

} // namespace ti
