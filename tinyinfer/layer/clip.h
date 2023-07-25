#pragma once
#include "tinyinfer/common/check_macro.h"
#include "tinyinfer/common/tensor.h"
#include "tinyinfer/common/base_layer.h"

namespace ti
{

  typedef struct ClipLayerParam : public BaseLayerParameter
  {
    float min;
    float max;
  } ClipLayerParameter;

  class Clip : public BaseLayer
  {
  public:
    Clip() : BaseLayer(LAYER_CLIP) {}
    Clip(ClipLayerParameter &&param)
        : param_(std::move(param)), BaseLayer(LAYER_CLIP) {}
    bool forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                 std::vector<std::shared_ptr<Tensor>> output_tensors) override;

  private:
    bool kernel(std::shared_ptr<Tensor> input_tensor,
                std::shared_ptr<Tensor> output_tensor);

  private:
    ClipLayerParameter param_;
  };

} // namespace ti
