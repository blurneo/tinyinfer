#pragma once

#include "tinyinfer/common/check_macro.h"
#include "tinyinfer/common/tensor.h"
#include "tinyinfer/common/base_layer.h"
#include "tinyinfer/reflection/serialize_macro.h"

namespace ti {

typedef struct MaxPoolLayerParameter : public BaseLayerParameter {
  int kernel_shape_x;
  int kernel_shape_y;
  int stride_x;
  int stride_y;
  int pad_t;
  int pad_d;
  int pad_l;
  int pad_r;
  DEFINE_SERIALIZE_MEMBER(
    ("kshapex", kernel_shape_x)
    ("kshapey", kernel_shape_y)
    ("stridex", stride_x)
    ("stridey", stride_y)
    ("padt", pad_t)
    ("padd", pad_d)
    ("padl", pad_l)
    ("padr", pad_r)
  )
} MaxPoolLayerParameter;

class Serializer;
class Deserializer;
class MaxPool : public BaseLayer {
public:
  MaxPool() : BaseLayer(LAYER_MAXPOOL) {}
  MaxPool(MaxPoolLayerParameter &&param)
      : param_(std::move(param)), BaseLayer(LAYER_MAXPOOL) {}
  bool forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
               std::vector<std::shared_ptr<Tensor>> output_tensors) override;
  virtual void serialize(Serializer &serializer);
  virtual bool deserialize(Deserializer& deserializer);
private:
  bool kernel(std::shared_ptr<Tensor> input_tensor,
              std::shared_ptr<Tensor> output_tensor);
private:
  MaxPoolLayerParameter param_;
  DEFINE_SERIALIZE_MEMBER(
    ("param_", &param_)
  )
};

} // namespace ti
