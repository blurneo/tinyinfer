#pragma once
#include <ostream>
#include "tinyinfer/common/check_macro.h"
#include "tinyinfer/reflection/serialize_macro.h"
#include "tinyinfer/common/tensor.h"
#include "tinyinfer/common/base_layer.h"

namespace ti
{

  typedef struct AddLayerParameter : public BaseLayerParameter
  {
    std::shared_ptr<Tensor> weights;
    DEFINE_SERIALIZE_MEMBER(
        (weights))
  } AddLayerParameter;

  class Serializer;
  class Deserializer;
  class Add : public BaseLayer
  {
  public:
    Add() : BaseLayer(LAYER_ADD) {}
    Add(AddLayerParameter &&param)
        : param_(std::move(param)), BaseLayer(LAYER_ADD) {}
    bool is_compatible(const std::shared_ptr<Tensor> &input_tensor,
                       std::vector<int> &broadcast_shapes);
    bool forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                 std::vector<std::shared_ptr<Tensor>> output_tensors) override;
    virtual void serialize(Serializer &serializer);
    virtual bool deserialize(Deserializer &deserializer);

  private:
    bool kernel(const std::shared_ptr<Tensor> &input_tensor,
                std::shared_ptr<Tensor> output_tensor);

  private:
    AddLayerParameter param_;
    DEFINE_SERIALIZE_MEMBER(
        (&param_))
  };

} // namespace ti
