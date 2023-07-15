#pragma once

#include "tinyinfer/common/tensor.h"
#include "tinyinfer/layer/layer_type.h"
#include <vector>
#include <ostream>
#include "tinyinfer/reflection/serialize_macro.h"

namespace ti {

typedef struct BaseLayerParameter {

} BaseLayerParameter;

class Serializer;
class Deserializer;
class BaseLayer {
public:
  BaseLayer() {}
  BaseLayer(LayerType layer_type) : layer_type_(layer_type) {}
  virtual bool
  forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
          std::vector<std::shared_ptr<Tensor>> output_tensors) { return true; }
  int get_layertype() { return layer_type_; }
  virtual void set_layer_name(std::string name) { layer_name_ = name; }
  virtual std::string get_layer_name() { return layer_name_; }
  virtual void set_input_names(std::vector<std::string> names) {
    input_names_ = names;
  }
  virtual void set_output_names(std::vector<std::string> names) {
    output_names_ = names;
  }
  virtual std::vector<std::string> get_input_names() const {
    return input_names_;
  }
  virtual std::vector<std::string> get_output_names() const {
    return output_names_;
  }
  virtual bool is_input_name(std::string name) const {
    for (auto input_name : input_names_) {
      if (input_name == name) {
        return true;
      }
    }
    return false;
  }
  virtual bool is_output_name(std::string name) const {
    for (auto output_name : output_names_) {
      if (output_name == name) {
        return true;
      }
    }
    return false;
  }
  virtual bool is_input(const Tensor &tensor) const {
    for (auto name : input_names_) {
      if (tensor.get_name() == name) {
        return true;
      }
    }
    return false;
  }
  virtual bool is_output(const Tensor &tensor) const {
    for (auto name : output_names_) {
      if (tensor.get_name() == name) {
        return true;
      }
    }
    return false;
  }
  virtual void serialize(Serializer& serializer);
  virtual bool deserialize(Deserializer& deserializer);

protected:
  std::string layer_name_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  LayerType layer_type_ = LAYER_NONE;

  DEFINE_SERIALIZE_MEMBER(
    ("layer_type", layer_type_)
    ("layer_name", layer_name_)
    ("input_names", input_names_)
    ("output_names", output_names_)
  )
};

} // namespace ti
