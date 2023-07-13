#pragma once

#include "tinyinfer/common/tensor.h"
#include <vector>
#include <ostream>
#include "tinyinfer/net/serializer.h"

namespace ti {

typedef enum LayerType {
  LAYER_ADD = 0,
  LAYER_CONVOLUTION = 1,
  LAYER_LRN = 2,
  LAYER_MATMUL = 3,
  LAYER_MAXPOOL = 4,
  LAYER_RELU = 5,
  LAYER_RESHAPE = 6,
  LAYER_SOFTMAX = 7,
  LAYER_CLIP = 8,
  LAYER_GLOBAL_AVERAGE_POOL = 9,
  LAYER_FLATTEN = 10,
  LAYER_BATCH_NORMALIZATION = 11,
  LAYER_GEMM = 12
} LayerType;

typedef struct BaseLayerParameter {

} BaseLayerParameter;

class BaseLayer {
public:
  BaseLayer(LayerType layer_type) : layer_type_(layer_type) {}
  virtual bool
  forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
          std::vector<std::shared_ptr<Tensor>> output_tensors) = 0;
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
  void serialize(Serializer& serializer) {
    serialize_internal(serializer);
  }

protected:
  std::string layer_name_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  LayerType layer_type_;
  #define DEFINE_SERIALIZE_MEMBER(x) template<class R> void serialize_internal(R &r) { r.begin(); r.operator()x; r.end(); }
  DEFINE_SERIALIZE_MEMBER(
    ("layer_name", layer_name_)
    ("layer_type", (int)layer_type_)
    ("input_names", input_names_)
    ("output_names", output_names_)
  )
};

} // namespace ti
