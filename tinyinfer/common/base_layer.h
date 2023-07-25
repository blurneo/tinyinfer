#pragma once

#include "tinyinfer/common/tensor.h"
#include "tinyinfer/common/check_macro.h"
#include "tinyinfer/common/layer_type.h"
#include <vector>
#include <ostream>
#include "tinyinfer/reflection/serialize_macro.h"

namespace ti
{

  typedef struct BaseLayerParameter
  {

  } BaseLayerParameter;

  typedef struct ProfileInfo
  {
    float time_ms;
  } ProfileInfo;
  class Serializer;
  class Deserializer;
  class BaseLayer
  {
  public:
    BaseLayer() {}
    BaseLayer(LayerType layer_type) : layer_type_(layer_type) {}
    virtual bool
    forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
            std::vector<std::shared_ptr<Tensor>> output_tensors) { return true; }
    virtual bool
    forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
            std::vector<std::shared_ptr<Tensor>> output_tensors, ProfileInfo &profile_info)
    {
      auto begin = std::chrono::high_resolution_clock::now();
      bool ret = this->forward(input_tensors, output_tensors);
      auto end = std::chrono::high_resolution_clock::now();
      auto elapsed = (std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)).count() * 1e-6;
      profile_info.time_ms = elapsed;
      return ret;
    }
    int get_layertype() { return layer_type_; }
    virtual void set_layer_name(std::string name) { layer_name_ = name; }
    virtual std::string get_layer_name() { return layer_name_; }
    virtual void set_input_names(std::vector<std::string> names)
    {
      input_names_ = names;
    }
    virtual void set_output_names(std::vector<std::string> names)
    {
      output_names_ = names;
    }
    virtual std::vector<std::string> get_input_names() const
    {
      return input_names_;
    }
    virtual std::vector<std::string> get_output_names() const
    {
      return output_names_;
    }
    virtual bool is_input_name(std::string name) const
    {
      for (auto input_name : input_names_)
      {
        if (input_name == name)
        {
          return true;
        }
      }
      return false;
    }
    virtual bool is_output_name(std::string name) const
    {
      for (auto output_name : output_names_)
      {
        if (output_name == name)
        {
          return true;
        }
      }
      return false;
    }
    virtual bool is_input(const Tensor &tensor) const
    {
      for (auto name : input_names_)
      {
        if (tensor.get_name() == name)
        {
          return true;
        }
      }
      return false;
    }
    virtual bool is_output(const Tensor &tensor) const
    {
      for (auto name : output_names_)
      {
        if (tensor.get_name() == name)
        {
          return true;
        }
      }
      return false;
    }
    virtual void serialize(Serializer &serializer);
    virtual bool deserialize(Deserializer &deserializer);
    // should be called after forward or after the input shape is predefined
    virtual uint64_t calc_computation_flops() { return flops_; }
    // should be called after forward or after the input shape is predefined
    virtual uint64_t calc_memory_bytes() { return bytes_; }

  protected:
    std::string layer_name_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    LayerType layer_type_ = LAYER_NONE;

    DEFINE_SERIALIZE_MEMBER(
        (layer_type_)(layer_name_)(input_names_)(output_names_))
    uint64_t flops_ = 0, bytes_ = 0;
  };

} // namespace ti
