#pragma once

#include "tinyinfer/common/tensor.h"
#include "tinyinfer/common/base_layer.h"
#include <map>
#include <memory>
#include <set>
#include <string>
#include "tinyinfer/reflection/serializer.h"
#include "tinyinfer/reflection/deserializer.h"

namespace ti
{

  class Graph;
  class Net
  {
  public:
    bool register_layer(std::string name, std::shared_ptr<BaseLayer> layer);
    void set_input_name(std::string name);
    std::string get_input_name() const;
    const std::map<std::string, std::shared_ptr<BaseLayer>> &
    get_layer_map() const;
    bool prepare_tensors();
    bool prepare_graph();
    void enable_profile(bool do_profile) { do_profile_ = do_profile; }
    const std::map<std::string, ProfileInfo> &get_profile() const { return layer_profile_info; }
    bool forward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> &output);
    bool serialize(std::string file_path);
    bool deserialize(std::string file_path);
    uint64_t calc_computation_flops() { return net_flops_; }
    uint64_t calc_memory_bytes() { return net_mem_bytes_; }
    friend Graph;

  private:
    std::map<std::string, std::shared_ptr<BaseLayer>> layers_;
    std::map<std::string, std::shared_ptr<Tensor>> tensors_;
    std::shared_ptr<Graph> graph_;
    std::string input_name_;
    std::shared_ptr<Serializer> serializer_;
    std::shared_ptr<Deserializer> deserializer_;
    uint64_t net_flops_ = 0, net_mem_bytes_ = 0;
    bool do_profile_ = false;
    std::map<std::string, ProfileInfo> layer_profile_info;
  };

} // namespace ti
