#pragma once

#include <map>
#include <memory>
#include <set>
#include "tinyinfer/common/tensor.h"
#include "tinyinfer/layer/base_layer.h"

namespace ti {

class Graph;
class Net {
 public:
    bool register_layer(std::string name, std::shared_ptr<BaseLayer> layer);
    void set_input_name(std::string name);
    std::string get_input_name() const;
    const std::map<std::string, std::shared_ptr<BaseLayer>> &get_layer_map() const;
    bool prepare_tensors();
    bool prepare_graph();
    bool forward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> &output);
    friend Graph;
 private:
    std::map<std::string, std::shared_ptr<BaseLayer>> layers_;
    std::map<std::string, std::shared_ptr<Tensor>> tensors_;
    std::shared_ptr<Graph> graph_;
    std::string input_name_;
};

}
