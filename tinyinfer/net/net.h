#pragma once

#include <map>
#include <memory>
#include "tinyinfer/common/tensor.h"
#include "tinyinfer/layer/base_layer.h"

namespace ti {

class Net;
class Graph {
 public:
    struct {
        std::vector<std::string> prev;
        std::string name;
        std::vector<std::string> next;
    } Node;
    Graph() {}
    static Graph FromNet(Net& net) {
        return Graph();
    }
    void restart() {}
    bool is_finished() { return true; }
    std::string next() { return ""; }
 private:
};

class Net {
 public:
    bool register_layer(std::string name, std::shared_ptr<BaseLayer> layer) {
        layers_[name] = layer;
        return true;
    }
    bool register_tensor() {
        return true;
    }
    bool Forward(const Tensor& input) {
        graph_.restart();
        while (graph_.is_finished()) {
            std::string layer_name = graph_.next();
            auto& layer = layers_[layer_name];
            const std::vector<std::string> &input_names = layer->get_input_names();
            std::vector<std::shared_ptr<Tensor>> input_tensors;
            for (const auto& name : input_names) {
                input_tensors.push_back(tensors_[name]);
            }
            const std::vector<std::string> &output_names = layer->get_output_names();
            std::vector<std::shared_ptr<Tensor>> output_tensors;
            for (const auto& name : output_names) {
                output_tensors.push_back(tensors_[name]);
            }
            bool ret = layer.Forward(input_tensors, output_tensors[0]);
            CHECK_BOOL_RET(ret, true, "Layer :" << layer_name << " forward failed\n");
        }
        return true;
    }
    friend Graph;
 private:
    std::map<std::string, std::shared_ptr<BaseLayer>> layers_;
    std::map<std::string, std::shared_ptr<Tensor>> tensors_;
    Graph graph_;
};

}