#pragma once

#include <map>
#include <memory>
#include <set>
#include "tinyinfer/common/tensor.h"
#include "tinyinfer/layer/base_layer.h"

namespace ti {

class Net;
class Graph {
 public:
    Graph() {}
    static std::shared_ptr<Graph> FromNet(const Net* net) {
        std::shared_ptr<Graph> graph(new Graph());
        auto layer_map = net->get_layer_map();
        auto net_input_name = net->get_input_name();
        auto find_layers = [&](std::shared_ptr<BaseLayer> input_layer) -> void {
            auto output_names = input_layer->get_output_names();
            for (auto pair : layer_map) {
                if (pair.first == input_layer->get_layer_name()) continue;
                auto layer_input_names = pair.second->get_input_names();
                for (auto output_name : output_names) {
                    for (auto input_name : layer_input_names) {
                        if (input_name == output_name)
                            graph->nodes_.push_back(pair.second);
                    }
                }
            }
        };

        for (auto pair : layer_map) {
            auto input_names = pair.second->get_input_names();
            for (auto input_name : input_names) {
                if (input_name == net_input_name) {
                    graph->nodes_.push_back(pair.second);
                }
            }
        }
        CHECK_RET(graph->nodes_.empty(), false, nullptr, "Graph can't find net input layer");

        int start_idx = 0;
        while (start_idx < graph->nodes_.size()) {
            find_layers(graph->nodes_[start_idx]);
            start_idx++;
        };

        return graph;
    }
    void restart() { current_ = 0;}
    bool is_finished() { return current_ >= nodes_.size(); }
    std::shared_ptr<BaseLayer> next() { return nodes_[current_]; }
 private:
    std::vector<std::shared_ptr<BaseLayer>> nodes_;
    int current_ = 0;
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
    void set_input_name(std::string name) { input_name_ = name; }
    std::string get_input_name() const { return input_name_; }
    const std::map<std::string, std::shared_ptr<BaseLayer>> &get_layer_map() const {
        return layers_;
    }
    bool prepare_graph() {
        graph_ = Graph::FromNet(this);
    }
    bool forward(std::shared_ptr<Tensor> input) {
        graph_->restart();
        tensors_[input->get_name()] = input;
        while (graph_->is_finished()) {
            auto layer = graph_->next();
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
            bool ret = layer->forward(input_tensors, output_tensors);
            CHECK_BOOL_RET(ret, true, "Layer :" << layer->get_layer_name() << " forward failed\n");
        }
        return true;
    }
    friend Graph;
 private:
    std::map<std::string, std::shared_ptr<BaseLayer>> layers_;
    std::map<std::string, std::shared_ptr<Tensor>> tensors_;
    std::shared_ptr<Graph> graph_;
    std::string input_name_;
};

}
