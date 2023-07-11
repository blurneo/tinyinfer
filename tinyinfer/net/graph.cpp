#include "tinyinfer/net/graph.h"
#include "tinyinfer/net/net.h"

namespace ti {

std::shared_ptr<Graph> Graph::FromNet(const Net* net) {
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
    // push layers with no input tensor
    for (auto pair : layer_map) {
        if (pair.second->get_input_names().empty()) {
            graph->nodes_.push_back(pair.second);
        }
    }
    // push layers that net input tensor forward to
    for (auto pair : layer_map) {
        auto input_names = pair.second->get_input_names();
        for (auto input_name : input_names) {
            if (input_name == net_input_name) {
                graph->nodes_.push_back(pair.second);
            }
        }
    }
    CHECK_RET(graph->nodes_.empty(), false, nullptr, "Graph can't find net input layer");
    // push layers with using Width-First-Search
    int start_idx = 0;
    while (start_idx < graph->nodes_.size()) {
        find_layers(graph->nodes_[start_idx]);
        start_idx++;
    };

    return graph;
}

}
