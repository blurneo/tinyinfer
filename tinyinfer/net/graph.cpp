#include "tinyinfer/net/graph.h"
#include "tinyinfer/net/net.h"

namespace ti {

std::shared_ptr<Graph> Graph::FromNet(const Net *net) {
  std::shared_ptr<Graph> graph(new Graph());
  auto layer_map = net->get_layer_map();
  auto net_input_name = net->get_input_name();
  // function to find the next layers of the input layer
  auto find_next_layers =
      [&](std::shared_ptr<BaseLayer> input_layer) -> std::vector<std::string> {
    std::vector<std::string> ret;
    auto output_names = input_layer->get_output_names();
    for (auto pair : layer_map) {
      if (pair.first == input_layer->get_layer_name())
        continue;
      for (auto output_name : output_names) {
        if (pair.second->is_input_name(output_name)) {
          ret.push_back(pair.first);
        }
      }
    }
    return ret;
  };
  // function to find the previous layers of the input layer
  auto find_pre_layers =
      [&](std::shared_ptr<BaseLayer> input_layer) -> std::vector<std::string> {
    std::vector<std::string> ret;
    auto input_names = input_layer->get_input_names();
    for (auto pair : layer_map) {
      if (pair.first == input_layer->get_layer_name())
        continue;
      for (auto input_name : input_names) {
        if (pair.second->is_output_name(input_name)) {
          ret.push_back(pair.first);
        }
      }
    }
    return ret;
  };
  // init the traverse map of the layers
  std::map<std::string, bool> traversed;
  for (auto pair : layer_map) {
    traversed[pair.first] = false;
  }
  // push layers that net input tensor forward to
  for (auto pair : layer_map) {
    auto input_names = pair.second->get_input_names();
    for (auto input_name : input_names) {
      if (input_name == net_input_name) {
        graph->nodes_.push_back(pair.second);
        traversed[pair.first] = true;
      }
    }
  }
  // push layers with no input tensor
  for (auto pair : layer_map) {
    if (pair.second->get_input_names().empty()) {
      graph->nodes_.push_back(pair.second);
      traversed[pair.first] = true;
    }
  }
  CHECK_RET(graph->nodes_.empty(), false, nullptr,
            "Graph can't find net input layer");
  // push layers with using Width-First-Search
  int start_idx = 0;
  while (start_idx < graph->nodes_.size()) {
    // get all next layer names connect to the current node
    std::vector<std::string> next_layer_names =
        find_next_layers(graph->nodes_[start_idx]);
    // check whether all the previous layers of each next layer has all been
    // traversed
    for (auto next_layer_name : next_layer_names) {
      bool ok = true;
      // find out all the previous layers of the layer
      std::vector<std::string> pre_layer_names =
          find_pre_layers(layer_map[next_layer_name]);
      // check whether all the previous layers have been traversed
      for (auto pre_layer_name : pre_layer_names) {
        if (traversed[pre_layer_name] == false) {
          ok = false;
          break;
        }
      }
      // mark the layer as traversed only when all the previous layers have been
      // traversed
      if (ok) {
        graph->nodes_.push_back(layer_map[next_layer_name]);
        traversed[next_layer_name] = true;
      }
    }
    start_idx++;
  };

  return graph;
}

} // namespace ti
