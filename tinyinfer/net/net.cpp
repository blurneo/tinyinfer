#include "tinyinfer/net/net.h"
#include "tinyinfer/net/graph.h"

namespace ti {

bool Net::register_layer(std::string name, std::shared_ptr<BaseLayer> layer) {
  layers_[name] = layer;
  return true;
}
void Net::set_input_name(std::string name) { input_name_ = name; }
std::string Net::get_input_name() const { return input_name_; }
const std::map<std::string, std::shared_ptr<BaseLayer>> &
Net::get_layer_map() const {
  return layers_;
}
bool Net::prepare_tensors() {
  for (auto _layer : layers_) {
    auto layer = _layer.second;
    std::vector<std::string> output_names = layer->get_output_names();
    for (auto output_name : output_names) {
      std::shared_ptr<Tensor> tensor;
      tensor.reset(new Tensor());
      tensor->set_name(output_name);
      tensors_[output_name] = tensor;
    }
  }
  return true;
}
bool Net::prepare_graph() {
  graph_ = Graph::FromNet(this);
  return true;
}
bool Net::forward(std::shared_ptr<Tensor> input,
                  std::shared_ptr<Tensor> &output) {
  graph_->restart();
  tensors_[input->get_name()] = input;
  std::vector<std::shared_ptr<Tensor>> input_tensors;
  std::vector<std::shared_ptr<Tensor>> output_tensors;
  while (!graph_->is_finished()) {
    input_tensors.clear();
    output_tensors.clear();
    auto layer = graph_->next();
    const std::vector<std::string> &input_names = layer->get_input_names();
    for (const auto &name : input_names) {
      input_tensors.push_back(tensors_[name]);
    }
    const std::vector<std::string> &output_names = layer->get_output_names();
    for (const auto &name : output_names) {
      output_tensors.push_back(tensors_[name]);
    }
    bool ret = layer->forward(input_tensors, output_tensors);
    CHECK_BOOL_RET(ret, true,
                   "Layer :" << layer->get_layer_name() << " forward failed\n");
  }
  output = output_tensors[0];
  return true;
}

} // namespace ti
