#include "tinyinfer/net/net.h"
#include "tinyinfer/net/graph.h"

namespace ti
{

  bool Net::register_layer(std::string name, std::shared_ptr<BaseLayer> layer)
  {
    layers_[name] = layer;
    return true;
  }
  void Net::set_input_name(std::string name) { input_name_ = name; }
  std::string Net::get_input_name() const { return input_name_; }
  const std::map<std::string, std::shared_ptr<BaseLayer>> &
  Net::get_layer_map() const
  {
    return layers_;
  }
  bool Net::prepare_tensors()
  {
    for (auto _layer : layers_)
    {
      auto layer = _layer.second;
      std::vector<std::string> output_names = layer->get_output_names();
      for (auto output_name : output_names)
      {
        std::shared_ptr<Tensor> tensor;
        tensor.reset(new Tensor());
        tensor->set_name(output_name);
        tensors_[output_name] = tensor;
      }
    }
    return true;
  }
  bool Net::prepare_graph()
  {
    graph_ = Graph::FromNet(this);
    return true;
  }
  bool Net::forward(std::shared_ptr<Tensor> input,
                    std::shared_ptr<Tensor> &output)
  {
    graph_->restart();
    tensors_[input->get_name()] = input;
    std::vector<std::shared_ptr<Tensor>> input_tensors;
    std::vector<std::shared_ptr<Tensor>> output_tensors;
    while (!graph_->is_finished())
    {
      input_tensors.clear();
      output_tensors.clear();
      auto layer = graph_->next();
      const std::vector<std::string> &input_names = layer->get_input_names();
      for (const auto &name : input_names)
      {
        input_tensors.push_back(tensors_[name]);
      }
      const std::vector<std::string> &output_names = layer->get_output_names();
      for (const auto &name : output_names)
      {
        output_tensors.push_back(tensors_[name]);
      }
      bool ret;
      if (do_profile_)
        ret = layer->forward(input_tensors, output_tensors, layer_profile_info[layer->get_layer_name()]);
      else
        ret = layer->forward(input_tensors, output_tensors);
      CHECK_BOOL_RET(ret, true,
                     "Layer :" << layer->get_layer_name() << " forward failed\n");
      // calculation information
      net_flops_ += layer->calc_computation_flops();
      net_mem_bytes_ += layer->calc_memory_bytes();
    }
    output = output_tensors[0];
    return true;
  }

  bool Net::serialize(std::string file_path)
  {
    if (!serializer_)
      serializer_.reset(new Serializer());
    graph_->restart();
    CHECK_BOOL_RET(serializer_->start(file_path, graph_->layer_count()), true, "Graph serializer open failed");
    while (!graph_->is_finished())
    {
      auto layer = graph_->next();
      serializer_->serialize_one_layer(layer);
    }
    serializer_->finish();

    return true;
  }

  bool Net::deserialize(std::string file_path)
  {
    if (!deserializer_)
      deserializer_.reset(new Deserializer());
    bool ret = deserializer_->start(file_path);
    CHECK_BOOL_RET(ret, true, "Deserializer start failed\n")
    while (!deserializer_->is_finished())
    {
      std::shared_ptr<BaseLayer> layer = deserializer_->deserialize_one_layer();
      CHECK_RET(layer != nullptr, true, false, "Deserializer deserialize failed\n")
      layers_[layer->get_layer_name()] = layer;
    }
    ret = deserializer_->finish();
    CHECK_BOOL_RET(ret, true, "Deserializer finish failed\n")
    prepare_graph();
    prepare_tensors();
    return true;
  }

} // namespace ti
