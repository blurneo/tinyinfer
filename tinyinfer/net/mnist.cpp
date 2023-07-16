#include "tinyinfer/common/tensor.h"
#include "tinyinfer/layer/add.h"
#include "tinyinfer/layer/clip.h"
#include "tinyinfer/layer/convolution.h"
#include "tinyinfer/layer/flatten.h"
#include "tinyinfer/layer/gemm.h"
#include "tinyinfer/layer/global_average_pool.h"
#include "tinyinfer/layer/lrn.h"
#include "tinyinfer/layer/matmul.h"
#include "tinyinfer/layer/max_pool.h"
#include "tinyinfer/layer/relu.h"
#include "tinyinfer/layer/reshape.h"
#include "tinyinfer/layer/softmax.h"
#include "tinyinfer/net/graph.h"
#include "tinyinfer/net/net.h"
#include "tools/numpy_tensor.h"
#include <iostream>

const std::string project_root_dir = "/Users/ssc/Desktop/TinyInfer/";
// const std::string project_root_dir =
// "/Users/ssc/Desktop/workspace/git_repos/tinyinfer/";

bool init_conv_weight(std::string layer_name, std::string weight_file,
                      std::string bias_file,
                      ti::ConvolutionLayerParameter &param) {
  std::shared_ptr<ti::Tensor> weight =
      ti::NumpyTensor<float>::FromFile(weight_file);
  param.weights = weight;
  if (!bias_file.empty()) {
    std::shared_ptr<ti::Tensor> bias =
        ti::NumpyTensor<float>::FromFile(bias_file);
    param.bias = bias;
  } else {
    param.bias.reset(new ti::Tensor());
  }
  return true;
}

bool init_add_weight(std::string layer_name, std::string weight_file,
                     ti::AddLayerParameter &param) {
  std::shared_ptr<ti::Tensor> weight =
      ti::NumpyTensor<float>::FromFile(weight_file);
  param.weights = weight;
  return true;
}

bool init_reshape_weight(std::string layer_name, std::string data_file,
                         std::string shape_file,
                         ti::ReshapeLayerParameter &param) {
  if (!data_file.empty()) {
    std::shared_ptr<ti::Tensor> data =
        ti::NumpyTensor<float>::FromFile(data_file);
    param.data = data;
  } else {
    param.data.reset(new ti::Tensor());
  }
  if (!shape_file.empty()) {
    std::vector<unsigned long> shape;
    bool fortran_order;
    std::vector<int64_t> data;
    ::npy::LoadArrayFromNumpy<int64_t>(std::string(shape_file), shape,
                                       fortran_order, data);
    for (auto d : data) {
      param.shape.push_back(d);
    }
  }
  return true;
}

void register_1(ti::Net &mnist_net) {
  std::string layer_name = "Convolution28";
  std::string weight_file = project_root_dir + "/models/mnist/Parameter5.npy";
  std::shared_ptr<ti::Convolution> layer;
  ti::ConvolutionLayerParameter param = {
      .kernel_shape_x = 5,
      .kernel_shape_y = 5,
      .stride_x = 1,
      .stride_y = 1,
      .pad_t = 0,
      .pad_d = 0,
      .pad_l = 0,
      .pad_r = 0,
      .group = 1,
      .dilation_x = 1,
      .dilation_y = 1,
      .pad_type = 1,
  };
  init_conv_weight(layer_name, weight_file, "", param);
  layer.reset(new ti::Convolution(std::move(param)));
  layer->set_layer_name(layer_name);
  layer->set_input_names({"Input3"});
  layer->set_output_names({"Convolution28_Output_0"});
  mnist_net.register_layer(layer_name, layer);
}

void register_2(ti::Net &mnist_net) {
  std::string layer_name = "Plus30";
  std::string weight_file = project_root_dir + "/models/mnist/Parameter6.npy";
  std::shared_ptr<ti::Add> layer;
  ti::AddLayerParameter param;
  init_add_weight(layer_name, weight_file, param);
  layer.reset(new ti::Add(std::move(param)));
  layer->set_layer_name(layer_name);
  layer->set_input_names({"Convolution28_Output_0"});
  layer->set_output_names({"Plus30_Output_0"});
  mnist_net.register_layer(layer_name, layer);
}

void register_3(ti::Net &mnist_net) {
  std::string layer_name = "ReLU32";
  std::shared_ptr<ti::Relu> layer;
  ti::ReluLayerParameter param;
  layer.reset(new ti::Relu(std::move(param)));
  layer->set_layer_name(layer_name);
  layer->set_input_names({"Plus30_Output_0"});
  layer->set_output_names({"ReLU32_Output_0"});
  mnist_net.register_layer(layer_name, layer);
}

void register_4(ti::Net &mnist_net) {
  std::string layer_name = "Pooling66";
  std::shared_ptr<ti::MaxPool> layer;
  ti::MaxPoolLayerParameter param = {
      .kernel_shape_x = 2,
      .kernel_shape_y = 2,
      .stride_x = 2,
      .stride_y = 2,
      .pad_l = 0,
      .pad_r = 0,
      .pad_t = 0,
      .pad_d = 0,
  };
  layer.reset(new ti::MaxPool(std::move(param)));
  layer->set_layer_name(layer_name);
  layer->set_input_names({"ReLU32_Output_0"});
  layer->set_output_names({"Pooling66_Output_0"});
  mnist_net.register_layer(layer_name, layer);
}

void register_5(ti::Net &mnist_net) {
  std::string layer_name = "Convolution110";
  std::string weight_file = project_root_dir + "/models/mnist/Parameter87.npy";
  std::shared_ptr<ti::Convolution> layer;
  ti::ConvolutionLayerParameter param = {
      .kernel_shape_x = 5,
      .kernel_shape_y = 5,
      .stride_x = 1,
      .stride_y = 1,
      .pad_t = 0,
      .pad_d = 0,
      .pad_l = 0,
      .pad_r = 0,
      .group = 1,
      .dilation_x = 1,
      .dilation_y = 1,
      .pad_type = 1,
  };
  init_conv_weight(layer_name, weight_file, "", param);
  layer.reset(new ti::Convolution(std::move(param)));
  layer->set_layer_name(layer_name);
  layer->set_input_names({"Pooling66_Output_0"});
  layer->set_output_names({"Convolution110_Output_0"});
  mnist_net.register_layer(layer_name, layer);
}

void register_6(ti::Net &mnist_net) {
  std::string layer_name = "Plus112";
  std::string weight_file = project_root_dir + "/models/mnist/Parameter88.npy";
  std::shared_ptr<ti::Add> layer;
  ti::AddLayerParameter param;
  init_add_weight(layer_name, weight_file, param);
  layer.reset(new ti::Add(std::move(param)));
  layer->set_layer_name(layer_name);
  layer->set_input_names({"Convolution110_Output_0"});
  layer->set_output_names({"Plus112_Output_0"});
  mnist_net.register_layer(layer_name, layer);
}

void register_7(ti::Net &mnist_net) {
  std::string layer_name = "ReLU114";
  std::shared_ptr<ti::Relu> layer;
  ti::ReluLayerParameter param;
  layer.reset(new ti::Relu(std::move(param)));
  layer->set_layer_name(layer_name);
  layer->set_input_names({"Plus112_Output_0"});
  layer->set_output_names({"ReLU114_Output_0"});
  mnist_net.register_layer(layer_name, layer);
}

void register_8(ti::Net &mnist_net) {
  std::string layer_name = "Pooling160";
  std::shared_ptr<ti::MaxPool> layer;
  ti::MaxPoolLayerParameter param = {
      .kernel_shape_x = 3,
      .kernel_shape_y = 3,
      .stride_x = 3,
      .stride_y = 3,
      .pad_l = 0,
      .pad_r = 0,
      .pad_t = 0,
      .pad_d = 0,
  };
  layer.reset(new ti::MaxPool(std::move(param)));
  layer->set_layer_name(layer_name);
  layer->set_input_names({"ReLU114_Output_0"});
  layer->set_output_names({"Pooling160_Output_0"});
  mnist_net.register_layer(layer_name, layer);
}

void register_9(ti::Net &mnist_net) {
  std::string layer_name = "Times212_reshape0";
  std::string shape_file =
      project_root_dir + "/models/mnist/Pooling160_Output_0_reshape0_shape.npy";
  std::shared_ptr<ti::Reshape> layer;
  ti::ReshapeLayerParameter param;
  init_reshape_weight(layer_name, "", shape_file, param);
  layer.reset(new ti::Reshape(std::move(param)));
  layer->set_layer_name(layer_name);
  layer->set_input_names({"Pooling160_Output_0"});
  layer->set_output_names({"Pooling160_Output_0_reshape0"});
  mnist_net.register_layer(layer_name, layer);
}

void register_10(ti::Net &mnist_net) {
  std::string layer_name = "Times212_reshape1";
  std::string data_file = project_root_dir + "/models/mnist/Parameter193.npy";
  std::string shape_file =
      project_root_dir + "/models/mnist/Parameter193_reshape1_shape.npy";
  std::shared_ptr<ti::Reshape> layer;
  ti::ReshapeLayerParameter param;
  init_reshape_weight(layer_name, data_file, shape_file, param);
  layer.reset(new ti::Reshape(std::move(param)));
  layer->set_layer_name(layer_name);
  // layer->set_input_names({""});
  layer->set_output_names({"Parameter193_reshape1"});
  mnist_net.register_layer(layer_name, layer);
}

void register_11(ti::Net &mnist_net) {
  std::string layer_name = "Times212";
  std::shared_ptr<ti::Matmul> layer;
  ti::MatmulLayerParameter param;
  layer.reset(new ti::Matmul(std::move(param)));
  layer->set_layer_name(layer_name);
  layer->set_input_names(
      {"Pooling160_Output_0_reshape0", "Parameter193_reshape1"});
  layer->set_output_names({"Times212_Output_0"});
  mnist_net.register_layer(layer_name, layer);
}

void register_12(ti::Net &mnist_net) {
  std::string layer_name = "Plus214";
  std::string weight_file = project_root_dir + "/models/mnist/Parameter194.npy";
  std::shared_ptr<ti::Add> layer;
  ti::AddLayerParameter param;
  init_add_weight(layer_name, weight_file, param);
  layer.reset(new ti::Add(std::move(param)));
  layer->set_layer_name(layer_name);
  layer->set_input_names({"Times212_Output_0"});
  layer->set_output_names({"Plus214_Output_0"});
  mnist_net.register_layer(layer_name, layer);
}

#define IS_SERIALIZE 1

int main() {
  ti::Net mnist_net;
#if IS_SERIALIZE
  register_1(mnist_net);
  register_2(mnist_net);
  register_3(mnist_net);
  register_4(mnist_net);
  register_5(mnist_net);
  register_6(mnist_net);
  register_7(mnist_net);
  register_8(mnist_net);
  register_9(mnist_net);
  register_10(mnist_net);
  register_11(mnist_net);
  register_12(mnist_net);
#endif
  std::shared_ptr<ti::Tensor> net_input = ti::NumpyTensor<float>::FromFile(
      project_root_dir + "models/mnist/input_0.npy");
  net_input->set_name("Input3");
  mnist_net.set_input_name("Input3");

#if IS_SERIALIZE
  mnist_net.prepare_graph();
  mnist_net.prepare_tensors();
  mnist_net.serialize(project_root_dir + "mnist-8.ti");
#else
  mnist_net.deserialize(project_root_dir + "mnist-8.ti");
#endif
  std::shared_ptr<ti::Tensor> net_output;
  int warup_cnt = 3, count = 10;
  for (int i = 0; i < warup_cnt; i++) {
    bool ret = mnist_net.forward(net_input, net_output);
    std::cout << "Mnist forward warm up :" << i << ", return : " << ret << "\n";
  }
  __TIC__(MnistForward)
  for (int i = 0; i < count; i++) {
    bool ret = mnist_net.forward(net_input, net_output);
    std::cout << "Mnist forward: " << i << ", return : " << ret << "\n";
  }
  __TOC__(MnistForward)
  std::cout << "MnistForward average time measured:"
            << __TIME_IN_MS__(MnistForward) / count << " ms\n";
  std::cout << "Net output: ";
  for (auto val : net_output->get_values()) {
    std::cout << val << ", ";
  }
  std::cout << "\n";
  std::cout << "Mnist calculation information:\n";
  std::cout << "Flops in MFlops: " << mnist_net.calc_computation_flops() / 1024.f / 1024.f << "\n"
            << "Memory access in MB: " << mnist_net.calc_memory_bytes() / 1024.f / 1024.f << "\n";
  return 0;
}
