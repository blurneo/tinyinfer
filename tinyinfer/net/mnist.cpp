#include <iostream>
#include "tinyinfer/common/tensor.h"
#include "tinyinfer/layer/convolution.h"
#include "tinyinfer/layer/add.h"
#include "tinyinfer/layer/max_pool.h"
#include "tinyinfer/layer/relu.h"
#include "tinyinfer/layer/lrn.h"
#include "tinyinfer/layer/reshape.h"
#include "tinyinfer/layer/matmul.h"
#include "tinyinfer/net/graph.h"
#include "tinyinfer/net/net.h"
#include "tools/numpy_tensor.h"

bool init_conv_weight(std::string layer_name, std::string weight_file, std::string bias_file,
               ti::ConvolutionLayerParameter &param) {
    std::shared_ptr<ti::Tensor> weight = ti::NumpyTensor<float>::FromFile(weight_file);
    param.weights = weight;
    if (!bias_file.empty()) {
        std::shared_ptr<ti::Tensor> bias = ti::NumpyTensor<float>::FromFile(bias_file);
        param.bias = bias;
    }
    return true;
}

bool init_add_weight(std::string layer_name, std::string weight_file, ti::AddLayerParameter &param) {
    std::shared_ptr<ti::Tensor> weight = ti::NumpyTensor<float>::FromFile(weight_file);
    param.weights = weight;
    return true;
}

bool init_reshape_weight(std::string layer_name, std::string data_file, std::string shape_file, ti::ReshapeLayerParameter &param) {
    if (!data_file.empty()) {
        std::shared_ptr<ti::Tensor> data = ti::NumpyTensor<float>::FromFile(data_file);
        param.data = data;
    }
    if (!shape_file.empty()) {
        std::vector<unsigned long> shape;
        bool fortran_order;
        std::vector<unsigned long> data;
        ::npy::LoadArrayFromNumpy<unsigned long>(std::string(shape_file), shape, fortran_order, data);
        param.shape = data;
    }
    return true;
}

void register_1(ti::Net& mnist_net) {
    std::string layer_name = "Convolution28";
    std::string weight_file = "/Users/ssc/Desktop/workspace/git_repos/tinyinfer/models/mnist/Parameter5.npy";
    std::shared_ptr<ti::Convolution> layer;
    ti::ConvolutionLayerParameter param = {
        .kernel_shape_x = 5,
        .kernel_shape_y = 5,
        .stride_x = 1,
        .stride_y = 1,
        .pad_l = 0,
        .pad_r = 0,
        .pad_t = 0,
        .pad_d = 0,
        .group = 1,
        .dilation_x = 1,
        .dilation_y = 1,
    };
    init_conv_weight(layer_name, weight_file, "", param);
    layer.reset(new ti::Convolution(std::move(param)));
    layer->set_input_names({"Input3"});
    layer->set_output_names({"Convolution28_Output_0"});
    mnist_net.register_layer(layer_name, layer);
}

void register_2(ti::Net& mnist_net) {
    std::string layer_name = "Plus30";
    std::string weight_file = "/Users/ssc/Desktop/workspace/git_repos/tinyinfer/models/mnist/Parameter6.npy";
    std::shared_ptr<ti::Add> layer;
    ti::AddLayerParameter param;
    init_add_weight(layer_name, weight_file, param);
    layer.reset(new ti::Add(std::move(param)));
    layer->set_input_names({"Convolution28_Output_0"});
    layer->set_output_names({"Plus30_Output_0"});
    mnist_net.register_layer(layer_name, layer);
}

void register_3(ti::Net& mnist_net) {
    std::string layer_name = "ReLU32";
    std::shared_ptr<ti::Relu> layer;
    ti::ReluLayerParameter param;
    layer.reset(new ti::Relu(std::move(param)));
    layer->set_input_names({"Plus30_Output_0"});
    layer->set_output_names({"ReLU32_Output_0"});
    mnist_net.register_layer(layer_name, layer);
}

void register_4(ti::Net& mnist_net) {
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
    layer->set_input_names({"ReLU32_Output_0"});
    layer->set_output_names({"Pooling66_Output_0"});
    mnist_net.register_layer(layer_name, layer);
}

void register_5(ti::Net& mnist_net) {
    std::string layer_name = "Convolution110";
    std::string weight_file = "/Users/ssc/Desktop/workspace/git_repos/tinyinfer/models/mnist/Parameter87.npy";
    std::shared_ptr<ti::Convolution> layer;
    ti::ConvolutionLayerParameter param = {
        .kernel_shape_x = 5,
        .kernel_shape_y = 5,
        .stride_x = 1,
        .stride_y = 1,
        .pad_l = 0,
        .pad_r = 0,
        .pad_t = 0,
        .pad_d = 0,
        .group = 1,
        .dilation_x = 1,
        .dilation_y = 1,
    };
    init_conv_weight(layer_name, weight_file, "", param);
    layer.reset(new ti::Convolution(std::move(param)));
    layer->set_input_names({"Pooling66_Output_0"});
    layer->set_output_names({"Convolution110_Output_0"});
    mnist_net.register_layer(layer_name, layer);
}

void register_6(ti::Net& mnist_net) {
    std::string layer_name = "Plus112";
    std::string weight_file = "/Users/ssc/Desktop/workspace/git_repos/tinyinfer/models/mnist/Parameter88.npy";
    std::shared_ptr<ti::Add> layer;
    ti::AddLayerParameter param;
    init_add_weight(layer_name, weight_file, param);
    layer.reset(new ti::Add(std::move(param)));
    layer->set_input_names({"Convolution110_Output_0"});
    layer->set_output_names({"Plus112_Output_0"});
    mnist_net.register_layer(layer_name, layer);
}

void register_7(ti::Net& mnist_net) {
    std::string layer_name = "ReLU114";
    std::shared_ptr<ti::Relu> layer;
    ti::ReluLayerParameter param;
    layer.reset(new ti::Relu(std::move(param)));
    layer->set_input_names({"Plus112_Output_0"});
    layer->set_output_names({"ReLU114_Output_0"});
    mnist_net.register_layer(layer_name, layer);
}

void register_8(ti::Net& mnist_net) {
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
    layer->set_input_names({"ReLU114_Output_0"});
    layer->set_output_names({"Pooling160_Output_0"});
    mnist_net.register_layer(layer_name, layer);
}

void register_9(ti::Net& mnist_net) {
    std::string layer_name = "Times212_reshape0";
    std::string shape_file = "/Users/ssc/Desktop/workspace/git_repos/tinyinfer/models/mnist/Pooling160_Output_0_reshape0_shape.npy";
    std::shared_ptr<ti::Reshape> layer;
    ti::ReshapeLayerParameter param;
    init_reshape_weight(layer_name, "", shape_file, param);
    layer.reset(new ti::Reshape(std::move(param)));
    layer->set_input_names({"Pooling160_Output_0"});
    layer->set_output_names({"Pooling160_Output_0_reshape0"});
    mnist_net.register_layer(layer_name, layer);
}

void register_10(ti::Net& mnist_net) {
    std::string layer_name = "Times212_reshape1";
    std::string data_file = "/Users/ssc/Desktop/workspace/git_repos/tinyinfer/models/mnist/Parameter193.npy";
    std::string shape_file = "/Users/ssc/Desktop/workspace/git_repos/tinyinfer/models/mnist/Parameter193_reshape1_shape.npy";
    std::shared_ptr<ti::Reshape> layer;
    ti::ReshapeLayerParameter param;
    init_reshape_weight(layer_name, data_file, shape_file, param);
    layer.reset(new ti::Reshape(std::move(param)));
    // layer->set_input_names({""});
    layer->set_output_names({"Parameter193_reshape1"});
    mnist_net.register_layer(layer_name, layer);
}

void register_11(ti::Net& mnist_net) {
    std::string layer_name = "Times212";
    std::shared_ptr<ti::Matmul> layer;
    ti::MatmulLayerParameter param;
    layer.reset(new ti::Matmul(std::move(param)));
    layer->set_input_names({"Pooling160_Output_0_reshape0", "Parameter193_reshape1"});
    layer->set_output_names({"Parameter193_reshape1"});
    mnist_net.register_layer(layer_name, layer);
}

void register_12(ti::Net& mnist_net) {
    std::string layer_name = "Plus214";
    std::string weight_file = "/Users/ssc/Desktop/workspace/git_repos/tinyinfer/models/mnist/Parameter194.npy";
    std::shared_ptr<ti::Add> layer;
    ti::AddLayerParameter param;
    init_add_weight(layer_name, weight_file, param);
    layer.reset(new ti::Add(std::move(param)));
    layer->set_input_names({"Times212_Output_0"});
    layer->set_output_names({"Plus214_Output_0"});
    mnist_net.register_layer(layer_name, layer);
}

int main() {
    ti::Net mnist_net;
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

    return 0;
}
