#include <vector>
#include "tinyinfer/common/tensor.h"
#include "tinyinfer/layer/convolution.h"
#include "tinyinfer/layer/max_pool.h"
#include "tinyinfer/layer/relu.h"
#include "tinyinfer/layer/lrn.h"

int main() {
    ti::Tensor weights(96, 3, 11, 11);
    ti::Tensor bias(96, 1, 1, 1);
    ti::ConvolutionLayerParam conv_param = {
        .kernel_shape_x = 11,
        .kernel_shape_y = 11,
        .stride_x = 4,
        .stride_y = 4,
        .pad_l = 0,
        .pad_r = 0,
        .pad_t = 0,
        .pad_d = 0,
        .group = 0,
        .weights = weights,
        .bias = bias,
    };
    ti::Tensor input_tensor(1,3,224,224), output_tensor;
    ti::Convolution conv_layer(std::move(conv_param));
    conv_layer.Forward(input_tensor, output_tensor);
    return 0;
}