#pragma once

#include "tinyinfer/common/check_macro.h"
#include "tinyinfer/common/tensor.h"
#include "tinyinfer/layer/base_layer.h"
#include <cmath>
#include <iostream>
#include <vector>

namespace ti {

typedef struct ConvolutionLayerParam : public BaseLayerParameter {
  int kernel_shape_x = 0;
  int kernel_shape_y = 0;
  int stride_x = 1;
  int stride_y = 1;
  int pad_t = 0; // first of 4 pads values in onnx
  int pad_d = 0; // second of 4 pads values in onnx
  int pad_l = 0; // third of 4 pads values in onnx
  int pad_r = 0; // fourth of 4 pads values in onnx
  int group = 1;
  int dilation_x = 1;
  int dilation_y = 1;
  int pad_type = 0; // 0: NotSet, 1: SameUpper, 2: SameLower, 3: Valid;
  std::shared_ptr<Tensor> weights;
  std::shared_ptr<Tensor> bias;
} ConvolutionLayerParameter;

class Convolution : public BaseLayer {
public:
  Convolution() : BaseLayer(LAYER_CONVOLUTION) {}
  Convolution(ConvolutionLayerParameter &&param)
      : param_(std::move(param)), BaseLayer(LAYER_CONVOLUTION) {}
  void get_pad(int input_h, int input_w, int s_h, int s_w, int pad_type,
               int kernel_shape_y, int kernel_shape_x, int &pad_t, int &pad_d,
               int &pad_l, int &pad_r);
  bool forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
               std::vector<std::shared_ptr<Tensor>> output_tensors) override;

private:
  bool kernel(const std::shared_ptr<Tensor> &input_tensor,
              std::shared_ptr<Tensor> output_tensor);

private:
  ConvolutionLayerParameter param_;
};

} // namespace ti
