#pragma once

#include "tinyinfer/common/check_macro.h"
#include "tinyinfer/common/tensor.h"
#include "tinyinfer/common/base_layer.h"
#include <cmath>
#include <iostream>
#include <vector>
#include "tinyinfer/reflection/serialize_macro.h"

namespace ti
{

  typedef struct ConvolutionLayerParam : public BaseLayerParameter
  {
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
    DEFINE_SERIALIZE_MEMBER(
        (kernel_shape_x)(kernel_shape_y)(stride_x)(stride_y)(pad_t)(pad_d)(pad_l)(pad_r)(group)(dilation_x)(dilation_y)(pad_type)(weights)(bias))
  } ConvolutionLayerParameter;

  class Serializer;
  class Deserializer;
  class Convolution : public BaseLayer
  {
  public:
    Convolution() : BaseLayer(LAYER_CONVOLUTION) {}
    Convolution(ConvolutionLayerParameter &&param)
        : param_(std::move(param)), BaseLayer(LAYER_CONVOLUTION) {}
    void get_pad(int input_h, int input_w, int s_h, int s_w, int pad_type,
                 int kernel_shape_y, int kernel_shape_x, int &pad_t, int &pad_d,
                 int &pad_l, int &pad_r);
    bool forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                 std::vector<std::shared_ptr<Tensor>> output_tensors) override;
    virtual void serialize(Serializer &serializer);
    virtual bool deserialize(Deserializer &deserializer);

  private:
    bool kernel(const std::shared_ptr<Tensor> &input_tensor,
                std::shared_ptr<Tensor> output_tensor);
    bool kernel_gemm(const std::shared_ptr<Tensor> &input_tensor,
                     std::shared_ptr<Tensor> output_tensor);

  private:
    ConvolutionLayerParameter param_;
    DEFINE_SERIALIZE_MEMBER(
        (&param_))
  };

} // namespace ti
