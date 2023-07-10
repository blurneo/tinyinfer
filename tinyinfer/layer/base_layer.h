#pragma once

#include <vector>
#include "tinyinfer/common/tensor.h"

namespace ti {

typedef enum LayerType {
    LAYER_ADD                 = 0,
    LAYER_CONVOLUTION         = 1,
    LAYER_LRN                 = 2,
    LAYER_MATMUL              = 3,
    LAYER_MAXPOOL             = 4,
    LAYER_RELU                = 5,
    LAYER_RESHAPE             = 6
} LayerType;

typedef struct BaseLayerParameter {

} BaseLayerParameter;

class BaseLayer {
 public:
    BaseLayer(LayerType layer_type) : layer_type_(layer_type) {}
    virtual bool Forward(const std::vector<Tensor> &input_tensors, Tensor &output_tensor) = 0;
    virtual void set_layer_name(std::string name) {
        layer_name_ = name;
    }
    virtual void set_input_names(std::vector<std::string> names) {
        input_names_ = names;
    }
    virtual void set_output_names(std::vector<std::string> names) {
        output_names_ = names;
    }
    virtual std::vector<std::string> get_input_names() const {
        return input_names_;
    }
    virtual std::vector<std::string> get_output_names() const {
        return output_names_;
    }
    virtual bool is_input(const Tensor &tensor) const {
        for (auto name : input_names_) {
            if (tensor.get_name() == name) {
                return true;
            }
        }
        return false;
    }
    virtual bool is_output(const Tensor &tensor) const {
        for (auto name : output_names_) {
            if (tensor.get_name() == name) {
                return true;
            }
        }
        return false;
    }
 protected:
    std::string layer_name_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    LayerType layer_type_;
};

}