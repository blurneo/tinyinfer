#pragma once

#include <vector>
#include "tinyinfer/common/tensor.h"
#include "tinyinfer/common/check_macro.h"
#include "tinyinfer/layer/base_layer.h"

namespace ti {

typedef struct ReshapeLayerParameter : public BaseLayerParameter {
    std::vector<int> shapes;
} ReshapeLayerParameter;

class Reshape : public BaseLayer {
 public:
    Reshape(ReshapeLayerParameter &&param) : param_(param), BaseLayer(LAYER_RESHAPE) {}
    bool Forward(const std::vector<Tensor> &input_tensors, Tensor &output_tensor) override {
        CHECK_BOOL_RET(input_tensors.size(), 1, "Maxpool input tensor number should be 1")
        const Tensor &input_tensor = input_tensors[0];
        CHECK_BOOL_RET(param_.shapes.size() > 0, true, "Reshape layer shape param is empty");
        CHECK_BOOL_RET(param_.shapes.size() <= 4, true, "Reshape layer shape param is too large");
        int count = 1;
        for (auto shape : param_.shapes) {
            count *= shape;
        }
        CHECK_BOOL_RET(count == input_tensor.get_count(), true, "Reshape layer input count not same with param");
        return kernel(input_tensor, output_tensor);
    }
 private:
    bool kernel(const Tensor &input_tensor, Tensor &output_tensor) {
        int shape_size = param_.shapes.size();
        switch (shape_size) {
            case 1:
                output_tensor.reshape(0, 0, 0, param_.shapes[0]);
                break;
            case 2:
                output_tensor.reshape(0, 0, param_.shapes[0], param_.shapes[1]);
                break;
            case 3:
                output_tensor.reshape(0, param_.shapes[0], param_.shapes[1], param_.shapes[2]);
                break;
            case 4:
                output_tensor.reshape(param_.shapes[0], param_.shapes[1], param_.shapes[2], param_.shapes[3]);
                break;
            default:
                break;
        }
        return true;
    }
    ReshapeLayerParameter param_;
};

}
