#pragma once

#include <vector>
#include "common/tensor.h"
#include "common/check_macro.h"

namespace ti {

typedef struct ReshapeLayerParam {
    std::vector<int> shapes;
} ReshapeLayerParam;

class Reshape {
 public:
    Reshape(ReshapeLayerParam &&param) : param_(param) {}
    bool Forward(const Tensor &input_tensor, Tensor &output_tensor) {
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
    ReshapeLayerParam param_;
};

}
