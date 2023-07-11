#pragma once

#include <vector>
#include "tinyinfer/common/tensor.h"
#include "tinyinfer/common/check_macro.h"
#include "tinyinfer/layer/base_layer.h"

namespace ti {

typedef struct MatmulLayerParameter : public BaseLayerParameter {

} MatmulLayerParameter;

class Matmul : public BaseLayer {
 public:
    Matmul(MatmulLayerParameter &&param) : param_(param), BaseLayer(LAYER_MATMUL) {}
 private:
    bool forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors, std::vector<std::shared_ptr<Tensor>> output_tensors) override {
        CHECK_BOOL_RET(input_tensors.size(), 2, "Matmul input tensor number should be 2")
        CHECK_BOOL_RET(input_tensors[0]->can_multiply(input_tensors[1]), true, "Two input tensors can't multiply.");
        std::shared_ptr<Tensor> output_tensor = output_tensors[0];
        output_tensor->reshape(0, 0, input_tensors[0]->get_h(), input_tensors[1]->get_w());
        return kernel(input_tensors[0], input_tensors[1], output_tensor);
    }
    bool kernel(std::shared_ptr<Tensor> input_tensor1, std::shared_ptr<Tensor>input_tensor2, std::shared_ptr<Tensor> output_tensor) {
        int H1 = input_tensor1->get_h();
        int W1 = input_tensor1->get_w();
        const float* val_ptr1 = input_tensor1->get_values().data();
        int H2 = input_tensor2->get_h();
        int W2 = input_tensor2->get_w();
        const float* val_ptr2 = input_tensor2->get_values().data();
        float* out_ptr = output_tensor->get_values().data();
        for (int h1 = 0; h1 < H1; h1++) {
            for (int w2 = 0; w2 < W2; w2++) {
                float sum = 0.f;
                for (int w1 = 0, h2 = 0; w1 < W1; w1++, h2++) {
                    sum += val_ptr1[h1 * W1 + w1] * val_ptr2[h2 * W2 + w2];
                }
                out_ptr[h1 * W2 + w2] = sum;
            }
        }
        return true;
    }

 private:
    MatmulLayerParameter param_;
};

}
