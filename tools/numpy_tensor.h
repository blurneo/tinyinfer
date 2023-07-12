#pragma once
#include <string>
#include <vector>
#include <memory>
#include <optional>
#include <memory>

#include "tinyinfer/common/tensor.h"
#include "tinyinfer/layer/base_layer.h"
#include "third_party/libnpy/include/npy.hpp"

namespace ti {

template <typename dtype>
class NumpyTensor {
 public:
    static std::shared_ptr<Tensor> FromFile(std::string file_path) {
        std::vector<unsigned long> shape;
        bool fortran_order;
        std::vector<dtype> data;
        ::npy::LoadArrayFromNumpy<dtype>(std::string(file_path), shape, fortran_order, data);
        int n = 0, c = 0, h = 0, w = 0;
        if (shape.size() == 4) {
            n = shape[0];
            c = shape[1];
            h = shape[2];
            w = shape[3];
        }
        if (shape.size() == 3) {
            c = shape[0];
            h = shape[1];
            w = shape[2];
        }
        if (shape.size() == 2) {
            h = shape[0];
            w = shape[1];
        }
        if (shape.size() == 1) {
            w = shape[0];
        }
        return std::make_shared<Tensor>(n, c, h, w, std::move(data));
    }
};

}
