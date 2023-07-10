#pragma once
#include <string>
#include <vector>
#include <optional>

#include "tinyinfer/common/tensor.h"
#include "tinyinfer/layer/base_layer.h"
#include "third_party/libnpy/include/npy.hpp"

namespace ti {

class NumpyTensor {
 public:
    static std::optional<Tensor> FromFile(std::string_view file_path) {
        std::vector<unsigned long> shape;
        bool fortran_order;
        std::vector<float> data;
        ::npy::LoadArrayFromNumpy<float>(std::string(file_path), shape, fortran_order, data);
        int n = 0, c = 0, h = 0, w = 0;
        if (shape.size() >= 0) {
            n = shape[0];
        }
        if (shape.size() >= 1) {
            c = shape[1];
        }
        if (shape.size() >= 2) {
            h = shape[2];
        }
        if (shape.size() >= 3) {
            w = shape[3];
        }
        Tensor ret(n, c, h, w, std::move(data));
        return std::make_optional<Tensor>(ret);
    }
};

}