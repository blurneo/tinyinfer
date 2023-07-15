#pragma once
#include <string>
#include <vector>
#include <memory>
#include <optional>
#include <memory>

#include "tinyinfer/common/tensor.h"
#include "tinyinfer/common/base_layer.h"
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
        n = shape.size() == 4 ? shape[shape.size() - 4] : 0;
        c = shape.size() >= 3 ? shape[shape.size() - 3] : 0;
        h = shape.size() >= 2 ? shape[shape.size() - 2] : 0;
        w = shape.size() >= 1 ? shape[shape.size() - 1] : 0;
        return std::make_shared<Tensor>(n, c, h, w, std::move(data));
    }
};

}
