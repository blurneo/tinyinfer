#include "tinyinfer/common/check_macro.h"
#include "tools/numpy_tensor.h"

int main() {
    std::string_view file_path = "models/mnist/Parameter5.npy";
    std::optional<ti::Tensor> npy_tensor = ti::NumpyTensor::FromFile(file_path);
    CHECK_INT_RET(npy_tensor->dims(), 4, -1, "Read numpy tensor dims wrong: " << npy_tensor->dims());
    CHECK_INT_RET(npy_tensor->get_n(), 8, -1, "Read numpy tensor shape n wrong" << npy_tensor->get_n());
    CHECK_INT_RET(npy_tensor->get_c(), 1, -1, "Read numpy tensor shape c wrong" << npy_tensor->get_c());
    CHECK_INT_RET(npy_tensor->get_h(), 5, -1, "Read numpy tensor shape h wrong" << npy_tensor->get_h());
    CHECK_INT_RET(npy_tensor->get_w(), 5, -1, "Read numpy tensor shape w wrong" << npy_tensor->get_w());
    CHECK_INT_RET(npy_tensor->get_count(), 200, -1, "Read numpy tensor count wrong" << npy_tensor->get_count());
    CHECK_INT_RET(npy_tensor->get_values().size(), 200, -1, "Read numpy tensor value size wrong" << npy_tensor->get_values().size());
    return 0;
}