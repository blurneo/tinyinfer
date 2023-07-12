#pragma once

#include <vector>
#include <optional>
#include <string>
#include <memory>
#include "common/check_macro.h"

namespace ti {

class Tensor {
 public:
    Tensor() : n_(0), c_(0), h_(0), w_(0) {}
    Tensor(int n, int c, int h, int w) :
        n_(n), c_(c), h_(h), w_(w), values_(std::vector<float>(n*c*h*w)) {
            dims_from_shapes(n, c, h, w);
        }
    Tensor(int n, int c, int h, int w, std::vector<float> &&values) :
        n_(n), c_(c), h_(h), w_(w), values_(std::move(values)) {
            dims_from_shapes(n, c, h, w);
        }
    void set_name(std::string name) { name_ = name; }
    std::string get_name() const { return name_; }
    int get_count() const { return values_.size(); }
    int get_n() const { return n_; }
    int get_c() const { return c_; }
    int get_h() const { return h_; }
    int get_w() const { return w_; }
    void set_n(int n) { n_ = n; }
    void set_c(int c) { c_ = c; }
    void set_h(int h) { h_ = h; }
    void set_w(int w) { w_ = w; }
    const std::vector<float> &get_values() const { return values_;}
    std::vector<float> &get_values() { return values_;}
    void reshape(std::vector<int> dims_vec) {
        int n = dims_vec.size() == 4 ? dims_vec[dims_vec.size() - 4] : 0;
        int c = dims_vec.size() >= 3 ? dims_vec[dims_vec.size() - 3] : 0;
        int h = dims_vec.size() >= 2 ? dims_vec[dims_vec.size() - 2] : 0;
        int w = dims_vec.size() >= 1 ? dims_vec[dims_vec.size() - 1] : 0;
        reshape(n, c, h, w);
    }
    void reshape(int n, int c, int h, int w) {
        set_n(n);
        set_c(c);
        set_h(h);
        set_w(w);
        int cnt = 0;
        if (n > 0) {
            cnt = cnt == 0 ? n : cnt * n;
        }
        if (c > 0) {
            cnt = cnt == 0 ? c : cnt * c;
        }
        if (h > 0) {
            cnt = cnt == 0 ? h : cnt * h;
        }
        if (w > 0) {
            cnt = cnt == 0 ? w : cnt * w;
        }
        values_.resize(cnt, 0);
        dims_from_shapes(n, c, h, w);
    }
    float* ptr() { return values_.data(); }
    const float* ptr() const { return values_.data(); }
    int dims() { return dims_; }
    int dims() const { return dims_; }
    std::vector<int> dims_vector() const {
        std::vector<int> ret;
        if (get_n() != 0) ret.push_back(get_n());
        if (get_c() != 0) ret.push_back(get_c());
        if (get_h() != 0) ret.push_back(get_h());
        if (get_w() != 0) ret.push_back(get_w());
        return ret;
    }
    std::optional<int> dim_stride(int dim_idx) const {
        auto dims_vec = dims_vector();
        if (dim_idx >= dims_vec.size()) {
            return std::nullopt;
        }
        int stride = 1;
        for (int idx = dim_idx + 1; idx < dims_vec.size(); idx++) {
            stride *= dims_vec[idx];
        }
        return std::make_optional<int>(stride);
    }

    // TODO: handle when tensor dimension is not 4
    static void pad(const std::shared_ptr<Tensor> &in, std::shared_ptr<Tensor> &out,
                    int pad_t, int pad_d, int pad_l, int pad_r) {
        int in_n = in->get_n();
        int in_c = in->get_c();
        int in_h = in->get_h();
        int in_w = in->get_w();
        int out_n = out->get_n();
        int out_c = out->get_c();
        int out_h = out->get_h() + pad_t + pad_d;
        int out_w = out->get_w() + pad_l + pad_r;
        out->reshape(out_n, out_c, out_h, out_w);
        int start_h = pad_t;
        int start_w = pad_l;
        for (int c = 0; c < in_c; c++) {
            for (int ih = 0, oh = start_h; ih < in_h; oh++, ih++) {
                const float *src = in->ptr() + c * in_h * in_w + ih * in_w;
                float *dst = out->ptr() + c * out_h * out_w + oh * out_w + start_w;
                std::memcpy(dst, src, in_w * sizeof(float));
            }
        }
    }

    bool is_alike(const std::shared_ptr<Tensor> &in) const {
        return (in->get_n() == get_n() &&
                in->get_c() == get_c() &&
                in->get_h() == get_h() &&
                in->get_w() == get_w() &&
                in->get_values().size() == get_values().size());
    }

    void reshape_like(const std::shared_ptr<Tensor> &in) {
        reshape(in->get_n(), in->get_c(), in->get_h(), in->get_w());
    }

    bool is_matrix() {
        return dims_ == 2;
    }
    bool can_multiply(const std::shared_ptr<Tensor> &in) const {
        if (dims() != 2 || in->dims() != 2) {
            return false;
        }
        return get_w() == in->get_h();
    }
    void copy_if_same_count(const std::shared_ptr<Tensor> &in) {
        if (get_count() == in->get_count()) {
            std::memcpy(get_values().data(), in->get_values().data(), sizeof(float) * get_count());
        }
    }
 private:
    void dims_from_shapes(int n, int c, int h, int w) {
        bool dims_start = false;
        int dims = 0;
        if (n > 0) {
            dims_start = true;
            dims++;
        } else {
            CHECK(dims_start, false, "Tensor shapes illegal:" << n << ", " << c << ", " << h << ", " << w << "\n");
        }
        if (c > 0) {
            dims_start = true;
            dims++;
        } else {
            CHECK(dims_start, false, "Tensor shapes illegal:" << n << ", " << c << ", " << h << ", " << w << "\n");
        }
        if (h > 0) {
            dims_start = true;
            dims++;
        } else {
            CHECK(dims_start, false, "Tensor shapes illegal:" << n << ", " << c << ", " << h << ", " << w << "\n");
        }
        if (w > 0) {
            dims_start = true;
            dims++;
        } else {
            CHECK(dims_start, false, "Tensor shapes illegal:" << n << ", " << c << ", " << h << ", " << w << "\n");
        }
        dims_ = dims;
    }

 private:
    std::vector<float> values_;
    int n_, c_, h_, w_;
    int dims_;
    std::string name_;
};

}
