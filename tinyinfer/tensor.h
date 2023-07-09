#pragma once

#include <vector>

namespace ti {

class Tensor {
 public:
    Tensor() {}
    Tensor(int n, int c, int h, int w) :
        n_(n), c_(c), h_(h), w_(w), values_(std::vector<float>(n*c*h*w)) {}
    Tensor(int n, int c, int h, int w, std::vector<float> &&values) :
        n_(n), c_(c), h_(h), w_(w), values_(std::move(values)) {}
    int get_count() const { return n_ * c_ * h_ * w_; }
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

 private:
    std::vector<float> values_;
    int n_, c_, h_, w_;
};

}