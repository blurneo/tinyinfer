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
    void reshape(int n, int c, int h, int w) {
        set_n(n);
        set_c(c);
        set_h(h);
        set_w(w);
        values_.resize(n * c * h * w, 0);
    }
    float* ptr() { return values_.data(); }
    const float* ptr() const { return values_.data(); }

    static void pad(const Tensor &in, Tensor &out,
                    int pad_t, int pad_d, int pad_l, int pad_r) {
        int in_n = in.get_n();
        int in_c = in.get_c();
        int in_h = in.get_h();
        int in_w = in.get_w();
        int out_n = in.get_n();
        int out_c = in.get_c();
        int out_h = in.get_h() + pad_t + pad_d;
        int out_w = in.get_w() + pad_l + pad_r;
        out.reshape(out_n, out_c, out_h, out_w);
        int start_h = pad_t;
        int start_w = pad_l;
        for (int c = 0; c < in_c; c++) {
            for (int ih = 0, oh = start_h; ih < in_h; oh++, ih++) {
                const float *src = in.ptr() + c * in_h * in_w + ih * in_w;
                float *dst = out.ptr() + c * out_h * out_w + oh * out_w + start_w;
                std::memcpy(dst, src, in_w * sizeof(float));
            }
        }
    }

 private:
    std::vector<float> values_;
    int n_, c_, h_, w_;
};

}