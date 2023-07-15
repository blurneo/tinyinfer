#pragma once

#include "common/check_macro.h"
#include <memory>
#include <optional>
#include <string>
#include <cmath>
#include <vector>
#include "tinyinfer/reflection/serialize_macro.h"

namespace ti {

class Serializer;
class Deserializer;
class Tensor {
public:
  Tensor() : n_(0), c_(0), h_(0), w_(0) {}
  Tensor(int n, int c, int h, int w)
      : n_(n), c_(c), h_(h), w_(w), values_(std::vector<float>(n * c * h * w)) {
    dims_from_shapes(n, c, h, w);
  }
  Tensor(int n, int c, int h, int w, std::vector<float> &&values)
      : n_(n), c_(c), h_(h), w_(w), values_(std::move(values)) {
    dims_from_shapes(n, c, h, w);
  }
  std::shared_ptr<Tensor> clone();
  void set_name(std::string name);
  std::string get_name() const;
  int get_count() const;
  int get_n() const;
  int get_c() const;
  int get_h() const;
  int get_w() const;
  void set_n(int n);
  void set_c(int c);
  void set_h(int h);
  void set_w(int w);
  const std::vector<float> &get_values() const;
  std::vector<float> &get_values();
  void reshape(std::vector<int> dims_vec);
  void reshape(int n, int c, int h, int w);
  float *ptr();
  const float *ptr() const;
  int dims();
  int dims() const;
  std::vector<int> dims_vector() const;
  std::optional<int> dim_stride(int dim_idx) const;

  // TODO: handle when tensor dimension is not 4
  static void pad(const std::shared_ptr<Tensor> &in,
                  std::shared_ptr<Tensor> &out, int pad_t, int pad_d, int pad_l,
                  int pad_r);

  bool is_alike(const std::shared_ptr<Tensor> &in) const;

  void reshape_like(const std::shared_ptr<Tensor> &in);

  bool is_matrix();
  bool can_multiply(const std::shared_ptr<Tensor> &in) const;
  void copy_if_same_count(const std::shared_ptr<Tensor> &in);
  void transpose_2d();
  std::shared_ptr<Tensor> get_transpose_2d();
  bool can_uni_broadcast(const std::shared_ptr<Tensor> &tensor);

private:
  void dims_from_shapes(int n, int c, int h, int w);

public:
  virtual void serialize(Serializer& serializer);
  virtual bool deserialize(Deserializer& deserializer);


private:
  std::vector<float> values_;
  int n_, c_, h_, w_;
  int dims_;
  std::string name_;
  DEFINE_SERIALIZE_MEMBER(
  ("values", values_)
  ("n", n_)
  ("c", c_)
  ("h", h_)
  ("w", w_)
  ("dims", dims_)
  ("name", name_)
  )
};

} // namespace ti
