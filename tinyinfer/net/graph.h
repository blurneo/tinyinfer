#pragma once

#include "tinyinfer/layer/base_layer.h"
#include <memory>
#include <vector>

namespace ti {

class Net;
class Graph {
public:
  Graph() {}
  static std::shared_ptr<Graph> FromNet(const Net *net);
  void restart() { current_ = 0; }
  bool is_finished() { return current_ >= nodes_.size(); }
  std::shared_ptr<BaseLayer> next() { return nodes_[current_++]; }

private:
  std::vector<std::shared_ptr<BaseLayer>> nodes_;
  int current_ = 0;
};

} // namespace ti
