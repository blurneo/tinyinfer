#pragma once
#include <vector>
#include <cmath>
#include "ann/k2dnode.h"

namespace ta {

class K2dTree {
 public:

    K2dTree() {}

    virtual ~K2dTree() {}

    void add_node(float x, float y) {
        if (!root_) {
            root_ = new K2dNode(x, y);
            return;
        }
        bool x_flag = true;
        K2dNode* cur_node = root_;
        while (true) {
            cur_node->subtree_size++;
            if (cur_node->range_min.gt(x, y))
                cur_node->range_min = K2dNodeValue(x, y);
            if (cur_node->range_max.lt(x, y))
                cur_node->range_max = K2dNodeValue(x, y);

            if (cur_node->value.gt_flag(x_flag, x, y)) {
                if (!cur_node->left) {
                    cur_node->left = new K2dNode(x, y);
                    break;
                } else {
                    cur_node = cur_node->left;
                }
            } else {
                if (!cur_node->right) {
                    cur_node->right = new K2dNode(x, y);
                    break;
                } else {
                    cur_node = cur_node->right;
                }
            }
            x_flag = !x_flag;
        }
    }

    void lazy_delete(float x, float y) {
        K2dNode* searched_node = this->search(x, y);
        if (!searched_node) {
            return;
        }
        K2dNode* cur_node = root_;
        bool x_flag = true;
        while (cur_node != searched_node) {
            cur_node->invalid_num++;
            if (cur_node->value.gt_flag(x_flag, x, y)) {
                cur_node = cur_node->left;
            } else {
                cur_node = cur_node->right;
            }
            x_flag = !x_flag;
        }
        cur_node->invalid_num++;
        cur_node->deleted = true;
    }

    K2dNode* search(float x, float y, bool *ret_flag = nullptr) {
        if (!root_) {
            return nullptr;
        }
        bool x_flag = true;
        K2dNode* cur_node = root_;
        while (true) {
            if (cur_node->value.equal(x, y)) {
                if (ret_flag) *ret_flag = !x_flag;
                return cur_node;
            } else if (cur_node->value.gt_flag(x_flag, x, y)) {
                if (!cur_node->left) {
                    break;
                } else {
                    cur_node = cur_node->left;
                }
            } else {
                if (!cur_node->right) {
                    break;
                } else {
                    cur_node = cur_node->right;
                }
            }
            x_flag = !x_flag;
        }
        return nullptr;
    }

    std::vector<K2dNode*> search_knn(float x, float y) {
        return {};
    }

    void rebalance() {

    }

 private:
    K2dNode* root_ = nullptr;
};

}
