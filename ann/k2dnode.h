#pragma once
#include <cmath>

namespace ta {

typedef struct K2dNodeValue {
    float x = 0.f;
    float y = 0.f;
    K2dNodeValue(float _x, float _y) : x(_x), y(_y) {}
    bool x_gt(const K2dNodeValue &value) {
        return this->x > value.x;
    }
    bool y_gt(const K2dNodeValue &value) {
        return this->y > value.y;
    }
    bool x_gt(float in) {
        return this->x > in;
    }
    bool y_gt(float in) {
        return this->y > in;
    }
    bool x_lt(float in) {
        return this->x < in;
    }
    bool y_lt(float in) {
        return this->y < in;
    }
    float distance() {
        return this->x * this->x + this->y * this->y;
    }
    bool gt(float in_x, float in_y) {
        return this->distance() > (in_x * in_x + in_y * in_y);
    }
    bool lt(float in_x, float in_y) {
        return this->distance() < (in_x * in_x + in_y * in_y);
    }
    bool gt_flag(bool x_flag, float in_x, float in_y) {
        if (x_flag) {
            return this->x_gt(in_x);
        } else {
            return this->y_gt(in_y);
        }
    }
    bool equal(float in_x, float in_y) {
        return std::fabs(this->x - in_x) < 1e-5 && std::fabs(this->y - in_y) < 1e-5;
    }
} K2dNodeValue;

typedef struct K2dNode {
    K2dNode *left = nullptr;
    K2dNode *right = nullptr;
    K2dNodeValue value;
    K2dNodeValue range_min;
    K2dNodeValue range_max;
    int subtree_size = 1;
    int invalid_num = 0;
    bool deleted = false;
    K2dNode(float x, float y) : value(x, y), range_min(x, y), range_max(x, y) {}
} K2dNode;

}
