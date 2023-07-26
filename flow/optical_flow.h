#pragma once
#include <memory>
#include <vector>
#include <cmath>
#include <string.h>
#include "flow/matrix.h"
#include "tinyinfer/common/check_macro.h"
#include "flow/vector.h"

namespace tf
{

    template <typename T>
    struct Image
    {
        int rows;
        int cols;
        int channels;
        int type;
        std::shared_ptr<T> data;
        const T *operator[](int row_idx) const
        {
            return &((data.get())[row_idx * cols * channels]);
        }
    };
    struct Size
    {
        int width;
        int rows;
    };
    
    bool resize_image(const Image<uint8_t> &in,
            float x_scale, float y_scale, Image<uint8_t> &out);

    bool get_image_derivative_x(const Image<uint8_t> &in, Image<uint8_t> &out);
    bool get_image_derivative_y(const Image<uint8_t> &in, Image<uint8_t> &out);

    bool calc_spatial_gradient_matrix(const Image<uint8_t> &Ix, const Image<uint8_t> &Iy,
                                      const Vec2 &input_point, Vec2 window_size, Matrix2x2 &G);

    bool iterate_guess(const Vec2 &input_point, const Image<uint8_t> &I_L, const Image<uint8_t> &J_L,
                       const Image<uint8_t> &Ix, const Image<uint8_t> &Iy, Vec2 window_size,
                       const Vec2 &g_L, Vec2 &v, int K, const Matrix2x2 &G);
    bool calc_flow_one_level(const Image<uint8_t> &I_L, const Image<uint8_t> &J_L, const Image<uint8_t> &Ix, const Image<uint8_t> &Iy,
                             const Vec2 &input_point, Vec2 &d_L, Vec2 &g_L_1, Vec2 &g_L, int level, Vec2 window_size, int K);
    bool calc_optical_flow_pyramid_lk(const Image<uint8_t> &from, const Image<uint8_t> &to,
                                      const std::vector<Vec2> &input_points, std::vector<Vec2> &output_points,
                                      Vec2 window_size, int max_level, int K);
}
