#pragma once
#include <memory>
#include <vector>
#include <cmath>
#include <string.h>
#include "flow/optical_flow.h"
#include "tinyinfer/common/check_macro.h"

namespace tf
{

    bool
    resize_image(const Image<uint8_t> &in, float x_scale, float y_scale, Image<uint8_t> &out);

    bool get_image_derivative_x(const Image<uint8_t> &in, Image<uint8_t> &out)
    {
        return true;
    }
    bool get_image_derivative_y(const Image<uint8_t> &in, Image<uint8_t> &out)
    {
        return true;
    }

    bool calc_spatial_gradient_matrix(const Image<uint8_t> &Ix, const Image<uint8_t> &Iy,
                                      const Vec2 &input_point, Vec2 window_size, Matrix2x2 &G)
    {
        if (Ix.channels != 1 || Iy.channels != 1)
            return false;
        if (Ix.rows != Iy.rows || Ix.cols != Iy.cols || Ix.channels != Iy.channels)
            return false;
        int l_bound = input_point.x - window_size.x;
        int r_bound = input_point.x + window_size.x;
        l_bound = l_bound < 0 ? 0 : l_bound;
        r_bound = r_bound >= Ix.cols ? Ix.cols - 1 : r_bound;
        int d_bound = input_point.y - window_size.y;
        int u_bound = input_point.y + window_size.y;
        u_bound = u_bound < 0 ? 0 : u_bound;
        d_bound = d_bound >= Ix.rows ? Ix.rows - 1 : r_bound;
        for (int x = l_bound; x <= r_bound; x++)
        {
            for (int y = d_bound; y <= u_bound; y++)
            {
                G.vals[0] += (Ix[y][x] * Ix[y][x]);
                G.vals[1] += (Ix[y][x] * Iy[y][x]);
                G.vals[2] += (Ix[y][x] * Iy[y][x]);
                G.vals[3] += (Iy[y][x] * Iy[y][x]);
            }
        }
        return true;
    }

    bool iterate_guess(const Vec2 &input_point, const Image<uint8_t> &I_L, const Image<uint8_t> &J_L,
                       const Image<uint8_t> &Ix, const Image<uint8_t> &Iy, Vec2 window_size,
                       const Vec2 &g_L, Vec2 &v, int K, const Matrix2x2 &G)
    {
        // image mismatch vector
        int l_bound = input_point.x - window_size.x;
        int r_bound = input_point.x + window_size.x;
        l_bound = l_bound < 0 ? 0 : l_bound;
        r_bound = r_bound >= Ix.cols ? Ix.cols - 1 : r_bound;
        int d_bound = input_point.y - window_size.y;
        int u_bound = input_point.y + window_size.y;
        u_bound = u_bound < 0 ? 0 : u_bound;
        d_bound = d_bound >= Ix.rows ? Ix.rows - 1 : r_bound;
        for (int k = 1; k <= K; k++)
        {
            int jx = g_L.x + v.x;
            int jy = g_L.y + v.y;
            Vec2 bk;
            for (int x = l_bound; x <= r_bound; x++)
            {
                for (int y = u_bound; y <= d_bound; y++)
                {
                    auto deltaIk = I_L[y][x] - J_L[y + jy][x + jx];
                    bk.x = bk.x + deltaIk * Ix[y][x];
                    bk.y = bk.y + deltaIk * Iy[y][x];
                }
            }
            Matrix2x2 G_inv;
            bool ret = G.inverse(G_inv);
            CHECK_BOOL_RET(ret, true, "G is not inversable")
            Vec2 nk; // = G_inv * bk;
            v = v + nk;
        }
        return true;
    }
    bool calc_flow_one_level(const Image<uint8_t> &I_L, const Image<uint8_t> &J_L, const Image<uint8_t> &Ix, const Image<uint8_t> &Iy,
                             const Vec2 &input_point, Vec2 &d_L, Vec2 &g_L_1, Vec2 &g_L, int level, Vec2 window_size, int K)
    {
        Vec2 input_points_l = input_point;
        float _2L = (float)std::pow(2.0, level);
        input_points_l = input_point / _2L;
        Matrix2x2 G;
        calc_spatial_gradient_matrix(Ix, Iy, input_points_l, window_size, G);
        Vec2 v;
        iterate_guess(input_point, I_L, J_L, Ix, Iy, window_size, g_L, v, K, G);
        d_L = v;
        g_L_1 = (g_L + d_L) * 2;
        return true;
    }
    bool calc_optical_flow_pyramid_lk(const Image<uint8_t> &from, const Image<uint8_t> &to,
                                      const std::vector<Vec2> &input_points, std::vector<Vec2> &output_points,
                                      Vec2 window_size, int max_level, int K)
    {
        if (from.channels != 1 || to.channels != 1)
            return false;
        std::vector<Image<uint8_t>> I_L(max_level);
        std::vector<Image<uint8_t>> J_L(max_level);
        bool ret = false;
        float xscale = 0.5, yscale = 0.5;
        std::vector<Image<uint8_t>> Ixs, Iys;
        for (int level = max_level - 1; level >= 0; max_level--)
        {
            ret = resize_image(from, xscale, yscale, I_L[level]);
            ret = resize_image(to, xscale, yscale, J_L[level]);

            get_image_derivative_x(I_L[level], Ixs[level]);
            get_image_derivative_y(I_L[level], Iys[level]);
            xscale *= 0.5;
            yscale *= 0.5;
        }
        output_points.resize(input_points.size());
        for (int pi = 0; pi < input_points.size(); pi++)
        {
            std::vector<Vec2> d_L(max_level);
            std::vector<Vec2> g_L(max_level + 1);
            for (int level = max_level - 1; level >= 0; max_level--)
            {
                ret = calc_flow_one_level(I_L[level], J_L[level], Ixs[level], Iys[level],
                                          input_points[pi], d_L[level], g_L[level], g_L[level + 1], level, window_size, K);
            }
            Vec2 d = g_L[0] + d_L[0];
            Vec2 v = input_points[pi] + d;
            output_points[pi] = v;
        }
        return true;
    }
}
