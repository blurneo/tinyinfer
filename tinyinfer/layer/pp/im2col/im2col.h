#pragma once

bool im2col(int in_N, int in_C, int in_H, int in_W, const float* input_ptr,
            int k_N, int k_C, int k_H, int k_W, const float* weight_ptr, const float* bias_ptr,
            int stride_x, int stride_y, int pad_t, int pad_d, int pad_l, int pad_r,
            int group, int dilation_x, int dilation_y, int pad_type,
            int out_H, int out_W, float** mat_b);
