#include "tinyinfer/layer/pp/im2col/im2col.h"
#include <stdlib.h>

bool im2col(int in_N, int in_C, int in_H, int in_W, const float *input_ptr,
            int k_N, int k_C, int k_H, int k_W, const float *weight_ptr, const float *bias_ptr,
            int stride_x, int stride_y, int pad_t, int pad_d, int pad_l, int pad_r,
            int group, int dilation_x, int dilation_y, int pad_type,
            int out_H, int out_W, float **mat_b)
{
  // if (group != 1) return false;
  if (bias_ptr)
    return false;
  // matrix of row: in_C * k_H * k_W and col: out_H * out_W
  *mat_b = (float *)(malloc(sizeof(float) * (in_C * k_H * k_W) * (out_H * out_W)));
  // TODO: gemm support transpose a or b to directly use the weight and bias of the kernels
  // matrix of row: k_C * k_H * k_N and col: k_N
  // *mat_b = (float*)(malloc(sizeof(float) * (k_C * k_H * k_N) * (k_N)));

  // param def
  int IN_T_C_DIV_GRP = in_C / group;
  int KERNEL_N_DIV_GRP = k_N / group;
  int k_chw = k_C * k_H * k_W;

  int out_HW = out_H * out_W;
  for (int gidx = 0; gidx < group; gidx++)
  {
    // for (int k_n_ = 0; k_n_ < KERNEL_N_DIV_GRP; k_n_++) {
    //   // TODO: fix when group is not 1
    //   int k_n = k_n_ + gidx * KERNEL_N_DIV_GRP;
    for (int in_h = 0, oh = 0; oh < out_H; in_h += stride_y, oh++)
    {
      int idx1 = in_h * in_W;
      for (int in_w = 0, ow = 0; ow < out_W; in_w += stride_x, ow++)
      {
        int idx2 = idx1 + in_w;
        float res = 0;
        // int b_idx1 = (oh + ow) * k_chw;
        for (int k_c = 0; k_c < k_C; k_c++)
        {
          int in_c = k_c + gidx * IN_T_C_DIV_GRP;
          int idx3 = in_c * (in_H * in_W) + idx2;
          for (int h = 0; h < k_H; h++)
          {
            for (int w = 0; w < k_W; w++)
            {
              int t_idx = idx3 + h * in_W + w;
              //   int w_idx = k_n * k_C * k_H * k_W +
              //               k_c * k_H * k_W + h * k_W + w;
              int b_idx2 = (k_c * k_H * k_W + h * k_W + w) * out_HW + oh * out_W + ow;
              (*mat_b)[b_idx2] = input_ptr[t_idx];
            }
          }
        }
      }
    }
    // }
  }

  return true;
}
