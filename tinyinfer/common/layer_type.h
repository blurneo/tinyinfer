#pragma once

typedef enum LayerType
{
  LAYER_ADD = 0,
  LAYER_CONVOLUTION = 1,
  LAYER_LRN = 2,
  LAYER_MATMUL = 3,
  LAYER_MAXPOOL = 4,
  LAYER_RELU = 5,
  LAYER_RESHAPE = 6,
  LAYER_SOFTMAX = 7,
  LAYER_CLIP = 8,
  LAYER_GLOBAL_AVERAGE_POOL = 9,
  LAYER_FLATTEN = 10,
  LAYER_BATCH_NORMALIZATION = 11,
  LAYER_GEMM = 12,
  LAYER_SIGMOID = 13,
  LAYER_PRELU = 14,
  LAYER_NONE = 15,
} LayerType;
