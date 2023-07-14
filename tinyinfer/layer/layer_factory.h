#pragma once

#include <memory>
#include "base_layer.h"
#include "tinyinfer/layer/add.h"
#include "tinyinfer/layer/clip.h"
#include "tinyinfer/layer/convolution.h"
#include "tinyinfer/layer/flatten.h"
#include "tinyinfer/layer/gemm.h"
#include "tinyinfer/layer/global_average_pool.h"
#include "tinyinfer/layer/batch_normalization.h"
#include "tinyinfer/layer/lrn.h"
#include "tinyinfer/layer/matmul.h"
#include "tinyinfer/layer/max_pool.h"
#include "tinyinfer/layer/relu.h"
#include "tinyinfer/layer/reshape.h"
#include "tinyinfer/layer/softmax.h"

namespace ti {

class LayerFactory {
 public:
    static std::shared_ptr<BaseLayer> get(LayerType type) {
        std::shared_ptr<BaseLayer> ret;
        switch (type) {
        case LAYER_ADD:
            ret.reset(new Add());
            break;
        case LAYER_CONVOLUTION:
            ret.reset(new Convolution());
            break;
        case LAYER_LRN:
            ret.reset(new Lrn());
            break;
        case LAYER_MATMUL:
            ret.reset(new Matmul());
            break;
        case LAYER_MAXPOOL:
            ret.reset(new MaxPool());
            break;
        case LAYER_RELU:
            ret.reset(new Relu());
            break;
        case LAYER_RESHAPE:
            ret.reset(new Reshape());
            break;
        case LAYER_SOFTMAX:
            ret.reset(new Softmax());
            break;
        case LAYER_CLIP:
            ret.reset(new Clip());
            break;
        case LAYER_GLOBAL_AVERAGE_POOL:
            ret.reset(new GlobalAveragePool());
            break;
        case LAYER_FLATTEN :
            ret.reset(new Flatten());
            break;
        case LAYER_BATCH_NORMALIZATION :
            ret.reset(new BatchNormalization());
            break;
        case LAYER_GEMM :
            ret.reset(new Gemm());
            break;
        case LAYER_NONE :
            break;
        default:
            break;
        }
        return ret;
    }
 private:
};

}
