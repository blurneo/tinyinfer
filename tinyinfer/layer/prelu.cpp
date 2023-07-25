#pragma once
#include <cmath>
#include "tinyinfer/common/check_macro.h"
#include "tinyinfer/common/tensor.h"
#include "tinyinfer/common/base_layer.h"
#include "tinyinfer/layer/prelu.h"
#include "tinyinfer/reflection/serializer.h"
#include "tinyinfer/reflection/deserializer.h"

namespace ti
{

    bool PRelu::forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                        std::vector<std::shared_ptr<Tensor>> output_tensors)
    {
        CHECK_BOOL_RET(input_tensors.size(), 1,
                       "PRelu input tensor number should be 1")
        CHECK_BOOL_RET(param_.slope->can_uni_broadcast(input_tensors[0]), true,
                       "PRelu input tensor count should be uni broadcastable to slope")
        std::shared_ptr<Tensor> input_tensor = input_tensors[0];
        std::shared_ptr<Tensor> output_tensor = output_tensors[0];
        output_tensor->reshape_like(input_tensor);
        // calculate computation and memory infomation
        flops_ = 0;
        bytes_ = input_tensor->get_bytes() + output_tensor->get_bytes();
        return kernel(input_tensor, output_tensor);
    }

    bool PRelu::kernel(std::shared_ptr<Tensor> input_tensor,
                       std::shared_ptr<Tensor> output_tensor)
    {
        int IN_T_N = input_tensor->get_n();
        int IN_T_C = input_tensor->get_c();
        int IN_T_H = input_tensor->get_h();
        int IN_T_W = input_tensor->get_w();
        const std::vector<float> &input_vals = input_tensor->get_values();
        int SLOPE_T_N = param_.slope->get_n();
        int SLOPE_T_C = param_.slope->get_c();
        int SLOPE_T_H = param_.slope->get_h();
        int SLOPE_T_W = param_.slope->get_w();
        std::vector<float> &slope_vals = param_.slope->get_values();
        int SLOPE_LOOP_N = SLOPE_T_N == 0 ? 1 : SLOPE_T_N;
        int SLOPE_LOOP_C = SLOPE_T_C == 0 ? 1 : SLOPE_T_C;
        int SLOPE_LOOP_H = SLOPE_T_H == 0 ? 1 : SLOPE_T_H;
        int SLOPE_LOOP_W = SLOPE_T_W == 0 ? 1 : SLOPE_T_W;
        int OUT_T_N = output_tensor->get_n();
        int OUT_T_C = output_tensor->get_c();
        int OUT_T_H = output_tensor->get_h();
        int OUT_T_W = output_tensor->get_w();
        std::vector<float> &output_vals = output_tensor->get_values();
        float inf = std::numeric_limits<float>::max();
        float neg_inf = std::numeric_limits<float>::lowest();
        for (int in_n = 0; in_n < IN_T_N; in_n++)
        {
            int s_n = std::min(in_n, SLOPE_T_N);
            for (int in_c = 0; in_c < IN_T_C; in_c++)
            {
                int s_c = std::min(in_c, SLOPE_T_C);
                for (int in_h = 0; in_h < IN_T_H; in_h++)
                {
                    int s_h = std::min(in_h, SLOPE_T_H);
                    for (int in_w = 0; in_w < IN_T_W; in_w++)
                    {
                        int i_idx = in_n * IN_T_C * IN_T_H * IN_T_W +
                                    in_c * IN_T_H * IN_T_W + in_h * IN_T_W + in_w;
                        int s_w = std::min(in_w, SLOPE_T_W);
                        float in = input_vals[i_idx];
                        float slo = slope_vals[s_n * SLOPE_T_C * SLOPE_T_H * SLOPE_T_W +
                                               s_c * SLOPE_T_H * SLOPE_T_W + s_h * SLOPE_T_W + s_w];
                        output_vals[i_idx] = std::max(std::min(inf, in), 0.f) + std::max(std::min(in, 0.f), neg_inf) * slo;
                    }
                }
            }
        }
        return true;
    }

    void PRelu::serialize(Serializer &serializer)
    {
        BaseLayer::serialize(serializer);
        PRelu::serialize_internal(serializer);
    }

    bool PRelu::deserialize(Deserializer &deserializer)
    {
        CHECK_BOOL_RET(BaseLayer::deserialize(deserializer), true, "PRelu baselayer deserialize failed");
        return PRelu::deserialize_internal(deserializer);
    }

} // namespace ti
