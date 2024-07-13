#pragma once

#include "ai3.hpp"
#include "conv2d_exec_cudnn.hpp"

template <typename dtype>
Tensor<dtype>
guess_conv2d(Tensor<dtype> input, const Tensor<dtype> &kernel,
             const std::optional<const Tensor<dtype>> &bias,
             const std::vector<uint> &padding, const std::vector<uint> &stride,
             const std::vector<uint> &dilation, const PaddingMode padding_mode,
             uint groups, Context &ctx) {
    return conv_bias_forward_with_algo<dtype>(
        std::move(input), kernel, bias, padding, stride, dilation, padding_mode,
        groups, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, ctx, true);
}
