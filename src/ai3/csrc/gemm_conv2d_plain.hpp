#pragma once

#include "ai3.hpp"

template <typename dtype>
Tensor<dtype>
guess_conv2d(Tensor<dtype> input, const Tensor<dtype> &kernel,
             const std::optional<const Tensor<dtype>> &bias,
             const std::vector<uint> &padding, const std::vector<uint> &stride,
             const std::vector<uint> &dilation, const PaddingMode padding_mode,
             uint groups, Context &ctx) {
    (void)input;
    (void)kernel;
    (void)bias;
    (void)padding;
    (void)stride;
    (void)dilation;
    (void)padding_mode;
    (void)groups;
    (void)ctx;
    errs::bail("gemm not implemented outside of cuDNN");
}
