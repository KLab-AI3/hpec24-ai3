#pragma once
#include <ai3.hpp>

constexpr bool DEFAULT_TO_CUSTOM_CONV2D = false;

template <typename dtype>
Tensor<dtype> conv2d(Tensor<dtype> input, const Tensor<dtype> &kernel,
                     const std::optional<const Tensor<dtype>> &bias,
                     const std::vector<uint> &padding,
                     const std::vector<uint> &stride,
                     const std::vector<uint> &dilation,
                     const PaddingMode padding_mode, uint groups) {
    errs::no_user_def("conv2d");
}
