#pragma once

#include <ai3.hpp>

constexpr bool DEFAULT_TO_CUSTOM_MAXPOOL2D = false;

template <typename dtype>
Tensor<dtype>
maxpool2d(Tensor<dtype> input, const std::vector<uint> kernel_shape,
          const std::vector<uint> &padding, const std::vector<uint> &stride,
          const std::vector<uint> &dilation, const bool ceil_mode) {
    errs::no_user_def("maxpool2d");
}
