#pragma once

#include <ai3.hpp>

constexpr bool DEFAULT_TO_CUSTOM_AVGPOOL2D = false;

template <typename dtype>
Tensor<dtype>
avgpool2d(Tensor<dtype> input, const std::vector<uint> kernel_shape,
          const std::vector<uint> &padding, const std::vector<uint> &stride,
          const bool ceil_mode, const bool count_include_pad,
          const std::optional<int> divisor_override) {
    errs::no_user_def("avgpool2d");
}
