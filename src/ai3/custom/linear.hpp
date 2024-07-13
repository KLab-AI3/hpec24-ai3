#pragma once
#include <ai3.hpp>

constexpr bool DEFAULT_TO_CUSTOM_LINEAR = false;

template <typename dtype>
Tensor<dtype> linear(Tensor<dtype> input, const Tensor<dtype> &weight,
                     const std::optional<const Tensor<dtype>> &bias) {
    errs::no_user_def("linear");
}
