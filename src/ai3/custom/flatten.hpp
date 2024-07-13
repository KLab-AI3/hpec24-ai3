#pragma once

#include <ai3.hpp>

constexpr bool DEFAULT_TO_CUSTOM_FLATTEN = false;

template <typename dtype>
Tensor<dtype> flatten(Tensor<dtype> input, const uint start_dim,
                      int end_dim_orig) {
    errs::no_user_def("flatten");
}
