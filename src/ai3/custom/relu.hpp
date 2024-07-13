#pragma once

#include <ai3.hpp>

constexpr bool DEFAULT_TO_CUSTOM_RELU = false;

template <typename dtype> Tensor<dtype> relu(Tensor<dtype> input) {
    errs::no_user_def("relu");
}
