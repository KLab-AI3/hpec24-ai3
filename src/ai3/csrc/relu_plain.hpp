#pragma once

#include "ai3.hpp"
#include <optional>
#include <vector>

template <typename dtype> Tensor<dtype> _relu(Tensor<dtype> input) {
    int total_elements = input.count();
    for (int i = 0; i < total_elements; i++) {
        input.data[i] = (input.data[i] > 0) ? input.data[i] : 0;
    }
    return input;
}
