#pragma once

#include "ai3.hpp"
#include <optional>
#include <vector>

template <typename dtype>
Tensor<dtype> _linear(Tensor<dtype> input, const Tensor<dtype> &weight,
                      const std::optional<const Tensor<dtype>> &bias) {
    errs::bail_if(input.width() != weight.width(),
                  "Invalid matrix multiplication: input width=", input.width(),
                  " weight width=", weight.width());

    const uint in_features = input.width();
    const uint out_features = weight.height();

    Tensor<dtype> output;
    uint num_samples;
    if (input.batched(input_dims::LINEAR)) {
        num_samples = input.batch_size(input_dims::LINEAR);
        output = Tensor<dtype>({num_samples, out_features});
    } else {
        num_samples = 1;
        output = Tensor<dtype>({out_features});
    }

    const bool has_bias = bias.has_value();
    for (uint s = 0; s < num_samples; s++) {
        for (uint i = 0; i < out_features; i++) {
            dtype res = 0;
            for (int j = 0; j < in_features; ++j) {
                res += weight.data[to_linear(i, j, in_features)] *
                       input.data[to_linear(s, j, in_features)];
            }
            if (has_bias) {
                res += bias->data[i];
            }
            output.data[to_linear(s, i, out_features)] = res;
        }
    }
    return output;
}
