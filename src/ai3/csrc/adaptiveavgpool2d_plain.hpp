#pragma once

#include "ai3.hpp"
#include "avgpool2d_plain.hpp"
#include <optional>
#include <vector>

template <typename dtype>
Tensor<dtype> _adaptiveavgpool2d(
    Tensor<dtype> input,
    const std::optional<std::vector<std::optional<uint>>> output_shape) {
    uint input_height = input.height();
    uint input_width = input.width();
    const std::vector<std::optional<uint>> opt_in_shape = {input_height,
                                                           input_width};
    const uint output_height =
        output_shape.value_or(opt_in_shape)[0].value_or(input_height);
    const uint output_width =
        output_shape.value_or(opt_in_shape)[1].value_or(input_width);
    errs::bail_if(
        input_height % output_height != 0 || input_width % output_width != 0,
        "Adaptive average pooling not implemented for cases where "
        "input size is not a multiple of output size given input shape=(",
        input_height, ", ", input_width, ") and output shape=(", output_height,
        ", ", output_width, ")");
    std::vector<uint> stride = {input_height / output_height,
                                input_width / output_width};
    std::vector<uint> kernel_shape = {
        input_height - ((output_height - 1) * stride[0]),
        input_width - ((output_width - 1) * stride[1])};

    return _avgpool2d<dtype>(std::move(input), kernel_shape, {0, 0}, stride,
                             false, false, std::nullopt);
}
