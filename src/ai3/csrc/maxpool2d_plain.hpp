#pragma once

#include "ai3.hpp"
#include <optional>
#include <vector>

template <typename dtype>
Tensor<dtype>
_maxpool2d(Tensor<dtype> input, const std::vector<uint> kernel_shape,
           const std::vector<uint> &padding, const std::vector<uint> &stride,
           const std::vector<uint> &dilation, const bool ceil_mode) {
    const uint input_channels = input.input_channels();
    const uint input_height = input.height();
    const uint input_width = input.width();

    const uint kernel_height = kernel_shape[0];
    const uint kernel_width = kernel_shape[1];

    const uint output_channels = input_channels;

    const uint output_height =
        output_size_for_2d<dtype>(input_height, kernel_height, padding[0],
                                  dilation[0], stride[0], ceil_mode);
    const uint output_width =
        output_size_for_2d<dtype>(input_width, kernel_width, padding[1],
                                  dilation[1], stride[1], ceil_mode);

    Tensor<dtype> output;
    uint num_samples;
    if (input.batched(input_dims::POOL2D)) {
        num_samples = input.batch_size(input_dims::POOL2D);
        output = Tensor<dtype>(
            {num_samples, output_channels, output_height, output_width});
    } else {
        num_samples = 1;
        output = Tensor<dtype>({output_channels, output_height, output_width});
    }

    const dtype lowest = std::numeric_limits<dtype>::lowest();
    for (uint samp = 0; samp < num_samples; samp++) {
        for (uint out_c = 0; out_c < output_channels; out_c++) {
            for (uint out_h = 0; out_h < output_height; out_h++) {
                for (uint out_w = 0; out_w < output_width; out_w++) {
                    dtype cur_max = lowest;
                    for (uint kern_r = 0; kern_r < kernel_height; ++kern_r) {
                        for (uint kern_c = 0; kern_c < kernel_width; ++kern_c) {
                            uint h_offset = out_h * stride[0] - padding[0] +
                                            kern_r * dilation[0];
                            uint w_offset = out_w * stride[1] - padding[1] +
                                            kern_c * dilation[1];

                            if (h_offset >= 0 && h_offset < input_height &&
                                w_offset >= 0 && w_offset < input_width) {
                                dtype cur = input.data[to_linear(
                                    samp, out_c, h_offset, w_offset,
                                    output_channels, input_height,
                                    input_width)];
                                if (cur > cur_max) {
                                    cur_max = cur;
                                }
                            }
                        }
                    }
                    output.data[to_linear(samp, out_c, out_h, out_w,
                                          output_channels, output_height,
                                          output_width)] = cur_max;
                }
            }
        }
    }

    return output;
}
