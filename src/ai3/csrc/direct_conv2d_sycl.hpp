#pragma once

#include "ai3.hpp"
#include <CL/sycl.hpp>
#include <chrono>
#include <optional>
using namespace cl;

template <typename dtype>
Tensor<dtype> direct_conv2d(Tensor<dtype> input, const Tensor<dtype> &kernel,
                            const std::optional<const Tensor<dtype>> &bias,
                            const std::vector<uint> &padding,
                            const std::vector<uint> &stride,
                            const std::vector<uint> &dilation,
                            const PaddingMode padding_mode, uint groups) {
    errs::bail_if(padding_mode != PaddingMode::Zeros,
                  "padding mode must be zeroes");
    errs::bail_if(groups != 1, "groups must be 1");

    const uint input_channels = input.input_channels();
    const uint input_height = input.height();
    const uint input_width = input.width();

    const uint kernel_height = kernel.height();
    const uint kernel_width = kernel.width();

    const uint output_channels = kernel.out_channels();

    const uint output_height = output_size_for_2d<dtype>(
        input_height, kernel_height, padding[0], dilation[0], stride[0], false);
    const uint output_width = output_size_for_2d<dtype>(
        input_width, kernel_width, padding[1], dilation[1], stride[1], false);

    uint num_samples;
    Tensor<dtype> output;
    if (input.batched(input_dims::CONV2D)) {
        num_samples = input.batch_size(input_dims::CONV2D);
        output = Tensor<dtype>(
            {num_samples, output_channels, output_height, output_width});
    } else {
        num_samples = 1;
        output = Tensor<dtype>({output_channels, output_height, output_width});
    }

    uint padding_h = padding[0];
    uint padding_w = padding[1];
    uint stride_h = stride[0];
    uint stride_w = stride[1];
    uint dilation_h = dilation[0];
    uint dilation_w = dilation[1];

    sycl::queue queue(sycl::default_selector_v);

    sycl::buffer<dtype> buf_input(input.data, input.count());
    sycl::buffer<dtype> buf_ker(kernel.data, kernel.count());
    sycl::buffer<dtype> buf_output(output.data, output.count());
    const bool has_bias = bias.has_value();
    sycl::buffer<dtype> buf_bias =
        has_bias ? sycl::buffer<dtype>(bias->data, bias->count())
                 : sycl::buffer<dtype>(sycl::range<1>(0));

    const uint output_area = output_height * output_width;
    const uint max_work_group_size =
        queue.get_device().get_info<sycl::info::device::max_work_group_size>();
    uint work_group_size = max_work_group_size;
    if (GROUP_SIZE_GUESS < work_group_size) {
        work_group_size = GROUP_SIZE_GUESS;
    }
    if (output_area < work_group_size) {
        work_group_size = output_area;
    }
    dtype scaler =
        std::sqrt(dtype(work_group_size) / (output_channels * output_area));
    uint each_channel = output_channels * scaler;
    uint each_output_area = output_area * scaler;
    if (each_channel == 0) {
        each_channel = 1;
        scaler = 1 / (scaler * output_channels);
        each_output_area /= scaler;
        if (each_output_area == 0) {
            each_output_area = 1;
        }
    }
    if (each_output_area == 0) {
        scaler = 1 / (scaler * each_output_area);
        each_output_area = 1;
        each_channel /= scaler;
        if (each_channel == 0) {
            each_channel = 1;
        }
    }

    const uint total_output_area =
        ((output_area + each_output_area - 1) / each_output_area) *
        each_output_area;
    const uint total_channels =
        ((output_channels + each_channel - 1) / each_channel) * each_channel;

    queue.submit([&](sycl::handler &h) {
        auto acc_input =
            buf_input.template get_access<sycl::access::mode::read>(h);
        auto acc_bias =
            buf_bias.template get_access<sycl::access::mode::read>(h);
        auto acc_kernel =
            buf_ker.template get_access<sycl::access::mode::read>(h);
        auto acc_output =
            buf_output.template get_access<sycl::access::mode::write>(h);
        h.parallel_for(
            sycl::nd_range(
                sycl::range(num_samples, total_channels, total_output_area),
                sycl::range(1, each_channel, each_output_area)),
            [=](sycl::nd_item<3> item) {
                const uint samp = item.get_global_id(0);
                const uint out_c = item.get_global_id(1);
                if (out_c >= output_channels) {
                    return;
                }
                const uint area_id = item.get_global_id(2);
                if (area_id >= output_area) {
                    return;
                }
                const uint out_h = area_id / output_width;
                const uint out_w = area_id % output_width;
                dtype sum = 0;
                uint h_offset_base = out_h * stride_h - padding_h;
                uint w_offset_base = out_w * stride_w - padding_w;
                for (uint in_c = 0; in_c < input_channels; ++in_c) {
                    for (uint ker_h = 0; ker_h < kernel_height; ++ker_h) {
                        for (uint ker_w = 0; ker_w < kernel_width; ++ker_w) {
                            uint h_offset = h_offset_base + ker_h * dilation_h;
                            uint w_offset = w_offset_base + ker_w * dilation_w;
                            if (h_offset >= 0 && h_offset < input_height &&
                                w_offset >= 0 && w_offset < input_width) {
                                sum += acc_input[to_linear(
                                           samp, in_c, h_offset, w_offset,
                                           input_channels, input_height,
                                           input_width)] *
                                       acc_kernel[to_linear(
                                           out_c, in_c, ker_h, ker_w,
                                           input_channels, kernel_height,
                                           kernel_width)];
                            }
                        }
                    }
                }
                if (has_bias) {
                    sum += acc_bias[out_c];
                }
                acc_output[to_linear(samp, out_c, out_h, out_w, output_channels,
                                     output_height, output_width)] = sum;
            });
    });

    queue.wait_and_throw();

    return output;
}
