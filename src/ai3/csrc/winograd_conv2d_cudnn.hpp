#pragma once

#include "ai3.hpp"
#include "conv2d_exec_cudnn.hpp"

template <typename dtype>
Tensor<dtype> winograd_conv2d(Tensor<dtype> input, const Tensor<dtype> &kernel,
                              const std::optional<const Tensor<dtype>> &bias,
                              const std::vector<uint> &padding,
                              const std::vector<uint> &stride,
                              const std::vector<uint> &dilation,
                              const PaddingMode padding_mode, uint groups,
                              Context &ctx) {
    errs::bail_if(stride[0] != 1 || stride[1] != 1,
                  "winograd not implemented for stride not equal to 1 "
                  "see `Supported Algorithms for cudnnConvolutionForward() 2D "
                  "Convolutions. Filter descriptor wDesc: _NCHW` "
                  "at "
                  "https://docs.nvidia.com/deeplearning/cudnn/latest/api/"
                  "cudnn-cnn-library.html");
    errs::bail_if(kernel.width() != 3 || kernel.height() != 3,
                  "winograd not implemented for kernel height and kernel width "
                  "not equal to 3 "
                  "see `Supported Algorithms for cudnnConvolutionForward() 2D "
                  "Convolutions. Filter descriptor wDesc: _NCHW` "
                  "at "
                  "https://docs.nvidia.com/deeplearning/cudnn/latest/api/"
                  "cudnn-cnn-library.html");
    return conv_bias_forward_with_algo(
        std::move(input), kernel, bias, padding, stride, dilation, padding_mode,
        groups, CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD, ctx);
}
