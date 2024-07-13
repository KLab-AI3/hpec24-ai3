#pragma once

#include <cudnn.h>

template <typename T> cudnnDataType_t cudnnDataType();

template <> cudnnDataType_t cudnnDataType<float>() { return CUDNN_DATA_FLOAT; }

template <> cudnnDataType_t cudnnDataType<double>() {
    return CUDNN_DATA_DOUBLE;
}

#define CUDA_CHECK(status)                                                     \
    if (status != cudaSuccess) {                                               \
        std::cerr << "CUDA error: " << cudaGetErrorString(status)              \
                  << std::endl;                                                \
        exit(1);                                                               \
    }

#define CUDNN_CHECK(status)                                                    \
    if (status != CUDNN_STATUS_SUCCESS) {                                      \
        std::cerr << "cuDNN error: " << cudnnGetErrorString(status)            \
                  << std::endl;                                                \
        exit(1);                                                               \
    }
