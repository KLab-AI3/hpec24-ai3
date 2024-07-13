#pragma once

#include <iostream>
#include <memory>
#include <vector>

#ifdef USE_CUDNN
#include "cuda_utils.hpp"
#include <cudnn.h>
#endif

#ifdef USE_SYCL
#include <CL/sycl.hpp>
using namespace cl;
#endif

struct Context {
    Context() {
#ifdef USE_CUDNN
        CUDNN_CHECK(cudnnCreate(&cudnn_handle));
#endif

#ifdef USE_SYCL
        sycl_queue = sycl::queue(sycl::default_selector_v);
#endif
    }

#ifdef USE_CUDNN
    cudnnHandle_t cudnn_handle;
#endif
#ifdef USE_SYCL
    sycl::queue sycl_queue;
#endif

    ~Context() {
#ifdef USE_CUDNN
        cudnnDestroy(cudnn_handle);
#endif
    }
};
