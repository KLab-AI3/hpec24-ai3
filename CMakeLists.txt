cmake_minimum_required(VERSION 3.15...3.27)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

project(
    ${SKBUILD_PROJECT_NAME}
    VERSION ${SKBUILD_PROJECT_VERSION}
    LANGUAGES CXX)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

set(CSRC_DIR src/ai3/csrc)

python_add_library(core MODULE ${CSRC_DIR}/ai3.cpp WITH_SOABI)

set_property(TARGET core PROPERTY CXX_STANDARD 17)
set_property(TARGET core PROPERTY CXX_STANDARD_REQUIRED YES)
set_property(TARGET core PROPERTY CXX_EXTENSIONS NO)
target_link_libraries(core PRIVATE pybind11::headers)
target_include_directories(core PRIVATE ${CSRC_DIR})

find_package(CUDAToolkit)

if(CUDAToolkit_FOUND)
    message(STATUS "Using CUDAToolkit")
    set_target_properties(core PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED TRUE
        CUDA_EXTENSIONS OFF
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "86"
    )

    target_link_libraries(core PRIVATE CUDA::cudart CUDA::cublas cudnn)
    target_compile_definitions(core PRIVATE USE_CUDNN)
endif()

find_package(SYCL)
if(SYCL_FOUND)
    message(STATUS "Using SYCL")
    set(CMAKE_CXX_COMPILER "${SYCL_COMPILER}")

    target_include_directories(core PRIVATE ${SYCL_INCLUDE_DIR})
    target_include_directories(core PRIVATE ${SYCL_SYCL_INCLUDE_DIR})
    target_link_directories(core PRIVATE ${SYCL_LIBRARY_DIR})

    separate_arguments(SYCL_CFLAGS)
    target_compile_options(core PRIVATE "${SYCL_CFLAGS}")

    separate_arguments(SYCL_LFLAGS)
    target_link_options(core PRIVATE "${SYCL_LFLAGS}")

    target_compile_definitions(core PRIVATE USE_SYCL)
endif()

install(TARGETS core DESTINATION ${SKBUILD_PROJECT_NAME})
