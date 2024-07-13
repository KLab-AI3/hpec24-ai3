#pragma once

#include <cmath>
#include <iostream>
#include <optional>
#include <sstream>
#include <vector>

#ifndef GROUP_SIZE_GUESS
#define GROUP_SIZE_GUESS 1024
#endif

enum class PaddingMode { Zeros, Reflect, Replicate, Circular };

struct GroupSplit2D {
    uint each_a;
    uint each_b;
    uint total_a;
    uint total_b;
};

template <typename dtype>
void proportionate_2d_work_split(uint num_a, uint num_b, uint total,
                                 uint *set_each_a, uint *set_each_b,
                                 uint *set_total_a, uint *set_total_b) {
    dtype scaler = std::sqrt(dtype(total) / (num_a * num_b));
    uint each_a = num_a * scaler;
    uint each_b = num_b * scaler;
    if (each_a == 0) {
        each_a = 1;
        scaler = 1 / (scaler * num_a);
        each_b /= scaler;
        if (each_b == 0) {
            each_b = 1;
        }
    }
    if (each_b == 0) {
        each_b = 1;
        scaler = 1 / (scaler * num_b);
        each_a /= scaler;
        if (each_a == 0) {
            each_a = 1;
        }
    }
    *set_each_a = each_a;
    *set_each_b = each_b;
    *set_total_a = ((num_a + each_a - 1) / each_a) * each_a;
    *set_total_b = ((num_b + each_b - 1) / each_b) * each_b;
}

inline uint to_linear(uint i, uint j, uint k, uint l, uint m, uint J, uint K,
                      uint L, uint M) {
    return (((i * J + j) * K + k) * L + l) * M + m;
}
inline uint to_linear(uint i, uint j, uint k, uint l, uint m, uint n, uint J,
                      uint K, uint L, uint M, uint N) {
    return ((((i * J + j) * K + k) * L + l) * M + m) * N + n;
}
inline uint to_linear(uint i, uint j, uint k, uint l, uint J, uint K, uint L) {
    return ((i * J + j) * K + k) * L + l;
}
inline uint to_linear(uint i, uint j, uint k, uint J, uint K) {
    return (i * J + j) * K + k;
}
inline uint to_linear(uint i, uint j, uint J) { return i * J + j; }

template <typename dtype>
inline uint output_size_for_2d(const uint input, const uint kernel,
                               const uint padding,
                               const std::optional<uint> dilation,
                               const uint stride, const bool ceil_mode) {
    const uint top =
        input + (2 * padding) - (dilation.value_or(1) * (kernel - 1)) - 1;
    const uint bot = stride;

    const uint poss =
        (ceil_mode ? static_cast<uint>(std::ceil(static_cast<dtype>(top) / bot))
                   : top / bot) +
        1;

    if (!ceil_mode) {
        return poss;
    }

    return (poss - 1) * stride >= input + padding ? poss - 1 : poss;
}

namespace errs {
template <typename... Args> [[noreturn]] void bail(Args... args) {
    std::stringstream ss;
    (ss << ... << args);
    throw std::runtime_error(ss.str());
}

[[noreturn]] inline void no_user_def(const std::string &name) {
    bail("trying to use custom ", name, " when no implementation exists");
}

[[noreturn]] inline void invalid_algo(const std::string &op,
                                      const std::string &algo) {
    bail("invalid ", op, " algorithm: ", algo);
}

template <typename... Args> void bail_if(bool check, Args... args) {
    if (check) {
        bail(args...);
    }
}

} // namespace errs

namespace input_dims {
const int LINEAR = 1;
const int POOL2D = 3;
const int CONV2D = 3;
const int ACTIVATION = -1;
const int FLATTEN = -1;
}; // namespace input_dims
