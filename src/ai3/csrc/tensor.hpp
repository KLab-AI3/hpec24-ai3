#pragma once

#include "utils.hpp"
#include <algorithm>
#include <cstring>
#include <optional>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

template <typename dtype> class Tensor {
  public:
    Tensor(const intptr_t data_address, const std::vector<uint> &s,
           bool owned = true)
        : shape(s), owned(owned) {
        if (owned) {
            data = new dtype[_count(s)];
            std::memcpy(data, reinterpret_cast<const dtype *>(data_address),
                        _count(s) * sizeof(dtype));
        } else {
            data = reinterpret_cast<dtype *>(data_address);
        }
    }

    Tensor(const std::vector<uint> &s)
        : data(new dtype[_count(s)]), shape(s), owned(true) {}

    static Tensor<dtype> concat(Tensor<dtype> *tens, uint len) {
        uint each_size = tens[0].count();
        tens[0].shape.insert(tens[0].shape.begin(), len);
        Tensor<dtype> output(tens[0].shape);
        for (uint i = 0; i < len; ++i) {
            std::memcpy(&output.data[i * each_size], tens[i].data,
                        each_size * sizeof(dtype));
        }
        return output;
    }

    static std::optional<Tensor>
    from_optional(const std::optional<intptr_t> &data_address,
                  const std::vector<uint> &s, bool input_data = false) {
        if (data_address.has_value()) {
            return Tensor<dtype>((*data_address), s, input_data);
        } else {
            return std::nullopt;
        }
    }

    Tensor() = default;

    ~Tensor() {
        if (owned) {
            delete[] data;
        }
    };

    Tensor(Tensor &&other) noexcept { *this = std::move(other); }

    Tensor &operator=(Tensor &&other) noexcept {
        if (this != &other) {
            shape = std::move(other.shape);
            data = other.data;
            owned = other.owned;
            other.data = nullptr;
            other.owned = false;
        }
        return *this;
    }

    py::buffer_info buffer() {
        std::vector<uint> stride(shape.size());
        stride[shape.size() - 1] = sizeof(dtype);
        for (int i = shape.size() - 2; i >= 0; --i) {
            stride[i] = stride[i + 1] * shape[i + 1];
        }
        return py::buffer_info(data, sizeof(dtype),
                               py::format_descriptor<dtype>::format(),
                               shape.size(), shape, stride);
    }

    inline bool batched(const int data_dim = -1) const {
        if (data_dim == -1) {
            return false;
        }
        return shape.size() == unsigned(data_dim + 1);
    }
    inline uint batch_size(const uint input_dims) const {
        return shape[shape.size() - 1 - input_dims];
    }
    inline uint out_channels() const { return shape[shape.size() - 4]; }
    inline uint input_channels() const { return shape[shape.size() - 3]; }
    inline uint height() const { return shape[shape.size() - 2]; }
    inline uint width() const { return shape[shape.size() - 1]; }

    inline uint count() const { return _count(shape); }

    Tensor(const Tensor &) = delete;
    Tensor &operator=(const Tensor &) = delete;

    dtype *data;
    std::vector<uint> shape;
    bool owned;

  private:
    static uint _count(const std::vector<uint> &s) {
        if (s.empty()) {
            return 0;
        }
        uint count = 1;
        for (uint v : s) {
            count *= v;
        }
        return count;
    }
};
