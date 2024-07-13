from typing import (
    Union,
    Sequence,
    Optional,
)
from abc import ABC
from ai3 import core, errors, utils
from ai3.tensor import Tensor


class Layer(ABC):
    def __init__(self, core, algorithm):
        self.core = core
        self.algorithm = algorithm
        ...


class Conv2D(Layer):
    def __init__(self, dtype, weight, bias, stride: Union[int, Sequence[int]],
                 padding: Union[str, Union[int, Sequence[int]]], dilation:
                 Union[int, Sequence[int]], padding_mode: str, groups: int,
                 algorithm: str):
        stride = utils.make_2d(stride)
        dilation = utils.make_2d(dilation)
        padding = utils.make_padding_2d(
            padding, stride, dilation, weight.size())
        errors.bail_if(padding_mode not in [
                       'zeros', 'reflect', 'replicate', 'circular'], f"invalid padding mode: {padding_mode}")
        pad_mode = {
            'zeros': core.PaddingMode.zeros,
            'reflect': core.PaddingMode.reflect,
            'replicate': core.PaddingMode.replicate,
            'circular': core.PaddingMode.circular
        }[padding_mode]

        weight_addr = utils.get_address(weight)
        weight_shape = utils.get_shape(weight)
        if bias is not None:
            bias_addr = utils.get_address(bias)
        else:
            bias_addr = None
        (layer, self.typestr) = utils.get_item_and_type(dtype, core.Conv2D_float,
                                                        core.Conv2D_double)
        self.core = layer(weight_addr,
                          weight_shape,
                          bias_addr,
                          padding,
                          stride,
                          dilation,
                          pad_mode,
                          groups,
                          algorithm)

    def set_algo(self, algo: str):
        self.core.algorithm = algo

    def forward(self, input) -> Tensor:
        return Tensor(self.core.forward(utils.get_address(input), utils.get_shape(input)))


class Linear(Layer):
    def __init__(self, dtype, weight, bias, algorithm: str):
        weight_addr = utils.get_address(weight)
        weight_shape = utils.get_shape(weight)
        if bias is not None:
            bias_addr = utils.get_address(bias)
        else:
            bias_addr = None
        (layer, self.typestr) = utils.get_item_and_type(
            dtype, core.Linear_float, core.Linear_double)
        self.core = layer(weight_addr, weight_shape, bias_addr, algorithm)


class ReLU(Layer):
    def __init__(self, dtype, algorithm: str):
        (layer, self.typestr) = utils.get_item_and_type(
            dtype, core.ReLU_float, core.ReLU_double)
        self.core = layer(algorithm)


class MaxPool2D(Layer):
    def __init__(self, dtype, kernel_shape: Union[int, Sequence[int]],
                 stride: Union[int, Sequence[int]],
                 padding: Union[int, Sequence[int]],
                 dilation: Union[int, Sequence[int]],
                 ceil_mode: bool,
                 algorithm: str):
        stride = utils.make_2d(stride)
        dilation = utils.make_2d(dilation)
        kernel_shape = utils.make_2d(kernel_shape)
        padding = utils.make_2d(padding)
        ceil_mode = ceil_mode

        (layer, self.typestr) = utils.get_item_and_type(
            dtype, core.MaxPool2D_float, core.MaxPool2D_double)
        self.core = layer(kernel_shape,
                          padding, stride, dilation, ceil_mode, algorithm)


class AvgPool2D(Layer):
    def __init__(self, dtype, kernel_shape: Union[int, Sequence[int]],
                 stride: Union[int, Sequence[int]],
                 padding: Union[int, Sequence[int]],
                 ceil_mode: bool,
                 count_include_pad: bool,
                 divisor_override: Optional[int],
                 algorithm: str):
        stride = utils.make_2d(stride)
        kernel_shape = utils.make_2d(kernel_shape)
        padding = utils.make_2d(padding)

        (layer, self.typestr) = utils.get_item_and_type(
            dtype, core.AvgPool2D_float, core.AvgPool2D_double)
        self.core = layer(kernel_shape,
                          padding, stride, ceil_mode, count_include_pad, divisor_override, algorithm)


class AdaptiveAvgPool2D(Layer):
    def __init__(self, dtype, output_shape: Optional[Union[int, Sequence[Optional[int]]]], algorithm: str):
        if isinstance(output_shape, list):
            assert (len(output_shape) == 2)
        if isinstance(output_shape, int):
            output_shape = utils.make_2d(output_shape)
        (layer, self.typestr) = utils.get_item_and_type(
            dtype, core.AdaptiveAvgPool2D_float, core.AdaptiveAvgPool2D_double)
        self.core = layer(output_shape, algorithm)


class Flatten(Layer):
    def __init__(self, dtype, start_dim: int, end_dim: int, algorithm: str):
        (layer, self.typestr) = utils.get_item_and_type(
            dtype, core.Flatten_float, core.Flatten_double)
        self.core = layer(start_dim, end_dim, algorithm)
