import torch
import torch.nn.functional as F
from ai3 import Model
from ai3.layers import Conv2D
from test import compare_tensors
from typing import Union, Sequence


def test(*, num_samples=None, input_channels: int, in_height: int, in_width: int,
         output_channels: int, kernel_height: int, kernel_width: int,
         with_bias: bool = False,
         padding: Union[str, Union[int, Sequence[int]]] = 1,
         dilation: Union[int, Sequence[int]] = 1,
         stride: Union[int, Sequence[int]] = 1,
         groups: int = 1,
         test_name: str) -> None:
    if num_samples:
        input = torch.randn(num_samples, input_channels, in_height,
                            in_width, dtype=torch.float64)
    else:
        input = torch.randn(input_channels, in_height,
                            in_width, dtype=torch.float64)
    kernel = torch.randn(output_channels, input_channels // groups,
                         kernel_height, kernel_width, dtype=torch.float64)
    if with_bias:
        bias = torch.randn(output_channels, dtype=torch.float64)
    else:
        bias = None

    torch_output = F.conv2d(input, kernel, bias=bias, dilation=dilation,
                            padding=padding, stride=stride, groups=groups)
    algos = ['default', 'direct', 'smm']
    if torch.backends.cudnn.is_available():
        algos.extend(["implicit precomp gemm",
                     "implicit gemm", "gemm", "guess"])

    for algo in algos:
        model = Model(input.dtype, [Conv2D(input.dtype, kernel, bias,
                                           stride, padding, dilation, 'zeros', 1, algo)])
        out = model.predict(input, out_type=torch.Tensor)
        compare_tensors(out, torch_output, test_name + f' {algo}')


print('CONV2D')

test(input_channels=1,
     in_height=100,
     in_width=150,
     output_channels=1,
     kernel_height=15,
     kernel_width=12,
     with_bias=True,
     test_name='with bias')

test(input_channels=1,
     in_height=5,
     in_width=5,
     output_channels=1,
     kernel_height=3,
     kernel_width=3,
     test_name='basic no bias')

test(input_channels=4,
     in_height=30,
     in_width=40,
     output_channels=6,
     kernel_height=7,
     kernel_width=5,
     with_bias=True,
     padding='same',
     dilation=(2, 2),
     test_name='same odd kernel')

test(input_channels=4,
     in_height=30,
     in_width=40,
     output_channels=6,
     kernel_height=8,
     kernel_width=4,
     with_bias=True,
     padding='same',
     dilation=(2, 2),
     test_name='same even kernel')

test(input_channels=4,
     in_height=30,
     in_width=40,
     output_channels=6,
     kernel_height=8,
     kernel_width=4,
     with_bias=True,
     padding='valid',
     dilation=(1, 2),
     test_name='valid even kernel')

test(input_channels=4,
     in_height=30,
     in_width=40,
     output_channels=6,
     kernel_height=7,
     kernel_width=5,
     with_bias=True,
     padding='valid',
     dilation=(1, 2),
     test_name='valid odd kernel')

test(input_channels=4,
     in_height=30,
     in_width=40,
     output_channels=6,
     kernel_height=7,
     kernel_width=5,
     with_bias=True,
     dilation=(1, 2),
     test_name='2d dilation')

test(input_channels=4,
     in_height=30,
     in_width=40,
     output_channels=6,
     kernel_height=7,
     kernel_width=5,
     with_bias=True,
     dilation=3,
     test_name='1d dilation')

test(input_channels=4,
     in_height=30,
     in_width=40,
     output_channels=6,
     kernel_height=7,
     kernel_width=5,
     with_bias=True,
     padding=5,
     test_name='1d padding')

test(input_channels=4,
     in_height=30,
     in_width=40,
     output_channels=6,
     kernel_height=7,
     kernel_width=5,
     with_bias=True,
     padding=(2, 5),
     test_name='2d padding')

test(input_channels=4,
     in_height=30,
     in_width=40,
     output_channels=6,
     kernel_height=7,
     kernel_width=5,
     with_bias=True,
     stride=2,
     test_name='1d stride')

test(input_channels=4,
     in_height=30,
     in_width=40,
     output_channels=6,
     kernel_height=7,
     kernel_width=5,
     with_bias=True,
     stride=(2, 3),
     test_name='2d stride')

test(input_channels=1,
     in_height=10,
     in_width=15,
     output_channels=1,
     kernel_height=10,
     kernel_width=15,
     with_bias=True,
     test_name='kern.shape = input.shape')

test(input_channels=3,
     in_height=50,
     in_width=150,
     output_channels=4,
     kernel_height=10,
     kernel_width=10,
     test_name='multi channel')

test(input_channels=4,
     in_height=50,
     in_width=150,
     output_channels=6,
     kernel_height=10,
     kernel_width=10,
     with_bias=True,
     test_name='multi channel with bias')

test(input_channels=4,
     in_height=50,
     in_width=150,
     output_channels=6,
     kernel_height=10,
     kernel_width=10,
     with_bias=True,
     test_name='multi channel with bias')

test(num_samples=5,
     input_channels=4,
     in_height=50,
     in_width=150,
     output_channels=6,
     kernel_height=5,
     kernel_width=5,
     with_bias=True,
     test_name='batched multi channel with bias')
