import torch
import torch.nn.functional as F
from ai3 import Model
from ai3.layers import Linear
from test import compare_tensors


def test(*, num_samples, in_features: int, out_features: int,
         with_bias: bool = False,
         test_name: str) -> None:
    if num_samples:
        input = torch.randn((num_samples, in_features), dtype=torch.float32)
    else:
        input = torch.randn(in_features, dtype=torch.float32)
    weight = torch.randn((out_features, in_features), dtype=torch.float32)
    if with_bias:
        bias = torch.randn(out_features)
    else:
        bias = None

    model = Model(input.dtype, [Linear(input.dtype, weight, bias, "default")])
    ai3_output = model.predict(input)
    torch_output = F.linear(input, weight, bias=bias)
    compare_tensors(ai3_output, torch_output, test_name)


print('LINEAR')
test(num_samples=None,
     in_features=2,
     out_features=2,
     test_name='square')
test(num_samples=None,
     in_features=4,
     out_features=4,
     with_bias=True,
     test_name='square bias')
test(num_samples=None,
     in_features=100,
     out_features=5,
     test_name='in > out')
test(num_samples=None,
     in_features=5,
     out_features=100,
     test_name='out > in')
test(num_samples=None,
     in_features=40,
     out_features=30,
     with_bias=True,
     test_name='10s with bias')
test(num_samples=None,
     in_features=348,
     out_features=498,
     with_bias=True,
     test_name='100s with bias')
test(num_samples=5,
     in_features=348,
     out_features=498,
     with_bias=True,
     test_name='100s with bias multiple samples')
