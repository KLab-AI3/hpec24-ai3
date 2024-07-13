import torch
import torch.nn.functional as F
from ai3 import Model
from ai3.layers import ReLU
from test import compare_tensors


def test(*, input_shape,
         test_name: str) -> None:
    input = torch.randn(input_shape, dtype=torch.float32)
    model = Model(input.dtype, [ReLU(input.dtype, "default")])
    ai3_output = model.predict(input)
    torch_output = F.relu(input)
    compare_tensors(ai3_output, torch_output, test_name)


print('RELU')
test(input_shape=1,
     test_name='one')
test(input_shape=(1, 4, 56, 48),
     test_name='normal')
shape = (3, 1, 4, 5, 6, 8, 1, 1, 8, 4)
test(input_shape=shape,
     test_name=f'{len(shape)} dim')
