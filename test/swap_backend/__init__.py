import torch
from typing import Sequence
import ai3
from ai3.errors import UnsupportedCallableError
from test import BATCH, compare_tensors


def runner(module: torch.nn.Module, input_sample_shape: Sequence[int], name: str):
    input_data = torch.randn(BATCH, *input_sample_shape)
    with torch.inference_mode():
        target = module(input_data)
        for algo in ['default', 'direct', 'smm']:
            try:
                ai3_model = ai3.swap_backend(module, {"conv2d": algo})
            except UnsupportedCallableError as e:
                print(f"  {e} so skipping")
                return
            output = ai3_model(input_data)
            compare_tensors(
                output, target, f"{name} swap backend, {BATCH} samples")


def run():
    print('SWAP BACKEND')

    from . import alexnet
    from . import convnext
    from . import densenet
    from . import efficientnet
    from . import googlenet
    from . import inception
    from . import maxvit
    from . import mnasnet
    from . import mobilenet
    from . import regnet
    from . import resnet
    from . import shufflenetv2
    from . import simple_created
    from . import swin_transformer
    from . import vgg16
    from . import vision_transformer
