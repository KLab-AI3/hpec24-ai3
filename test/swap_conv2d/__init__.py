import torch
from typing import Sequence
import ai3
from test import BATCH, compare_tensors


def runner(module: torch.nn.Module, input_sample_shape: Sequence[int], name: str):
    input_data = torch.randn(BATCH, *input_sample_shape)
    with torch.inference_mode():
        target = module(input_data)
        for algo in ['default', 'direct', 'smm']:
            ai3.swap_conv2d(module, algo)
            output = module(input_data)
            compare_tensors(
                output, target, f"{name} swap conv2d with {algo}, {BATCH} samples")


def run():
    print("SWAP CONV2D")

    from . import simple_created
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
    from . import swin_transformer
    from . import vgg16
    from . import vision_transformer
