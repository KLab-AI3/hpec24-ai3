import torch
from typing import Sequence
import ai3
from ai3.errors import UnsupportedCallableError
from test import BATCH, compare_tensors
from bench import predict_show_time


def runner(module: torch.nn.Module, input_sample_shape: Sequence[int], name: str):
    input_data = torch.randn(BATCH, *input_sample_shape)

    with torch.inference_mode():
        target = predict_show_time(module, input_data, name + " torch")
        try:
            ai3_model = ai3.swap_backend(module)
        except UnsupportedCallableError as e:
            print(f"  {e} so skipping")
            return
        output = predict_show_time(ai3_model, input_data, name + "ai3")
        assert isinstance(target, torch.Tensor)
        compare_tensors(output, target, name + "ai3", print_pass=False)


def run():
    print("SWAP BACKEND")

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
