import torch
import ai3
from bench import predict_show_time
from test import compare_tensors, BATCH
from typing import Sequence


def run():
    print('SWAPPING CONV2D')
    from . import (simple_created, vgg16, alexnet, convnext, densenet, efficientnet, googlenet,
                   inception, shufflenetv2, mobilenet, mnasnet, squeezenet, vision_transformer,
                   swin_transformer, maxvit, regnet, resnet)


def runner(module: torch.nn.Module, input_sample_shape: Sequence[int], name: str):
    input_data = torch.randn(BATCH, *input_sample_shape)
    torch_out = predict_show_time(module, input_data, name + " torch")

    for algo in ['default', 'direct', 'smm']:
        ai3.swap_conv2d(module, algo)
        ai3_out = predict_show_time(module, input_data, f"{name} ai3 {algo}")
        compare_tensors(ai3_out, torch_out,
                        f"{name} ai3 {algo}", print_pass=False)
