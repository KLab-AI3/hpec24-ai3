from typing import Sequence
import inspect
import torch
import os

GROUPED_CONVOLUTION = False


def wrapped_run(module: torch.nn.Module, input_sample_shape: Sequence[int], runner):
    name, _ = os.path.splitext(os.path.basename(caller_name()))
    name = name.upper()
    print(f"{name}")
    module.eval()
    (needs_groups, has_conv) = check_mod(module)
    if needs_groups and not GROUPED_CONVOLUTION:
        print(f"  skipping {name} as it requires groups > 1")
    elif has_conv:
        runner(module, input_sample_shape, name)
    else:
        print(f"{name} doesn't use convolution")


def caller_name():
    caller_frame = inspect.stack()[2]
    name = caller_frame.filename
    return name


def check_mod(module: torch.nn.Module):
    found_conv2d = False
    grouped_conv = False
    for submodule in module.modules():
        if isinstance(submodule, torch.nn.Conv2d):
            found_conv2d = True
            if submodule.groups > 1:
                grouped_conv = True

    return grouped_conv, found_conv2d
