from ai3 import layers, errors
from typing import Mapping, Optional, List, Sequence, Union, Callable
from collections import defaultdict
import torch
from torch import nn, fx


def mod_to_op(mod: nn.Module) -> str:
    if isinstance(mod, (nn.Conv2d, Conv2D)):
        return "conv2d"
    elif isinstance(mod, nn.Linear):
        return "linear"
    elif isinstance(mod, nn.MaxPool2d):
        return "maxpool2d"
    elif isinstance(mod, nn.AvgPool2d):
        return "avgpool2d"
    elif isinstance(mod, nn.AdaptiveAvgPool2d):
        return 'adaptiveavgpool2d'
    elif isinstance(mod, nn.ReLU):
        return 'relu'
    elif isinstance(mod, nn.Flatten):
        return 'flatten'
    errors.unsupported(mod)


def iscontainer(name: str) -> bool:
    return '.' in name


def getmodule(module: nn.Module, name) -> nn.Module:
    if iscontainer(name):
        names = name.split('.', 1)
        return getmodule(getattr(module, names[0]), names[1])
    else:
        return getattr(module, name)


def setmodule(module: nn.Module, name, new: nn. Module) -> nn.Module:
    if iscontainer(name):
        names = name.split('.', 1)
        setmodule(getattr(module, names[0]), names[1], new)
    else:
        setattr(module, name, new)
    return module


class Conv2D(nn.Module):
    def __init__(self, internal: layers.Conv2D, target: str):
        super(Conv2D, self).__init__()
        self.internal = internal
        self.target = target

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            return self.internal.forward(x).torch()
        elif isinstance(x, fx.proxy.Proxy):
            tracer = x.tracer
            node = tracer.create_node('call_module', self.target, ('', ), {})
            return fx.proxy.Proxy(node, tracer)


def get_algo_inc_counter(orig: nn.Module,  algos: Mapping[str, Union[str, Sequence[str], Callable]], layer_counters: defaultdict[str, int]) -> str:
    op = mod_to_op(orig)
    if callable(algos[op]):
        algo = algos[op](orig)
    elif isinstance(algos[op], list):
        algo = algos[op][layer_counters[op]]
        layer_counters[op] += 1
    else:
        algo = algos[op]
    return algo


def get_swapped_backend_layers(complete_module: nn.Module, dtype, algos: Mapping[str, Union[str, Sequence[str], Callable]]) -> List[layers.Layer]:
    gm: fx.GraphModule = fx.symbolic_trace(complete_module)
    forwards = []

    layer_counters = defaultdict(int)

    for node in gm.graph.nodes:
        if node.op == 'placeholder' or node.op == 'output':
            pass
        elif node.op == 'call_function':
            if node.target == torch.flatten:
                start_dim = 0
                end_dim = -1
                if len(node.args) > 1:
                    start_dim = node.args[1]
                if len(node.args) > 2:
                    end_dim = node.args[2]
                if 'start_dim' in node.kwargs:
                    start_dim = node.kwargs['start_dim']
                if 'end_dim' in node.kwargs:
                    end_dim = node.kwargs['end_dim']
                algo = get_algo_inc_counter(
                    torch.nn.Flatten(), algos, layer_counters)
                errors.bail_if(algo == "torch",
                               "can't use torch backend when in swap_backend")
                forwards.append(layers.Flatten(
                    dtype, start_dim, end_dim, algo))
            elif node.target == torch.relu:
                algo = get_algo_inc_counter(
                    torch.nn.ReLU(), algos, layer_counters)
                errors.bail_if(algo == "torch",
                               "can't use torch backend when in swap_backend")
                forwards.append(layers.ReLU(dtype, algo))
            else:
                errors.unsupported(node.target)
        elif node.op == 'call_module':
            mod = getmodule(complete_module, node.target)
            if not isinstance(mod, nn.Dropout):
                algo = get_algo_inc_counter(mod, algos, layer_counters)
                errors.bail_if(algo == "torch",
                               "can't use torch backend when in swap_backend")
                swapped = swap_layer(mod, dtype, algo)
                if not swapped:
                    errors.bail(f"unsupported module: {mod}")
                forwards.append(swapped)
        else:
            errors.bail(f"unsupported call: {node.op}")

    return forwards


def swap_conv2d(complete_module: nn.Module, dtype, selector: Union[str, Sequence[str], Callable]):
    gm: fx.GraphModule = fx.symbolic_trace(complete_module)
    layer_counters = defaultdict(int)
    for node in gm.graph.nodes:
        if node.op == 'call_module':
            mod = getmodule(complete_module, node.target)
            if isinstance(mod, (nn.Conv2d, Conv2D)):
                algo = get_algo_inc_counter(
                    mod, {'conv2d': selector}, layer_counters)
                if algo == "torch":
                    continue
                if isinstance(mod, nn.Conv2d):
                    swapped = swap_layer(mod, dtype, algo)
                    assert isinstance(swapped, layers.Conv2D)
                    complete_module = setmodule(
                        complete_module, node.target, Conv2D(swapped, str(node.target)))
                else:
                    mod.internal.set_algo(algo)


def swap_layer(module: nn.Module, dtype, algo: str) -> Optional[layers.Layer]:
    if isinstance(module, nn.Conv2d):
        return layers.Conv2D(dtype, module.weight, module.bias, module.stride,
                             module.padding, module.dilation, module.padding_mode,
                             module.groups, algo)
    elif isinstance(module, nn.Linear):
        return layers.Linear(dtype, module.weight, module.bias, algo)
    elif isinstance(module, nn.MaxPool2d):
        return layers.MaxPool2D(dtype, module.kernel_size, module.stride,
                                module.padding, module.dilation, module.ceil_mode, algo)
    elif isinstance(module, nn.AvgPool2d):
        return layers.AvgPool2D(dtype, module.kernel_size, module.stride,
                                module.padding, module.ceil_mode,
                                module.count_include_pad,
                                module.divisor_override, algo)
    elif isinstance(module, nn.AdaptiveAvgPool2d):
        return layers.AdaptiveAvgPool2D(dtype, module.output_size, algo)
    elif isinstance(module, nn.ReLU):
        return layers.ReLU(dtype, algo)
    elif isinstance(module, nn.Flatten):
        return layers.Flatten(dtype, module.start_dim, module.end_dim, algo)
    return None
