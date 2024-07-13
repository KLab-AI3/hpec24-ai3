import torch
from ai3 import core, utils, errors
from typing import Union


class Tensor():
    def __init__(self, tens: Union[core.Tensor_float, core.Tensor_double]):
        self.core = tens
        if isinstance(tens, core.Tensor_float):
            self.typestr = utils.FLOAT32_STR
        else:
            self.typestr = utils.FLOAT64_STR

    def to(self, out_type):
        if out_type is torch.Tensor:
            return self.torch()
        elif out_type is None:
            return self
        errors.bail(f"unsupported type to transfer tensor to {out_type}")

    def numpy(self):
        import numpy
        dtype = {
            utils.FLOAT32_STR: numpy.float32,
            utils.FLOAT64_STR: numpy.float64
        }[self.typestr]
        errors.bail_if(dtype is None,
                       f"type, {self.typestr} is neither float32 or float64")
        data = numpy.frombuffer(self.core, dtype=dtype)
        return data.reshape(self.core.shape)

    def torch(self):
        return torch.frombuffer(self.core,
                                dtype=torch.__dict__[self.typestr]).view(self.core.shape)
