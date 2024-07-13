import torch
from bench import predict_show_time
from torch import nn
import ai3
from test import compare_tensors

N = 100


class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding)

    def forward(self, x):
        x = self.conv(x)
        return x


def run_on(input):
    orig = Conv2D(input.shape[1], input.shape[1], (3, 3))
    orig_out = predict_show_time(orig, input, "pytorch")
    assert (isinstance(orig_out, torch.Tensor))

    optim = ai3.swap_backend(orig, {"conv2d": "direct"})
    direct_out = predict_show_time(optim, input, "ai3 direct")

    optim = ai3.swap_backend(orig, {"conv2d": "smm"})
    smm_out = predict_show_time(optim, input, "ai3 smm")

    compare_tensors(direct_out, orig_out.detach().numpy(),
                    "direct", print_pass=False)
    compare_tensors(smm_out, orig_out.detach().numpy(),
                    "smm", print_pass=False)


print("Conv2D")
input = torch.randn(N, 3, 224, 224)
run_on(input)
input = torch.randn(N, 512, 14, 14)
run_on(input)
