import torch
from bench import predict_show_time
from torch import nn
import ai3
from test import compare_tensors


class Linear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Linear, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.linear(x)
        return x


print("Linear")
input = torch.randn(1000, 1200)
orig = Linear(1200, 800)
optim = ai3.swap_backend(orig)
orig_out = predict_show_time(orig, input, "pytorch")
assert (isinstance(orig_out, torch.Tensor))
optim_out = predict_show_time(optim, input, "ai3")
compare_tensors(optim_out, orig_out.detach().numpy(), print_pass=False)
