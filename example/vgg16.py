import torch
import torchvision
import ai3


def conv2d_selector(orig: torch.nn.Conv2d) -> str:
    in_channels = orig.weight.shape[1]
    if in_channels > 200:
        return "smm"
    return "direct"


input_data = torch.randn(1, 3, 224, 224)
vgg16 = torchvision.models.vgg16(
    weights=torchvision.models.VGG16_Weights.DEFAULT)
vgg16.eval()
torch_out = vgg16(input_data)

model: ai3.Model = ai3.swap_backend(vgg16, {"conv2d": conv2d_selector})
ai3_out = model(input_data)
assert torch.allclose(torch_out, ai3_out, atol=1e-6)

ai3.swap_conv2d(vgg16, conv2d_selector)
swapped_out = vgg16(input_data)
assert torch.allclose(torch_out, swapped_out, atol=1e-6)
