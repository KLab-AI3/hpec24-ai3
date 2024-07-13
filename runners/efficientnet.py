import torchvision
from runners import wrapped_run


def run_on(runner):
    wrapped_run(torchvision.models.efficientnet_v2_s(
        weights=torchvision.models.EfficientNet_V2_S_Weights.DEFAULT), (3, 224, 224), runner)
