import torchvision
from runners import wrapped_run


def run_on(runner):
    wrapped_run(torchvision.models.googlenet(
        weights=torchvision.models.GoogLeNet_Weights.DEFAULT),
        (3, 224, 224), runner)
