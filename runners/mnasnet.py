import torchvision
from runners import wrapped_run


def run_on(runner):
    wrapped_run(torchvision.models.mnasnet1_0(
        weights=torchvision.models.MNASNet1_0_Weights.DEFAULT),
        (3, 224, 224),
        runner)
