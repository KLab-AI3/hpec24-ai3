import torchvision
from runners import wrapped_run


def run_on(runner):
    wrapped_run(torchvision.models.inception_v3(
        weights=torchvision.models.Inception_V3_Weights.DEFAULT),
        (3, 224, 224), runner)
