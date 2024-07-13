import torchvision
from runners import wrapped_run


def run_on(runner):
    wrapped_run(torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT),
                (3, 224, 224), runner)
