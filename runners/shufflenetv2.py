import torchvision
from runners import wrapped_run


def run_on(runner):
    wrapped_run(torchvision.models.shufflenet_v2_x2_0(),
                (3, 224, 224), runner)
