import torchvision
from runners import wrapped_run


def run_on(runner):
    wrapped_run(torchvision.models.regnet_y_16gf(),
                (3, 224, 224), runner)
