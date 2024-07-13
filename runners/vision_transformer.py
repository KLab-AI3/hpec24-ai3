import torchvision
from runners import wrapped_run


def run_on(runner):
    wrapped_run(torchvision.models.vit_b_16(),
                (3, 224, 224), runner)
