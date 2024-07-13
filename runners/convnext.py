import torchvision
from runners import wrapped_run


def run_on(runner):
    wrapped_run(torchvision.models.convnext_base(weights='DEFAULT'),
                (3, 224, 224), runner)
