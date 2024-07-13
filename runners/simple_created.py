from example.simple_conv2d import ConvNet
from runners import wrapped_run


def run_on(runner):
    wrapped_run(ConvNet(), (3, 224, 224), runner)
