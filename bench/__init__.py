import time
import ai3
import torch


USE_TORCH_COMPILE = False


def warm_up(runner, data):
    data_sample = data[0]
    data_shape = (1,) + data_sample.size()
    runner(data_sample.view(data_shape))


def predict_show_time(runner, data, runner_name: str, recur: bool = True):
    out = None
    start_time = -1
    if isinstance(runner, torch.nn.Module):
        warm_up(runner, data)
        with torch.inference_mode():
            start_time = time.time()
            out = runner(data)
    elif isinstance(runner, ai3.Model):
        warm_up(runner, data)
        start_time = time.time()
        out = runner.predict(data, out_type=torch.Tensor)
    else:
        print(f"invalid runner f{type(runner)}")
        assert (False)
    end_time = time.time()
    assert (start_time > 0)
    inference_time = end_time - start_time
    print(f"  Time {runner_name}: {inference_time} seconds")

    if USE_TORCH_COMPILE and isinstance(runner, torch.nn.Module) and recur:
        predict_show_time(torch.compile(runner), data,
                          runner_name + " compiled", recur=False)
    return out
