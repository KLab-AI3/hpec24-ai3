from typing import Optional
import torch
import numpy as np
import atexit
import torch

FAILED_TESTS = []
BATCH = 2


def show_failed():
    if FAILED_TESTS:
        print(f"Failed {len(FAILED_TESTS)} tests:")
        for test in FAILED_TESTS:
            print(f"  - {test}")


atexit.register(show_failed)


def add_fail(mes):
    FAILED_TESTS.append(f"{mes}")


def compare_tensors(out_tensor, tar_tensor: torch.Tensor, mes: Optional[str] = None, atol=1e-4, print_pass=True) -> None:
    if isinstance(out_tensor, np.ndarray):
        out = out_tensor
    else:
        out = out_tensor.numpy()
    tar = np.array(tar_tensor)
    assert (isinstance(out, np.ndarray))

    if np.isnan(tar).any():
        add_fail(mes)
        print(f'Failed Test `{mes}`, target has NaNs')
        return

    if np.isnan(out).any():
        add_fail(mes)
        print(f'Failed Test `{mes}`, output has NaNs')
        return
    if tar.shape != out.shape:
        add_fail(mes)
        print(
            f'Failed Test `{mes}`, Tensors have different shapes, target: {tar.shape} and output {out.shape}')
        return

    different_elements = np.where(np.abs(out - tar) > atol)

    if len(different_elements[0]) == 0:
        if mes and print_pass:
            print(f'  Passed Test {mes}')
    else:
        add_fail(mes)
        print(
            f'Failed Test {mes}, {len(different_elements[0])} different elements out of {out.size}')
        print('  Tensors differ at the following indices:')
        for index in zip(*different_elements):
            index = tuple(map(int, index))
            print('  at:', index, 'target:', tar[index], 'output:', out[index])
