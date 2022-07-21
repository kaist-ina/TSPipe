
from typing import Iterable

import torch


class SequentialableModel(torch.nn.Module):
    def to_sequential(self) -> torch.nn.Sequential:
        return NotImplementedError(f"to_sequential is not implemented for {self.__class__}.")

    def test_sequential_validity(model_orig: 'SequentialableModel', test_tensor_size: Iterable):
        test_input = torch.rand(test_tensor_size)
        model_seq = model_orig.to_sequential()
        train_state = model_orig.training
        with torch.no_grad():
            model_orig.eval()
            model_seq.eval()

            output_base = model_orig(test_input)
            output_seq = model_seq(test_input)
            if not torch.allclose(output_base, output_seq):
                print("output_base", output_base, output_base.shape)
                print("output_seq ", output_seq, output_seq.shape)
                return False

        model_orig.train(train_state)
        return True


class FlattenWrapper(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._args, self._kwargs = args, kwargs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.flatten(*self._args, **self._kwargs)
