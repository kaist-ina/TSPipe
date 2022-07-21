"""Brought from TorchGPipe implementation.
Manipulation of micro-batches."""

import typing
from dataclasses import dataclass
from threading import Condition
from typing import (Callable, Dict, Iterable, Iterator, List, Optional, Tuple,
                    Union, cast)

import torch
import torch.cuda.comm
from torch import Tensor

from tspipe.utils import traverse_tensor_map

__all__: List[str] = []


Tensors = Union[Tuple[Tensor, ...], Dict]
TensorOrTensors = Union[Tensor, Tensors]
Function = Callable[[TensorOrTensors], TensorOrTensors]


class Batch:
    """An abstraction of an atomic tensor or a tuple of tensors. This
    eliminates every boilerplate code to classify an atomic tensor or a tuple
    of tensors.
    ::

        x = generate_tensor_or_tensors()
        x = Batch(x)

        # in-place update
        x[0] = F.apply(x[0])
        x[:] = F.apply(*x)

        # f(x) if x is a tensor.
        # f(*x) if x is a tuple of tensors.
        # y is also a batch.
        y = x.call(f)

    """

    def __init__(self, value: TensorOrTensors) -> None:
        self.value = value
        self.atomic = torch.is_tensor(value)
        self.debug_info = ""

    def to_(self, device: torch.device) -> 'Batch':
        self.value = traverse_tensor_map(self.value, lambda t: t.to(device=device))
        return self

    def clone(self):
        return Batch(traverse_tensor_map(self.value, lambda x: x.clone()))

    def detach(self):
        return Batch(traverse_tensor_map(self.value, lambda x: x.detach()))

    def share_memory_(self) -> 'Batch':
        self.value.share_memory_()
        return self

    def pin_memory(self) -> 'Batch':
        return Batch(self.value.pin_memory())

    @property
    def tensor(self) -> Tensor:
        """Retrieves the underlying tensor."""
        if not self.atomic:
            raise AttributeError('not atomic batch')
        return cast(Tensor, self.value)

    @property
    def tensors(self) -> Tensors:
        """Retrieves the underlying tensors."""
        if self.atomic:
            raise AttributeError('batch is atomic')
        return cast(Tensors, self.value)

    @property
    def tensor_or_tensors(self) -> TensorOrTensors:
        """Retrieves the underlying tensor or tensors regardless of type."""
        return self.value

    def call(self, function: Function) -> 'Batch':
        """Calls a function by the underlying tensor or tensors. It also wraps
        the output with :class:`Batch`.
        """
        return Batch(function(self.value))

    def __repr__(self) -> str:
        return f'Batch[atomic={self.atomic!r}]({self.value!r})'

    def __iter__(self) -> Iterator[Tensor]:
        if self.atomic:
            yield self.tensor
        else:
            yield from self.tensors

    def __len__(self) -> int:
        return 1 if self.atomic else len(self.tensors)

    def __getitem__(self, index: int) -> Tensor:
        if not self.atomic:
            return self.tensors[index]

        if index != 0:
            raise IndexError('atomic batch allows index 0 only')

        return self.tensor

    # NOTE(sublee): pyflakes can't detect "overload" instead of "typing.overload".
    @typing.overload
    def __setitem__(self, index: int, value: Tensor) -> None: ...

    @typing.overload
    def __setitem__(self, index: slice, value: Tensors) -> None: ...

    def __setitem__(self, index: Union[int, slice], value: TensorOrTensors) -> None:
        if isinstance(index, int):
            value = cast(Tensor, value)
            self._setitem_by_index(index, value)
        else:
            value = cast(Tensors, value)
            self._setitem_by_slice(index, value)

    def _setitem_by_index(self, index: int, value: Tensor) -> None:
        if not self.atomic:
            i = index
            self.value = self.value[:i] + (value,) + self.value[i+1:]
            return

        if index != 0:
            raise IndexError('atomic batch allows index 0 only')

        self.value = value

    def _setitem_by_slice(self, index: slice, value: Tensors) -> None:
        if not (index.start is index.stop is index.step is None):
            raise NotImplementedError('only slice [:] supported')

        if not self.atomic:
            self.value = value
            return

        if len(value) != 1:
            raise IndexError('atomic batch cannot be replaced with multiple tensors')

        self.value = value[0]


def check(input: TensorOrTensors) -> None:
    """Checks whether the input is a tensor or tensors.

    Raises:
        TypeError: input is not a tensor or tensors.

    """
    if isinstance(input, tuple):
        for x in input:
            check(x)
        return

    if not isinstance(input, Tensor):
        raise TypeError(f'expected Tensor, but got {input.__class__.__name__}')


def scatter(input: TensorOrTensors, chunks: int) -> List[Batch]:
    """Splits an input mini-batch into multiple micro-batches."""
    inputs: Iterable[TensorOrTensors]

    if isinstance(input, Tensor):
        inputs = input.chunk(chunks)
    else:
        rotated: List[Tensors] = []

        for tensor in input:
            tensors = tensor.chunk(chunks)
            rotated.append(cast(Tensors, tensors))

        inputs = zip(*rotated)

    return [Batch(x) for x in inputs]


def gather(outputs: List[Batch]) -> TensorOrTensors:
    """Concatenates output micro-batches into a mini-batch."""
    output: TensorOrTensors

    if outputs[0].atomic:
        tensors = tuple(b.tensor for b in outputs)
        output = torch.cat(tensors)
    else:
        rotated = [b.tensors for b in outputs]
        output_buf = []

        for tensors in zip(*rotated):
            output_buf.append(torch.cat(tensors))

        output = tuple(output_buf)

    # assert torch.isfinite(output).any()
    # assert not torch.isnan(output).any()

    return output


@dataclass
class ScatterGatherFn():
    scatter: Callable[[TensorOrTensors, int], List[Batch]]
    gather: Callable[[List[Batch]], TensorOrTensors]
    batch_size: Callable[[TensorOrTensors], int]


def batch_size(t: TensorOrTensors):
    if torch.is_tensor(t):
        return t.shape[0]
    raise NotImplementedError()


defaultScatterGatherFn = ScatterGatherFn(scatter=scatter, gather=gather, batch_size=batch_size)


class BatchList:
    def __init__(self, batch_id: int, *batches: Union[Batch, Iterable[Optional['Microbatch']]],
                 grad: Optional[Union[Dict[str, Tensor], Tensor]] = None):
        """batch_online_view_1, batch_target_view_2 , batch_online_view_2 , batch_target_view_1"""
        self.batch_id = batch_id
        self.batches: List[Batch] = batches
        self.grad: Optional[Union[Dict[str, Tensor], Tensor]] = grad
        # assert is_picklable(self)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.__dict__}>"


class ModelParameter:
    def __init__(self, batch_id: int, parameters: Optional[Iterable[Tensor]]):
        self.batch_id = batch_id
        self.parameters = parameters


class Microbatch:
    def __init__(self, batch_id: int, view_id: int, ubatch_id: int, is_target: bool, batch: Optional[Batch]):
        self.batch_id = batch_id
        self.view_id = view_id
        self.ubatch_id = ubatch_id
        self.is_target = is_target
        self.data = batch

    def __repr__(self):
        return f"Microbatch<batch_id={self.batch_id} view_id={self.view_id} ubatch_id={self.ubatch_id} " + \
                f"is_target={self.is_target} size={self.data.tensor.shape if self.data.atomic else []}>"


class BatchQueue():
    def __init__(self):
        self.next_id = 1
        self.highest_ready_batch_id = 0
        self.cv = Condition()
        self.running = True
        self.epoch_boundaries = []

    def get_new_batch_id(self) -> int:
        with self.cv:
            r = self.next_id
            self.highest_ready_batch_id = max(r - 1, self.highest_ready_batch_id)
            self.next_id += 1
            # print(f"Got new batch {r}")
            self.cv.notify()
        return r

    def report_epoch_boundary(self):
        self.epoch_boundaries.append(self.next_id - 1)
        self.highest_ready_batch_id = self.next_id - 1
        print(self.epoch_boundaries)

    def stop(self):
        self.cv.acquire()
        self.running = False
        self.cv.notify()
        self.cv.release()

    def wait_batch(self, batch_id: int):
        if batch_id <= 0:
            return

        self.cv.acquire()
        while True:
            if batch_id <= self.highest_ready_batch_id:
                break
            elif not self.running:
                break
            else:
                self.cv.wait()
                continue
        self.cv.release()
