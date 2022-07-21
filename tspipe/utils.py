 
from collections import OrderedDict, defaultdict
from functools import partial
import torch
from contextlib import contextmanager
from typing import Any, Callable, Generator, List, Union, cast
from collections import deque
from typing import Deque, List, Optional, Tuple
from torch import Tensor
import gc

from tspipe.model_wrapper import TSPipeModelOutput

__all__: List[str] = []
Tensors = Tuple[Tensor, ...]


class CPUStreamType:
    pass

# The placeholder on place of streams for the CPU device instead of CUDA.
CPUStream = CPUStreamType()

# It represents both CUDA streams and the CPU stream.
AbstractStream = Union[torch.cuda.Stream, CPUStreamType]

def new_stream(device: torch.device) -> AbstractStream:
    """Creates a new stream for either CPU or CUDA device."""
    if device.type != 'cuda':
        return CPUStream
    return torch.cuda.Stream(device)


def current_stream(device: torch.device) -> AbstractStream:
    """:func:`torch.cuda.current_stream` for either CPU or CUDA device."""
    if device.type != 'cuda':
        return CPUStream
    return torch.cuda.current_stream(device)


def default_stream(device: torch.device) -> AbstractStream:
    """:func:`torch.cuda.default_stream` for either CPU or CUDA device."""
    if device.type != 'cuda':
        return CPUStream
    return torch.cuda.default_stream(device)


def verify_module(module: torch.nn.Sequential) -> None:
    if module is None:
        return
        
    if not isinstance(module, torch.nn.Sequential):
        raise TypeError('module must be nn.Sequential to be partitioned')

    try:
        iter(module)
    except TypeError:
        raise TypeError('module must be iterable to be partitioned')

    named_children = list(module.named_children())
    if len(named_children) != len(module):
        raise ValueError('module with duplicate children is not supported')

    num_parameters = len(list(module.parameters()))
    num_child_parameters = sum(len(list(child.parameters())) for child in module.children())
    if num_parameters != num_child_parameters:
        raise ValueError('module with duplicate parameters in distinct children is not supported')

def default_stream(device: torch.device) -> AbstractStream:
    """:func:`torch.cuda.default_stream` for either CPU or CUDA device."""
    if device.type != 'cuda':
        return CPUStream
    return torch.cuda.default_stream(device)

@contextmanager
def use_device(device: torch.device) -> Generator[None, None, None]:
    """:func:`torch.cuda.device` for either CPU or CUDA device."""
    if device.type != 'cuda':
        yield
        return

    with torch.cuda.device(device):
        yield

@contextmanager
def use_stream(stream: AbstractStream) -> Generator[None, None, None]:
    """:func:`torch.cuda.stream` for either CPU or CUDA stream."""
    if not is_cuda(stream):
        yield
        return

    with torch.cuda.stream(as_cuda(stream)):
        yield

def get_bytes(obj: Any) -> int: 
    def _get_bytes(tensor: torch.Tensor) -> int:
        raw = tensor.nelement() * tensor.element_size()
        byte_align = 512
        return raw + (byte_align - raw) % byte_align
    return traverse_tensor_sum(obj, _get_bytes)

@contextmanager
def measure_gpu_mem(message: str = 'GPU', device: torch.device = torch.device('cuda')) -> Generator[None, None, None]:
    with use_device(device):
        torch.cuda.synchronize(device)
        prev = torch.cuda.memory_allocated(device)
        yield
        torch.cuda.synchronize(device)
        after = torch.cuda.memory_allocated(device)
        print(f"{message} : Used {(after-prev)/1024/1024:.2f} MiB on {device}, current total {after/1024/1024:.2f} MiB")

# still have 2% of errors... to be fixed later
def get_bytes_recursive(t: torch.Tensor) -> int:
    counted_data_ptr = set()
    estimated_x_size = 0

    def get_exact_bytes_fn(fn) -> int:
        if not fn:
            return 0

        tensor_attributes = [attr for attr in dir(fn) if '_saved_' in attr and torch.is_tensor(getattr(fn, attr))]
        result = 0
        # print(tensor_attributes)
        for attr in tensor_attributes:
            # if attr.endswith('result') or attr.endswith('weight') or attr.endswith('running_mean') or attr.endswith('running_var'):
            #     continue
            # or attr.endswith('running_mean') or attr.endswith('running_var'): # or attr.endswith('mat1') or attr.endswith('mat2'):
            #    continue
            tensor: torch.Tensor = getattr(fn, attr)
            if tensor.storage().data_ptr() not in counted_data_ptr:
                counted_data_ptr.add(tensor.storage().data_ptr())
                b1 = get_bytes(tensor)
                
                consider1 = not any(attr.endswith(x) for x in ['running_mean', 'running_var', 'weight'])
                consider2 = consider1

                if consider1:
                    result += b1
                if attr.endswith('self'):
                    nonlocal estimated_x_size
                    estimated_x_size = b1
                    
                b2 = get_exact_bytes_fn(tensor.grad_fn)
                if consider2:
                    result += b2
            else:
                b1 = get_bytes(tensor)
                
        for next_fn, _ in fn.next_functions:
            result += get_exact_bytes_fn(next_fn)

        return result

    result = get_bytes(t)
    counted_data_ptr.add(t.storage().data_ptr())
    result += get_exact_bytes_fn(t.grad_fn)
    result -= estimated_x_size

    return result

def get_shape(obj):
  def _internal_shape(o):
    if isinstance(o, list):
      return list(_internal_shape(c) for c in o)
    if isinstance(o, OrderedDict):
      return OrderedDict((k, _internal_shape(v)) for k, v in o.items())
    if isinstance(o, dict):
      return dict((k, _internal_shape(v)) for k, v in o.items())
    if isinstance(o, tuple):
      return tuple(_internal_shape(c) for c in o)
    if isinstance(o, torch.Tensor):
      return o.shape
    if isinstance(o, zip):
      return tuple(_internal_shape(c) for c in o)
    return o
  return _internal_shape(obj)


def traverse_tensor_map(obj, fn: Callable[[Tensor], Any]):
    def _internal_traverse(o):
        if isinstance(o, list):
            return list(_internal_traverse(c) for c in o)
        if isinstance(o, OrderedDict):
            return OrderedDict((k, _internal_traverse(v)) for k, v in o.items())
        if isinstance(o, dict):
            return dict((k, _internal_traverse(v)) for k, v in o.items())
        if isinstance(o, tuple):
            return tuple(_internal_traverse(c) for c in o)
        if isinstance(o, torch.Tensor):
            return fn(o)
        if isinstance(o, zip):
            return tuple(_internal_traverse(c) for c in o)
        return o

    if isinstance(obj, TSPipeModelOutput):
        return TSPipeModelOutput(
                data=_internal_traverse(obj.data), 
                lst_require_grad_tensor_path=obj.lst_require_grad_tensor_path
            )
    return _internal_traverse(obj)


def traverse_tensor_sum(obj, fn: Callable[[Tensor], Any]) -> int:
    def _internal_traverse(o):
        if isinstance(o, list):
            return sum(_internal_traverse(c) for c in o)
        if isinstance(o, OrderedDict):
            return sum(_internal_traverse(v) for v in o.values())
        if isinstance(o, dict):
            return sum(_internal_traverse(v) for v in o.values())
        if isinstance(o, tuple):
            return sum(_internal_traverse(c) for c in o)
        if isinstance(o, torch.Tensor):
            return fn(o)
        if isinstance(o, zip):
            return sum(_internal_traverse(c) for c in o)
        return 0
    if isinstance(obj, TSPipeModelOutput):
        return _internal_traverse(obj.data)
    return _internal_traverse(obj)




def generic_object_scatter(obj: Any, chunks: int) -> List[Any]:
    def _internal_traverse(obj):
        if isinstance(obj, list):
            return [list(x) for x in zip(*[_internal_traverse(x) for x in obj])]
        if isinstance(obj, tuple):
            return tuple(zip(*[_internal_traverse(x) for x in obj]))
        if isinstance(obj, dict):
            return tuple(dict(zip(*x)) for x in (zip(
                    zip(*[_internal_traverse(x) for x in obj.keys()]),
                    zip(*[_internal_traverse(x) for x in obj.values()])
                )))
        if isinstance(obj, OrderedDict):
            return tuple(OrderedDict(zip(*x)) for x in (zip(
                    zip(*[_internal_traverse(x) for x in obj.keys()]),
                    zip(*[_internal_traverse(x) for x in obj.values()])
                )))
        if isinstance(obj, torch.Tensor):
            post_chunk = obj.chunk(chunks)
            # print(f"Generic scatter: {obj.shape} => {[t.shape for t in post_chunk]}")
            return post_chunk
        return [obj for _ in range(chunks)]
    if isinstance(obj, TSPipeModelOutput):
        return [TSPipeModelOutput(
            data=x, lst_require_grad_tensor_path=obj.lst_require_grad_tensor_path
            ) for x in _internal_traverse(obj.data)]
    return _internal_traverse(obj)

def generic_object_gather(*objs: Any) -> Any:
    '''
    Traverse the object and concat inner tensors.
    The structure of each element inside `objs` must match.
    '''
    def _internal_traverse(*objs) -> Any:
        if len(objs) == 0:
            return tuple()

        o1 = objs[0]

        if any(type(o1) != type(x) for x in objs):
            raise ValueError(f"Type of given objects must match: {[type(x) for x in objs]}")
        if isinstance(o1, list):
            return list(_internal_traverse(*cx) for cx in zip(*objs))
        if isinstance(o1, tuple) or isinstance(o1, zip):
            return tuple(_internal_traverse(*cx) for cx in zip(*objs))
        if isinstance(o1, OrderedDict):
            for o2 in objs:
                if any(o1k != o2k for o1k, o2k in zip(o1.keys(), o2.keys())):
                    raise ValueError(f"Key of given objects must match: {o1} != {o2}")
            return OrderedDict((kvx[0][0], _internal_traverse(*[kv[1] for kv in kvx])) for kvx in zip(*map(lambda x: x.items(), objs)))
        if isinstance(o1, dict):
            for o2 in objs:
                if any(o1k != o2k for o1k, o2k in zip(o1.keys(), o2.keys())):
                    raise ValueError(f"Key of given objects must match: {o1} != {o2}")
            return dict((kvx[0][0], _internal_traverse(*[kv[1] for kv in kvx])) for kvx in zip(*map(lambda x: x.items(), objs)))
        if isinstance(o1, torch.Tensor):
            post_cat = torch.cat(objs)
            # print(f"Generic gather: {[t.shape for t in objs]} => {post_cat.shape}")
            return post_cat
        if any(o1 != x for x in objs):
            raise ValueError(f"Value of given objects must match: {objs}")
        return o1

    if all(isinstance(obj, TSPipeModelOutput) for obj in objs):
        return TSPipeModelOutput(
            data=_internal_traverse(*[obj.data for obj in objs]), 
            lst_require_grad_tensor_path=objs[0].lst_require_grad_tensor_path
        )
    return _internal_traverse(*objs)

def find_path_tensor_requires_grad(obj) -> List[Tuple]:
    tensor_paths: List[Tuple] = []
    def _internal_traverse(o, path):
        if isinstance(o, list) or isinstance(o, tuple) or isinstance(o, zip):
            for i, c in enumerate(o):
                _internal_traverse(c, (*path, i))
        if isinstance(o, dict):
            for k, v in o.items():
                _internal_traverse(v, (*path, k))
        if isinstance(o, torch.Tensor):
            if o.requires_grad:
                tensor_paths.append(path)
    _internal_traverse(obj, tuple())
    # print(tensor_paths)
    return tensor_paths

def traverse_object_tensor_foreach(obj, fn: Callable[[Tensor, Tuple], None]) -> List[Tuple]:
    def _internal_traverse(o, path):
        if isinstance(o, list) or isinstance(o, tuple) or isinstance(o, zip):
            for i, c in enumerate(o):
                _internal_traverse(c, (*path, i))
        if isinstance(o, dict):
            for k, v in o.items():
                _internal_traverse(v, (*path, k))
        if isinstance(o, torch.Tensor):
            fn(o, path)
    _internal_traverse(obj, tuple())

def traverse_path_apply(obj: Any, path: Tuple, fn: Callable[[Tensor], Any]):
    for subpath in path:
        obj = obj[subpath]
    return fn(obj)

def get_norm(t: torch.Tensor):
  if sum(t.shape) != 1:
    return t.double().norm().cpu().item()
  else:
    return t.cpu().item()

def is_cuda(stream: AbstractStream) -> bool:
    """Returns ``True`` if the given stream is a valid CUDA stream."""
    return stream is not CPUStream

def as_cuda(stream: AbstractStream) -> torch.cuda.Stream:
    """Casts the given stream as :class:`torch.cuda.Stream`."""
    return cast(torch.cuda.Stream, stream)

def get_device(stream: AbstractStream) -> torch.device:
    """Gets the device from CPU or CUDA stream."""
    if is_cuda(stream):
        return as_cuda(stream).device
    return torch.device('cpu')

def wait_stream(source: AbstractStream, target: AbstractStream) -> None:
    """:meth:`torch.cuda.Stream.wait_stream` for either CPU or CUDA stream. It
    makes the source stream wait until the target stream completes work queued.
    """
    if is_cuda(target):
        if is_cuda(source):
            # A CUDA stream waits another CUDA stream.
            as_cuda(source).wait_stream(as_cuda(target))
        else:
            # CPU waits a CUDA stream.
            as_cuda(target).synchronize()

    # If the target is CPU, synchronization is not required.

def record_stream(tensor: torch.Tensor, stream: AbstractStream) -> None:
    """:meth:`torch.Tensor.record_stream` for either CPU or CUDA stream."""
    if is_cuda(stream):
        # NOTE(sublee): record_stream() on a shifted view tensor throws
        # RuntimeError in PyTorch 1.1.0, and does nothing in 1.2.0. To safely
        # protect the tensor against unexpected reallocation, here we use a
        # temporal tensor associated with the same storage without shifting as
        # a workaround.
        #
        # Issue: https://github.com/pytorch/pytorch/issues/27366
        #
        tensor = tensor.new_empty([0]).set_(tensor.storage())

        tensor.record_stream(as_cuda(stream))

"""Autograd functions for stream-aware CUDA copy. It is used to overlap copy
and computation on the same GPU.
"""

# Common interface between :class:`Copy` and :class:`Wait`.
class Context:
    prev_stream: AbstractStream
    next_stream: AbstractStream


class Copy(torch.autograd.Function):
    """Copies tensors on specific streams."""
    @staticmethod
    def forward(ctx: Context,  # type: ignore
                prev_stream: AbstractStream,
                next_stream: AbstractStream,
                *input: Tensor,
                ) -> Tensors:
        ctx.prev_stream = prev_stream
        ctx.next_stream = next_stream

        output = []
        output_stream = current_stream(get_device(next_stream))

        with use_stream(prev_stream), use_stream(next_stream):
            for x in input:
                y = x.to(get_device(next_stream))
                output.append(y)

                # 'prev_stream' is not where 'x' has been allocated.
                record_stream(x, prev_stream)
                # 'y' has been allocated on 'next_stream'.
                # It might be used on the current stream captured as 'output_stream'.
                record_stream(y, output_stream)

        return tuple(output)

    @staticmethod
    def backward(ctx: Context,
                 *grad_output: Tensor,
                 ) -> Tuple[Optional[Tensor], ...]:
        prev_stream = ctx.prev_stream
        next_stream = ctx.next_stream

        grad_input: Deque[Tensor] = deque(maxlen=len(grad_output))
        input_stream = current_stream(get_device(prev_stream))

        with use_stream(prev_stream), use_stream(next_stream):
            for x in reversed(grad_output):
                y = x.to(get_device(prev_stream))
                grad_input.appendleft(y)

                # 'next_stream' is not where 'x' has been allocated.
                record_stream(x, next_stream)
                # 'y' has been allocated on 'prev_stream'.
                # It might be used on the current stream captured as 'input_stream'.
                record_stream(y, input_stream)

        grad_streams: Tuple[Optional[Tensor], ...] = (None, None)
        return grad_streams + tuple(grad_input)


class Wait(torch.autograd.Function):
    """Synchronizes a stream to another stream.

    Place it just before you want to start an operation on the next stream,
    provided that all operations on the previous stream are done.

    """
    @staticmethod
    def forward(ctx: Context,  # type: ignore
                prev_stream: AbstractStream,
                next_stream: AbstractStream,
                *input: Tensor,
                ) -> Tensors:
        ctx.prev_stream = prev_stream
        ctx.next_stream = next_stream

        wait_stream(next_stream, prev_stream)

        return tuple(x.detach() for x in input)

    @staticmethod
    def backward(ctx: Context,
                 *grad_input: Tensor,
                 ) -> Tuple[Optional[Tensor], ...]:
        prev_stream = ctx.prev_stream
        next_stream = ctx.next_stream

        wait_stream(prev_stream, next_stream)

        grad_streams: Tuple[Optional[Tensor], ...] = (None, None)
        return grad_streams + grad_input

def debug_gpu_tensors():
    dic = defaultdict(partial(defaultdict, list))
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                sz = get_bytes(obj) /1024/1024
                dic[type(obj).__name__][tuple(obj.size())].append(sz)
                # print(type(obj), obj.size())
        except:
            pass
    for t, d in dic.items():
        for s, sz in d.items():
            if sum(sz) > 2000:
                print(f"{t} (shape {s}) has {sz} MiB")
    return {k1: {k2: sum(v2) for k2, v2 in v1.items()} for k1, v1 in dic.items()}


glob_total_sz = 0
def track_tensor(*tag: str):
    global glob_total_sz
    total_sz = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                objs = tuple(obj.size())
                if len(objs) == 3 and objs[0] == 24 and objs[2] == 30522:
                    sz = get_bytes(obj) /1024/1024
                    total_sz += sz
                # print(type(obj), obj.size())
        except:
            pass
    if glob_total_sz != total_sz:
        print(f"{' '.join(tag)} - Detect Mem change : {glob_total_sz} -> {total_sz}")
        glob_total_sz = total_sz
    return total_sz