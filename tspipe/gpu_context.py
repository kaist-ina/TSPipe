from collections import defaultdict
from enum import Enum
from logging import warn
from queue import Queue as LocalQueue
from threading import Condition
from typing import (Any, Callable, Dict, Generic, Iterable, List, Optional,
                    Tuple, TypeVar, Union)

import torch
from torch import Tensor

import tspipe.multiprocessing as mp
from tspipe.batch_ops import Batch, BatchList, Microbatch
from tspipe.communicator import Communicator, DistributedQueue
from tspipe.logger import Log
from tspipe.utils import AbstractStream, CPUStream

__all__ = ['GradPartition', 'ParamStorage', 'BatchParam', 'TaskContext', 'GpuTaskContext',
           'LocalTaskContext', 'StreamType', 'StreamDescriptor', 'ActivationStorage']


class GradPartition:
    def __init__(self, grad: Iterable[Tensor], partition_id: int, batch_id: int):
        self.grad = grad
        self.partition_id = partition_id
        self.batch_id = batch_id


class BatchParam:
    def __init__(self, batch_id: int, param_online: List[List[Tensor]], param_target: List[List[Tensor]]):
        self.batch_id = batch_id
        self.param_online = param_online
        self.param_target = param_target


class DeviceState:
    def __init__(self, partition_id: int):
        self.partition_id = partition_id
        self.expected_online_model_id = 0
        self.expected_target_model_id = 0

        self.current_online_model_id = 0
        self.current_target_model_id = 0

        self.forward_complete_ubatch_count = 0
        self.backward_complete_ubatch_count = 0

        self.in_memory_ubatch_count = 0
        self.in_memory_ubatch_bytes = 0
        self.in_memory_activation_bytes = 0
        self.in_memory_model_count = 0
        self.in_memory_model_bytes = 0
        self.in_memory_gradient_bytes = 0

        self.debug_out_f = None
        self.write_cnt = 0

    def print_mem_stat(self):
        return

        # disabled for performance
        if self.debug_out_f is None:
            self.debug_out_f = open(f"mem_stat_{self.partition_id}.csv", "w")

        self.debug_out_f .write(",".join([
            str(self.partition_id),
            str(self.in_memory_ubatch_bytes),
            str(self.in_memory_activation_bytes + self.in_memory_gradient_bytes), str(self.in_memory_model_bytes)])
           + "\n")
        self.write_cnt += 1
        if self.write_cnt % 10 == 0:
            self.debug_out_f.flush()

        Log.i(f"Partition {self.partition_id} : holding {self.in_memory_ubatch_count} ubatch(es), " +
              f"{self.in_memory_activation_count} activations")


V = TypeVar('V')  # is usually Iterable[Tensor]


class ParamStorage(Generic[V]):
    def __init__(self):
        self.storage: Dict[int, Iterable[V]] = {}
        self.cv: Optional[Condition] = None
        self.initialized = False

    def push(self, batch_id: int, params: Iterable[V]) -> None:
        '''Push parameters for batch'''
        if batch_id in self.storage:
            warn(f'Detected duplicated model push (model id = {batch_id}).')
        if self.cv is None:
            self.cv = Condition()
        with self.cv:
            self.initialized = True
            self.storage[batch_id] = params
            self.cv.notify()

    def pop(self, batch_id: int) -> Iterable[V]:
        '''Pop parameters for batch'''
        assert batch_id in self.storage
        return self.storage.pop(batch_id)

    def discard_below(self, batch_id: int):
        '''Discard below. Inclusive'''
        for key in list(self.storage.keys()):
            if key <= batch_id:
                self.storage.pop(key)

    def pop_partition(self, batch_id: int, partition_id: int) -> V:
        '''Pop parameters for batch'''
        assert batch_id in self.storage
        partition = self.storage[batch_id][partition_id]
        assert partition is not None
        self.storage[batch_id][partition_id] = None
        if all(v is None for v in self.storage[batch_id]):
            self.storage.pop(batch_id)
        return partition

    def wait_partition(self, batch_id: int):
        if self.cv is None:
            self.cv = Condition()

        with self.cv:
            while True:
                if batch_id in self.storage:
                    break
                else:
                    self.cv.wait()
                    continue

    def peek(self, batch_id: int) -> Iterable[V]:
        '''Peek parameters for batch'''
        # print(f"Peek parameter for batch id {batch_id}")
        if batch_id not in self.storage:
            warn(f'Requested model {batch_id}, but we only have models for {list(self.storage.keys())}.')
        assert batch_id in self.storage
        return self.storage[batch_id]

    def has(self, batch_id: int) -> bool:
        return batch_id in self.storage

    def __len__(self) -> int:
        return len(self.storage)


class StreamType(Enum):
    STREAM_DEFAULT_COMPUTE = 0
    STREAM_COPY_BATCH_FROM = 1
    STREAM_COPY_BATCH_TO = 2
    STREAM_COPY_CPU_TX = 3


class StreamDescriptor:
    def __init__(self, device_id: int, type: StreamType):
        self.device_id = device_id
        self.type = type
        if self.device_id is None:
            self.device_id = -1

    def __eq__(self, o: object) -> bool:
        return self.device_id == o.device_id and self.type == o.type

    def __hash__(self) -> int:
        return hash(f"{self.device_id}_{self.type}")

    def __repr__(self) -> str:
        return f"StreamDescriptor<device_id={self.device_id}, type={self.type}>"


class ActivationStorage():
    def __init__(self, ubatch_num: int, alias: str = ''):
        self.storage = defaultdict(dict)
        self.ubatch_num = ubatch_num
        self.alias = alias

    @staticmethod
    def _key(ubatch: Microbatch) -> str:
        return f"{ubatch.ubatch_id}_{ubatch.view_id}"

    def push(self, ubatch: Microbatch, path: Tuple = tuple()) -> None:
        assert not ubatch.is_target, "Designed to store only activations from online network"
        key = ActivationStorage._key(ubatch)
        if key not in self.storage[ubatch.batch_id]:
            self.storage[ubatch.batch_id][key] = {}
        self.storage[ubatch.batch_id][key][path] = ubatch.data

    def push_loss(self, batch_id: int, loss: torch.Tensor) -> None:
        self.storage[batch_id]['loss'] = loss

    def pop(self, batch_id: int, view_id: int, ubatch_id: int) -> Dict[Tuple, Any]:
        key = f"{ubatch_id}_{view_id}"
        batches: Dict[Tuple, Batch] = self.storage[batch_id][key]
        del self.storage[batch_id][key]
        if len(self.storage[batch_id]) == 0:
            del self.storage[batch_id]
        return {k: b.value for k, b in batches.items()}

    def has(self, batch_id: int, view_id: int, ubatch_id: int) -> bool:
        key = f"{ubatch_id}_{view_id}"
        return batch_id in self.storage and key in self.storage[batch_id]


class TaskContext:
    def __init__(self):
        self.queue_copy_out: Optional[Union[mp.Queue[Microbatch], LocalQueue[Microbatch]]] = None
        self.num_partitions: Optional[int] = None
        self.num_ubatch: Optional[int] = None
        self.num_bwd_ubatch: Optional[int] = None
        self.cuda_streams: 'Dict[StreamDescriptor, AbstractStream]' = {}

        self.device_id = None
        self.comm: Optional[Communicator] = None
        self.worker = None
        self.sub_worker = None
        self.config = {}

    def find_stream(self, desc: StreamDescriptor):
        ''' Deprecated '''
        if desc.device_id < 0:
            return CPUStream
        for k, v in self.cuda_streams.items():
            if k == desc:
                return v
        assert False, (f"Warning: Couldn't find Stream {desc} in list {self.cuda_streams.keys()}")

    @staticmethod
    def create_local_contexts(num_partitions: int, num_ubatch: int,
                              num_bwd_ubatch: int, comm: Communicator, config: Dict) -> Tuple[List['LocalTaskContext']]:
        lst_local_ctx = [LocalTaskContext() for _ in range(num_ubatch)]
        lst_queue_batch_feed_out = []
        for ubatch_id in range(num_ubatch):
            comm.create_channel(f'batch_0_{ubatch_id}', 0)
            lst_queue_batch_feed_out.append(DistributedQueue(comm.world_size-1, 0, f'batch_0_{ubatch_id}'))

        def basic_init(ctx: TaskContext):
            ctx.num_partitions = num_partitions
            ctx.num_ubatch = num_ubatch
            ctx.num_bwd_ubatch = num_bwd_ubatch
            ctx.config = config

        for ubatch_id in range(max(num_ubatch, num_bwd_ubatch)):
            if ubatch_id < num_ubatch:
                basic_init(lst_local_ctx[ubatch_id])

            if ubatch_id < num_ubatch:
                lst_local_ctx[ubatch_id].lst_queue_batch_feed_out = lst_queue_batch_feed_out

        comm.create_channel('label_feed', num_partitions-1)
        lst_local_ctx[0].queue_label_feed_out = DistributedQueue(comm.world_size-1, num_partitions-1, 'label_feed')

        return lst_local_ctx


class LocalTaskContext(TaskContext):
    def __init__(self):
        super().__init__()
        self.lst_queue_batch_feed_out: Optional[mp.Queue[Microbatch]] = None
        self.queue_label_feed_out: Optional[DistributedQueue[Microbatch]] = None


class GpuTaskContext(TaskContext):
    def __init__(self):
        super().__init__()
        self.device_state: Optional['DeviceState'] = None

        # compute_optimizer
        self.loss_fn: Optional[Callable] = None

        self.activation: Optional['ActivationStorage'] = None
        self.gradient: Optional['ActivationStorage'] = None

        # compute_forward
        self.queue_compute_in: Optional[DistributedQueue[Microbatch]] = None
        self.queue_compute_out: Optional[LocalQueue[Microbatch]] = None

        # compute_loss
        self.queue_loss_compute_in: Optional[LocalQueue[Microbatch]] = None
        self.queue_loss_out: Optional[DistributedQueue[float]] = None
        self.queue_copy_label_in: Optional[LocalQueue[Microbatch]] = None
        self.queue_label_feed_in: Optional[DistributedQueue[Microbatch]] = None

        # copy_batch
        self.queue_copy_curr_out: Optional[Union[LocalQueue[Microbatch], DistributedQueue[Microbatch]]] = None
        self.queue_copy_curr_in: Optional[
            Union[mp.Queue[Microbatch], DistributedQueue[Microbatch], LocalQueue[Microbatch]]] = None
        self.queue_copy_curr_in_unprefetched: Optional[
            Union[mp.Queue[Microbatch], DistributedQueue[Microbatch]]] = None  # before prefetch

        # compute_backward
        self.queue_grad_in: Optional[DistributedQueue[BatchList]] = None
        self.queue_grad_copy: Optional[LocalQueue[BatchList]] = None
        self.queue_grad_pending: Optional[LocalQueue[BatchList]] = None
        self.queue_grad_out: Optional[DistributedQueue[BatchList]] = None

        # copy_grad_to_optimizer
        self.queue_grad_opt_copy: Optional[LocalQueue[GradPartition]] = None

        self.partition_online: Optional[torch.nn.Sequential] = None
        self.partition_target: Optional[torch.nn.Sequential] = None

        # for the main process

        self.args: Dict = {}
        self.extra_args: Dict = {}

        self.params_online: Optional[ParamStorage[Tensor]] = None
        self.params_target: Optional[ParamStorage[Tensor]] = None

        # gpu optimize
        self.update_target_fn: Optional[Callable] = None
        self.momentum: Optional[float] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

        self.cb_partition_id_to_internal_gpu_id: Optional[Callable] = None

        self.grad_accumulation: Optional[List[Tensor]] = None
        self.num_grad_accumulation: int = 0

    def get_stream(self, type: StreamType):
        assert self.device_id is not None, "Error: cannot get stream from non-GPU context"
        assert self.device_state.partition_id is not None, "Error: cannot get stream from non-GPU context"
        desc = StreamDescriptor(self.device_state.partition_id, type)
        for k, v in self.cuda_streams.items():
            if k == desc:
                return v
        assert False, (f"Warning: Couldn't find Stream {desc} in list {self.cuda_streams.keys()}")
