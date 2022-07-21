"""GPU Worker"""
from collections import deque
from copy import deepcopy
from queue import Empty
from queue import Queue as LocalQueue
from threading import Condition, Thread
from time import sleep
from typing import Callable, Dict, List, Optional, Union

import torch

import tspipe.multiprocessing as mp
from tspipe.affinity_manager import AffinityManager
from tspipe.batch_ops import Microbatch
from tspipe.communicator import (Communicator, CommunicatorParam,
                                 DistributedQueue)
from tspipe.gpu_context import (ActivationStorage, DeviceState, GpuTaskContext,
                                LocalTaskContext, ParamStorage,
                                StreamDescriptor, StreamType)
from tspipe.gpu_task import GpuTask, TaskType
from tspipe.logger import Log
from tspipe.profiler import remote_profile_init
from tspipe.utils import new_stream, traverse_tensor_map


class SubGpuWorker():
    def __init__(self, base_worker: 'GpuWorker', role: str = ''):
        self.base_worker = base_worker
        self.device = base_worker.device
        self.thread: Optional[Thread] = None
        self.task_queue: mp.Queue[Optional[GpuTask]] = mp.Queue()
        self.current_task = None
        self.last_task = None
        self.last2_task = None
        self.running = True
        self.debug_info = None
        self.backward_last_lst = [0]
        self.role = role
        self.cv: Optional[Condition] = None
        self.backward_cv: Optional[Condition] = None
        self.complete_task_queue: Optional[Union[LocalQueue, mp.Queue]] = None
        self.complete_task_set: List[GpuTask] = []

    def post_init(self):
        self.cv = Condition()
        self.backward_cv = Condition()
        if self.complete_task_queue is None:
            self.complete_task_queue = LocalQueue()
        self.stashed_task_deque: deque[GpuTask] = deque()

    def start(self):
        self.post_init()
        role_name = f'GPU{self.device.index}{self.role[0:1].capitalize() + self.role[1:]}Thread'
        Log.d(f'{role_name} is starting...')
        self.thread = Thread(target=self._thread, args=())
        self.thread.name = role_name
        self.thread.start()
        Log.d(f'Thread {role_name} is running')
        return self.thread

    def wait_for_task_complete(self, task: 'GpuTask'):
        while True:
            if task in self.complete_task_set:
                self.complete_task_set.remove(task)
                return
            t = self.complete_task_queue.get()
            self.complete_task_set.append(t)

    def schedule(self, task: 'GpuTask'):
        task.schedule(self)
        self.task_queue.put(task)
        if self.base_worker.worker_level_cv:
            with self.base_worker.worker_level_cv:
                self.base_worker.worker_level_cv.notify_all()

        if self.cv is not None:
            self.cv.acquire()
            self.cv.notify()
            self.cv.release()

    def stop(self):
        if self.cv is None:
            self.cv = Condition()
        self.cv.acquire()
        self.running = False
        self.cv.notify()
        self.cv.release()
        self.task_queue.put(None)

    def log(self, *args):
        Log.d(" ".join([f"{self.thread.name if self.thread is not None else 'Uninitialized'}:\t", *args]))

    def log_(self, *args):
        Log.d(" ".join([f"{self.thread.name if self.thread is not None else 'Uninitialized'}:\t", *args]))

    def log__(self, *args):
        Log.d(" ".join([f"{self.thread.name if self.thread is not None else 'Uninitialized'}:\t", *args]))

    def debug_stat(self):
        self.log_(f"Current: {self.current_task}")
        self.log_(f"Last   : {self.last_task}")

    def get_task_ctx(self, task):
        return self.base_worker.list_context[task.ubatch_id]

    def _thread(self):

        def debug_print_condition(task: GpuTask):
            '''For debugging purposes'''
            return False

        while True:
            if not self.running:
                print(f"Terminating thread {self.thread.name}")
                return

            # Block until receiving task
            try:
                task: Optional[GpuTask] = self.task_queue.get(timeout=0.1)
            except Empty:
                continue
            if task is None:
                continue

            while not task.check_precondition(self.get_task_ctx(task)):
                with self.base_worker.worker_level_cv:
                    self.base_worker.worker_level_cv.wait()
                    if not self.running:
                        print(f"Terminating thread {self.thread.name}")
                        return

            self.current_task = task
            assert task is not None
            if debug_print_condition(task):
                self.log__(f"Starting {task}")
            ctx: GpuTaskContext = self.base_worker.list_context[task.ubatch_id]
            task.run(ctx)
            if debug_print_condition(task):
                self.log__(f"Finished {task}")
            task.completed = True
            self.last2_task = self.last_task
            self.last_task = self.current_task
            self.current_task = None
            with self.base_worker.worker_level_cv:
                self.base_worker.worker_level_cv.notify_all()
            if task.task_type == TaskType.TASK_TERMINATE:
                self.stop()
            del task


class SubCpuWorker(SubGpuWorker):
    def __init__(self, base_worker: 'LocalWorker', role: str = ''):
        super().__init__(base_worker, role)
        self.base_worker = base_worker

    def start(self):
        self.stashed_task_deque: deque[GpuTask] = deque()
        self.thread = Thread(target=self._thread, args=())
        self.thread.name = f'CPU0{self.role[0:1].capitalize() + self.role[1:]}Thread'
        self.thread.start()
        return self.thread


class BaseWorker():
    @staticmethod
    def gpu_device_id_to_internal_device_id(device_id: int) -> int:
        # For best NVLink Performance
        device_count = torch.cuda.device_count()
        assert device_id < device_count
        if device_count < 4:
            return device_id
        elif device_count == 4:
            return [0, 1, 2, 3][device_id]
        elif device_count == 8:
            return [0, 3, 2, 1, 5, 6, 7, 4][device_id]
        else:
            return device_id

    @staticmethod
    def partition_id_to_device_id(partition_id: int) -> int:
        device_count = torch.cuda.device_count()
        return partition_id % device_count

    @staticmethod
    def partition_id_to_internal_device_id(partition_id: int) -> int:
        return BaseWorker.gpu_device_id_to_internal_device_id(BaseWorker.partition_id_to_device_id(partition_id))

    def __init__(self, partition_id: Optional[int], num_ubatch: Optional[int], num_bwd_ubatch: Optional[int],
                 communicator_param: Optional[CommunicatorParam], list_context: List[GpuTaskContext]):

        self.device: Optional[torch.device] = None
        self.device_id: Optional[int] = None
        self.internal_device_id: Optional[int] = None
        self.partition_id = partition_id

        if partition_id is not None and partition_id >= 0:
            # allocate gpu
            self.device_id = partition_id % torch.cuda.device_count()
            self.internal_device_id = BaseWorker.gpu_device_id_to_internal_device_id(self.device_id)
            self.device = torch.device('cuda', self.internal_device_id)
            Log.v(f"Partition ID = {self.partition_id}, Device ID = {self.device_id}, Device = {self.device}")
            torch.cuda.set_device(self.device)
        else:
            self.partition_id = -1
        self.num_ubatch = num_ubatch
        self.num_bwd_ubatch = num_bwd_ubatch
        self.process = None
        self.list_context = list_context

        self.communicator_param = communicator_param

        self.complete_task_set = []
        self.complete_task_set_cv = None
        self.running = True

        self.oob_resp_queue = mp.Queue()

        self.worker_level_cv: Optional[Condition] = None

    def start(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

    def cleanup(self):
        raise NotImplementedError

    def wait_until_ready(self):
        while True:
            resp = self.oob_resp_queue.get()
            if resp == 'ready':
                return

    def init_distributed_comm(self):
        self.comm = Communicator(self.device, self.communicator_param)
        Log.v(f"DistributedWorker {self.communicator_param.rank} is now ready.")

        num_partition = self.communicator_param.num_partition
        num_ubatch = max(self.num_ubatch, self.num_bwd_ubatch)
        partition_id = self.partition_id

        assert partition_id == self.communicator_param.rank

        self.num_partition = num_partition

        for cname in [f'init_config_{partition_id}', f'init_model_{partition_id}', f'task_{partition_id}']:
            self.comm.create_channel(cname, self.comm.scheduler_process_rank)

        if partition_id < num_partition - 1:
            for cname in [f'batch_{partition_id+1}_{ubatch_id}' for ubatch_id in range(num_ubatch)]:
                self.comm.create_channel(cname, partition_id+1)

        if partition_id > 0:
            for cname in [f'batch_{partition_id}_{ubatch_id}' for ubatch_id in range(num_ubatch)]:
                self.comm.create_channel(cname, partition_id-1)
        else:
            for cname in [f'batch_{partition_id}_{ubatch_id}' for ubatch_id in range(num_ubatch)]:
                self.comm.create_channel(cname, self.comm.world_size - 1)

        if partition_id > 0:
            for cname in [f'grad_{partition_id-1}_{ubatch_id}' for ubatch_id in range(num_ubatch)]:
                self.comm.create_channel(cname, partition_id-1)

        for cname in [f'grad_{partition_id}_{ubatch_id}' for ubatch_id in range(num_ubatch)]:
            self.comm.create_channel(cname, partition_id+1)

        if partition_id == num_partition - 1:
            cname = 'loss_out'
            self.comm.create_channel(cname, self.comm.scheduler_process_rank)

            cname = 'label_feed'
            self.comm.create_channel(cname, self.comm.scheduler_process_rank)

        cname = f'log_{partition_id}'
        self.comm.create_channel(cname, self.comm.scheduler_process_rank)

        cname = f'model_out_{partition_id}'
        self.comm.create_channel(cname, self.comm.scheduler_process_rank)

        self.comm.mark_ready()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}<{self.partition_id}WorkerProcess>'


class PrefetchWorker():
    def __init__(self, queue_in: Union[DistributedQueue, mp.Queue], queue_out: LocalQueue,
                 to_cuda: Optional[torch.device] = None):
        self.queue_in = queue_in
        self.queue_out = queue_out
        self.to_cuda = to_cuda
        self.running = True
        self.thread = Thread(target=self._thread)
        self.thread.start()

    def _thread(self):
        while self.running:
            try:
                itm = self.queue_in.get(timeout=0.1)
            except Empty:
                continue
            if isinstance(itm, Microbatch):
                traverse_tensor_map(itm.data, lambda t: t.pin_memory() if not t.is_cuda else t)
            if isinstance(itm, Microbatch) and self.to_cuda is not None:
                itm.data = itm.data.to_(self.to_cuda)
            self.queue_out.put(itm)

    def stop(self):
        print("Terminating PrefetchWorker at ", self.thread)
        self.running = False
        self.thread.join()
        print("PrefetchWorker terminated.")


class GpuWorker(BaseWorker):
    def __init__(self,
                 partition_id: Optional[int],
                 num_ubatch: Optional[int],
                 num_bwd_ubatch: Optional[int],
                 communicator_param: CommunicatorParam,
                 optimizer: torch.optim.Optimizer,
                 momentum: float,
                 loss_fn: Callable,
                 update_target_fn: Callable):

        if num_ubatch is None:
            num_ubatch = (communicator_param.world_size - 2)

        if num_bwd_ubatch is None:
            num_bwd_ubatch = num_ubatch * 2

        super().__init__(partition_id, num_ubatch, num_bwd_ubatch, communicator_param, [])

        self.comm: Optional[Communicator] = None
        self.worker_compute = SubGpuWorker(self, 'compute')
        self.worker_copy_batch = SubGpuWorker(self, 'copy_batch')
        self.worker_copy_batch_out = SubGpuWorker(self, 'copy_batch_out')
        self.worker_copy_grad_out = SubGpuWorker(self, 'copy_grad_out')
        self.worker_copy_model_online = SubGpuWorker(self, 'copy_model_online')
        self.worker_copy_model_target = SubGpuWorker(self, 'copy_model_target')
        self.worker_copy_cpu = SubGpuWorker(self, 'copy_cpu')

        self.optimizer = optimizer
        self.momentum = momentum
        self.loss_fn = loss_fn
        self.update_target_fn = update_target_fn

        self.args = None
        self.extra_args = None
        self.partition_online = None
        self.partition_target = None
        self.optimizer_pg_param_map: Optional[Dict[int, List[int]]] = None

        self.start()

    def schedule(self, gpu_task: 'GpuTask'):
        if gpu_task.task_type == TaskType.TASK_COPY_BATCH or gpu_task.task_type == TaskType.TASK_COPY_GRAD:
            return self.worker_copy_batch.schedule(gpu_task)
        elif gpu_task.task_type == TaskType.TASK_COPY_BATCH_OUT:
            return self.worker_copy_batch_out.schedule(gpu_task)
        elif gpu_task.task_type == TaskType.TASK_COPY_GRAD_OUT:
            return self.worker_copy_grad_out.schedule(gpu_task)
        elif gpu_task.task_type == TaskType.TASK_COPY_MODEL:
            return self.worker_copy_model_target.schedule(gpu_task)
        elif gpu_task.task_type == TaskType.TASK_COMPUTE_FORWARD or gpu_task.task_type == TaskType.TASK_COMPUTE_LOSS:
            return self.worker_compute.schedule(gpu_task)
        elif gpu_task.task_type == TaskType.TASK_COMPUTE_BACKWARD:
            return self.worker_compute.schedule(gpu_task)  # do this on compute thread too
        elif gpu_task.task_type == TaskType.TASK_COMPUTE_OPTIMIZE_GPU:
            return self.worker_compute.schedule(gpu_task)  # do this on compute thread too
        elif gpu_task.task_type == TaskType.TASK_TERMINATE:
            self.worker_compute.schedule(deepcopy(gpu_task))
            self.worker_copy_cpu.schedule(deepcopy(gpu_task))
            self.worker_copy_batch.schedule(deepcopy(gpu_task))
            self.worker_copy_batch_out.schedule(deepcopy(gpu_task))
            self.worker_copy_grad_out.schedule(deepcopy(gpu_task))
            self.worker_copy_model_online.schedule(deepcopy(gpu_task))
            self.worker_copy_model_target.schedule(deepcopy(gpu_task))
            return
        assert False

    def init_ctx(self):

        rank = self.comm.rank
        self.args: Dict = self.comm.recv(f'init_config_{rank}')
        self.config: Dict = self.comm.recv(f'init_config_{rank}')
        self.extra_args: Dict = self.comm.recv(f'init_config_{rank}')
        self.optimizer_pg_param_map = self.comm.recv(f'init_config_{rank}')
        Log.d("Received args, args = ", self.args)
        Log.d("Received config, args = ", self.config)

        partition_online, partition_target = self.comm.recv(f'init_model_{rank}')
        self.partition_online = partition_online.to(self.device)
        self.partition_target = partition_target.to(self.device)

        rank, num_ubatch, num_partition = self.partition_id, self.num_ubatch, self.num_partition

        if len(self.list_context) == 0:
            self.list_context = [GpuTaskContext() for _ in range(max(num_ubatch, self.num_bwd_ubatch))]

        params_online, params_target = ParamStorage(), ParamStorage()
        for ctx in self.list_context:
            ctx.device_id = BaseWorker.partition_id_to_device_id(rank)
            ctx.params_online = params_online
            ctx.params_target = params_target
            ctx.config = self.config
            if ctx.config['gpipe_emulation']['enabled']:
                ctx.config['optimizer']['num_skip_initial_staleness'] = None
                print("======= GPipe Emulation Mode Enabled ========")

        params_online.push(0, self.partition_online.parameters())
        params_target.push(0, self.partition_target.parameters())

        for ubatch_id, ctx in enumerate(self.list_context):
            ctx.num_partitions = num_partition
            ctx.num_ubatch = num_ubatch
            ctx.num_bwd_ubatch = self.num_bwd_ubatch
            ctx.args = self.args
            ctx.extra_args = self.extra_args
            ctx.comm = self.comm

            if rank > 0:
                assert ctx.queue_copy_curr_in is None
                if ubatch_id < num_ubatch:
                    ctx.queue_copy_curr_in = DistributedQueue(rank - 1, rank, f'batch_{rank}_{ubatch_id}')
                ctx.queue_grad_out = DistributedQueue(rank, rank-1, f'grad_{rank-1}_{ubatch_id}')
            else:
                if ubatch_id < num_ubatch:
                    ctx.queue_copy_curr_in = DistributedQueue(self.communicator_param.world_size - 1,
                                                              rank, f'batch_{rank}_{ubatch_id}')

            ctx.queue_compute_out = LocalQueue()
            if rank < num_partition - 1:
                if ubatch_id < num_ubatch:
                    ctx.queue_copy_curr_out = DistributedQueue(rank, rank + 1, f'batch_{rank+1}_{ubatch_id}')
                ctx.queue_grad_in = DistributedQueue(rank+1, rank, f'grad_{rank}_{ubatch_id}')

            ctx.partition_online = partition_online
            ctx.partition_target = partition_target

            if rank == num_partition - 1:
                ctx.queue_loss_out = DistributedQueue(rank, self.comm.scheduler_process_rank, 'loss_out')
                if ubatch_id == ctx.num_bwd_ubatch - 1:
                    ctx.queue_label_feed_in = DistributedQueue(rank, self.comm.scheduler_process_rank, 'label_feed')

            ctx.cb_partition_id_to_internal_gpu_id = BaseWorker.partition_id_to_internal_device_id

    def _thread(self):
        Log.v(f"Starting Worker Process for partition {self.partition_id}, device {self.device}")
        self.init_distributed_comm()
        remote_profile_init(DistributedQueue(self.comm.rank, self.comm.scheduler_process_rank,
                                             f'log_{self.partition_id}'))
        self.init_ctx()

        # update optimizer
        list_param = list(self.partition_online.parameters())
        for pg_id, param_id_lst in self.optimizer_pg_param_map.items():
            self.optimizer.param_groups[pg_id]['params'].clear()
            for param_id in param_id_lst:
                self.optimizer.param_groups[pg_id]['params'].append(list_param[param_id])

        for ctx in self.list_context:
            ctx.loss_fn = self.loss_fn
            ctx.optimizer, ctx.momentum, ctx.update_target_fn = self.optimizer, self.momentum, self.update_target_fn
        self.thread_main()

    def thread_main(self):
        device_state = DeviceState(self.partition_id)
        device_id = BaseWorker.partition_id_to_device_id(self.partition_id)
        internal_device_id = BaseWorker.gpu_device_id_to_internal_device_id(device_id)
        cuda_device = torch.device('cuda', internal_device_id)
        prefetch_workers: List[PrefetchWorker] = []

        Log.v(f"Initializing Stream for {self.device}")
        streams = [StreamDescriptor(self.partition_id, StreamType.STREAM_DEFAULT_COMPUTE),
                   StreamDescriptor(self.partition_id, StreamType.STREAM_COPY_BATCH_TO),
                   StreamDescriptor(self.partition_id, StreamType.STREAM_COPY_BATCH_FROM),
                   StreamDescriptor(self.partition_id, StreamType.STREAM_COPY_CPU_TX)]

        def get_stream(desc: StreamDescriptor):
            assert desc.device_id == self.partition_id
            return new_stream(self.device)

        stream_map = {k: get_stream(k) for k in streams}

        local_loss_q = LocalQueue()
        local_grad_q = LocalQueue()
        shared_activation = None
        shared_gradient = None
        shared_final_grad_q = LocalQueue()
        for idx, ctx in enumerate(self.list_context):
            ctx.cuda_streams = stream_map
            ctx.worker = self
            ctx.device_state = device_state

            ctx.queue_compute_in = ctx.queue_copy_out = LocalQueue()

            # Prefetch input
            if ctx.queue_copy_curr_in:
                Log.d(f"Prefetching queue_copy_curr_in for ubatch_idx {idx}")
                ctx.queue_copy_curr_in_unprefetched = ctx.queue_copy_curr_in
                ctx.queue_copy_curr_in = LocalQueue()
                prefetch_workers.append(PrefetchWorker(ctx.queue_copy_curr_in_unprefetched, ctx.queue_copy_curr_in))

            if ctx.queue_label_feed_in:
                Log.d("Prefetching queue_label_feed_in")
                ctx.queue_copy_label_in = LocalQueue()
                prefetch_workers.append(PrefetchWorker(ctx.queue_label_feed_in, ctx.queue_copy_label_in, cuda_device))

            if self.partition_id == self.num_partition - 1:
                # make loss_q here
                ctx.queue_compute_out = local_loss_q
                ctx.queue_loss_compute_in = local_loss_q
            ctx.queue_grad_opt_copy = LocalQueue()

            # if self.partition_id > 0:
            if self.partition_id == self.num_partition - 1:
                ctx.queue_grad_copy = shared_final_grad_q
            else:
                ctx.queue_grad_copy = LocalQueue()
            ctx.queue_grad_pending = LocalQueue()

            if shared_activation is None:
                shared_activation = ActivationStorage(ctx.num_ubatch, f'activation{self.partition_id}')
                shared_gradient = ActivationStorage(ctx.num_ubatch, f'gradient{self.partition_id}')
            ctx.activation = shared_activation
            ctx.gradient = shared_gradient

            if ctx.queue_grad_in is None:
                ctx.queue_grad_in = local_grad_q

            ctx.partition_online, ctx.partition_target = self.partition_online, self.partition_target

        Log.v(f"Initializing Stream for {self.device} Complete!")

        self.worker_level_cv = Condition()

        for w in (self.worker_compute,
                  self.worker_copy_model_online,
                  self.worker_copy_model_target,
                  self.worker_copy_batch,
                  self.worker_copy_batch_out,
                  self.worker_copy_grad_out,
                  self.worker_copy_cpu):
            w.start()

        self.oob_resp_queue.put('ready')
        Log.v(f"Starting Task receiving task_{self.communicator_param.rank}")
        task_queue = DistributedQueue(self.comm.scheduler_process_rank, self.communicator_param.rank,
                                      f'task_{self.communicator_param.rank}')
        while True:
            task: Optional[GpuTask] = task_queue.get()
            if task is None:
                continue
            # print(f"Got {task}")
            self.schedule(task)
            if task.task_type == TaskType.TASK_TERMINATE:
                break
        self.cleanup()

        for worker in prefetch_workers:
            worker.stop()

    def cleanup(self):
        print("Cleaning up...")

        self.worker_compute.thread.join()
        print("worker_compute Thread Join OK")
        self.worker_copy_batch.thread.join()
        print("worker_copy_batch Thread Join OK")
        self.worker_copy_batch_out.thread.join()
        print("worker_copy_batch_out Thread Join OK")
        self.worker_copy_grad_out.thread.join()
        print("worker_copy_grad_out Thread Join OK")
        self.worker_copy_model_online.thread.join()
        print("worker_copy_model_online Thread Join OK")
        self.worker_copy_model_target.thread.join()
        print("worker_copy_model_target Thread Join OK")
        self.worker_copy_cpu.thread.join()
        print("worker_copy_cpu Thread Join OK")

        ctx = self.list_context[0]
        model = ctx.partition_target
        trained_model_state_dict = model.cpu().state_dict()

        # Send paratitions
        self.comm.send(f'model_out_{self.partition_id}', trained_model_state_dict)

        # dirty
        sleep(1)

        # print("Terminating comm")
        self.comm.finish()

    def start(self):
        self.process = mp.Process(target=self._thread)
        self.process.name = f'GPU{self.device_id}WorkerProcess'
        self.process.start()
        AffinityManager(self.process).set_affinity_for_gpu(self.device_id)

    def join(self):
        self.process.join()


class LocalWorker():
    def __init__(self, lst_local_ctx: List['LocalTaskContext']):
        self.device = None
        self.list_context = lst_local_ctx
        self.worker_compute = SubCpuWorker(self, 'cmpt')
        self.worker_copy_batch = SubCpuWorker(self, 'cpyb')
        self.start()

    def start(self):
        Log.v(f"Initializing Localworker {self}")

        self.worker_level_cv = Condition()
        self.worker_compute.start()
        self.worker_copy_batch.start()

    def schedule(self, gpu_task: 'GpuTask'):
        if gpu_task.task_type == TaskType.TASK_FEED_BATCH:
            self.worker_copy_batch.schedule(gpu_task)
        else:
            self.worker_compute.schedule(gpu_task)

    def stop(self):
        print("Stopping LocalWorker")
        self.worker_compute.stop()
        self.worker_copy_batch.stop()

        print("Waiting for LocalWorker join")
        self.worker_compute.thread.join()
        self.worker_copy_batch.thread.join()
