import warnings
from argparse import ArgumentParser, Namespace
from collections import OrderedDict, defaultdict
from copy import deepcopy
from enum import Enum
from itertools import chain
from platform import python_version
from queue import Empty
from threading import Thread
from time import sleep, time
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union, cast

import torch
import yaml

import tspipe
from tspipe import batch_ops
from tspipe.affinity_manager import AffinityManager
from tspipe.batch_ops import (Batch, BatchQueue, ScatterGatherFn,
                              defaultScatterGatherFn)
from tspipe.batchnorm import DeferredBatchNorm
from tspipe.communicator import (Channel, Communicator, CommunicatorParam,
                                 DistributedQueue)
from tspipe.gpu_context import StreamDescriptor, StreamType, TaskContext
from tspipe.gpu_task import GpuTask, TaskType
from tspipe.gpu_worker import GpuWorker, LocalWorker
from tspipe.logger import Log
from tspipe.profiler import Operation, ProfilerDelegateWorker, profile_init
from tspipe.scheduler import TSPipeScheduler
from tspipe.utils import get_shape, verify_module

BATCH_FEED_THESH = 4

Tensors = Tuple[torch.Tensor, ...]
TensorOrTensors = Union[torch.Tensor, Tensors]

class BalanceError(ValueError):
    pass

class PipelineRunState(Enum):
    STOPPED = 0
    RUNNING = 1
    PENDING_STOP = 2


class TSPipeMode(Enum):
    SELF_SUPERVISED_MOMENTUM = 0
    SUPERVISED_MOMENTUM = 1

class TSPipe():
    def __init__(self,
                 module_online: torch.nn.Sequential,
                 module_target: torch.nn.Sequential,
                 module_predictor: Optional[torch.nn.Sequential],
                 optimizer: torch.optim.Optimizer,
                 loss_fn: Callable,
                 target_update_fn: Optional[Callable],
                 momentum: float,
                 artifact_dir: str = '',
                 tspipe_mode: TSPipeMode = TSPipeMode.SELF_SUPERVISED_MOMENTUM,
                 target_train_mode: bool = True,
                 extra_args: Namespace = Namespace(),
                 scatter_gather_fn: ScatterGatherFn = defaultScatterGatherFn,
                ):
        parser = ArgumentParser()
        parser.add_argument("--tspipe-config", required=True, type=str)
        parser.add_argument("--ip", required=True, type=str)
        parser.add_argument("--rank", required=True, type=int, default=0)
        parser.add_argument("--num-nodes", required=True, type=int, default=1)

        args, _ = parser.parse_known_args()
        self.args               = args
        self.module_online      = module_online
        self.module_target      = module_target
        self.module_predictor   = module_predictor
        self.optimizer          = optimizer
        self.loss_fn            = loss_fn
        self.target_update_fn   = target_update_fn
        self.momentum           = momentum
        self.tspipe_mode      = tspipe_mode
        self.target_train_mode  = target_train_mode
        self.extra_args         = extra_args
        self.scatter_gather_fn  = scatter_gather_fn

        Log.i(f"====== TSPipe (v{tspipe.__version__}) Initializing =====")
        Log.i(f"Running on Python {python_version()}, PyTorch {torch.__version__}")
        with open(args.tspipe_config, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)['tspipe']
        self.config['__artifact_dir'] = artifact_dir

        self.num_devices = torch.cuda.device_count()
        Log.v(f"Detected {self.num_devices}")

        self.rank, self.num_nodes, self.total_world_size = args.rank, args.num_nodes, self.num_devices*args.num_nodes + 1
        self.num_total_devices = self.num_devices*self.num_nodes
        Log.v(f"Starting Node with rank {args.rank}")
        
        # configure number of microbatches
        # assuming all nodes have the same number of GPUs
        Log.v(f"Using {self.num_nodes} nodes x {self.num_devices} GPUs = {self.num_nodes*self.num_devices}")
        Log.v(self.config)
        
        if self.num_nodes*self.num_devices != len(self.config['model_split']['online']):
            raise ValueError(f"The number of GPUs and the number of partitions must match.")
        
        if self.config['gpipe_emulation']['enabled']:
            self.num_ubatches = self.config['gpipe_emulation']['num_ubatch']
            self.num_bwd_ubatches = self.num_ubatches
        else:
            self.num_ubatches = self.num_total_devices - 1
            self.num_bwd_ubatches = self.num_ubatches * 2

        if self.target_update_fn is None:
            warnings.warn("Target update function is not designated. Target network will not be updated.")
            
        # start gpu worker nodes
        self.gpu_workers: List[GpuWorker] = []
        for partition_id in range(self.rank * self.num_devices, (self.rank + 1) * self.num_devices):
            Log.v(f"Spawning worker with partition {partition_id}")
            self.gpu_workers.append(self.spawn_new_gpu_workers(partition_id))

        # start pipeline for primary node (rank 0)
        if self.rank == 0:
            sleep(1)
            self.start_primary()
        else:
            self.join_workers()
        
    def start_primary(self):
        """Initialize and start pipeline for the primary node."""
        self.running: PipelineRunState = PipelineRunState.STOPPED        
        self.batch_id: int = 1
        self.highest_scheduled_batch_id: int = 0
        self.forward_complete_batch_idx: int = 0
        self.batch_dict: Dict[int, Tuple[List[Batch], List[Batch],List[Batch], List[Batch]]] = {}
        self.comm = Communicator(None, CommunicatorParam(self.args.ip, self.total_world_size, rank=self.total_world_size-1))
        self.batch_q = BatchQueue()
        self.scheduled_momentum_update = {}
        self.scheduled_lr_update = {}
        
        Log.v("GPU-CPU affinity map:", AffinityManager().get_gpu_affinity_map())
        AffinityManager().set_affinity_for_scheduler()

        # module validity check
        if not self.config['model_split']['target'] or not self.config['model_split']['target']:
            raise NotImplementedError("Model Split is not configured.")
        verify_module(self.module_online)
        verify_module(self.module_target)
        verify_module(self.module_predictor)

        # convert module if deferred batch norm is enabled
        if self.config['train']['deferred_batch_norm']:
            self.module_online = DeferredBatchNorm.convert_deferred_batch_norm(self.module_online, self.num_ubatches)
            self.module_target = DeferredBatchNorm.convert_deferred_batch_norm(self.module_target, self.num_ubatches)
            if self.module_predictor:
                self.module_predictor = DeferredBatchNorm.convert_deferred_batch_norm(self.module_predictor, self.num_ubatches)

        # split module
        self.partitions_online: List[torch.nn.Sequential] = []
        self.partitions_target: List[torch.nn.Sequential] = []
        self.partitions_online, self.partitions_target = \
            self.split_module(self.config['model_split']['online'], self.config['model_split']['target'])
        assert(self.partitions_online)
        assert(self.partitions_target)
        assert(len(self.partitions_online) == len(self.partitions_target))
        self.num_partitions = len(self.partitions_online)
        self.task_scheduler = TSPipeScheduler(self.config, self.batch_q, self.num_partitions)

        # build optimizer map
        param_partitition_map = {}
        for partition_id, partition in enumerate(self.partitions_online):
            for param_id, param in enumerate(partition.parameters()):
                param_partitition_map[param] = param_id, partition_id
        optimizer_pg_param_map = [defaultdict(set) for _ in range(self.num_partitions)]
        for pg_id, pg in enumerate(self.optimizer.param_groups):
            for param in pg['params']:
                param_id, partition_id = param_partitition_map[param]
                optimizer_pg_param_map[partition_id][pg_id].add(param_id)
        optimizer_pg_param_map = [{k: list(v) for k, v in itm.items()} for itm in optimizer_pg_param_map]
        del param_partitition_map
            
        # enable training mode
        for partition in self.partitions_online:
            partition.train()

        if self.target_train_mode:
            for partition in self.partitions_target:
                partition.train()
        else:
            for partition in self.partitions_target:
                partition.eval()
            warnings.warn("Target train mode is disabled. Target network will be evaluated with eval mode.")        

        # Initialize channels
        for i in range(0, self.num_partitions):
            for prefix in ['task', 'init_config', 'init_model']:
                self.comm.create_channel(f'{prefix}_{i}', i)

        self.comm.create_channel('loss_out', self.num_partitions - 1)
        self.forward_out_queue = DistributedQueue(self.num_partitions - 1, self.comm.rank, 'loss_out')
        
        # collect model after training and profiler logs during training
        self.model_out_channels: List[Channel] = []
        log_channels: List[Channel] = []
        for i in range(0, self.num_partitions):
            self.model_out_channels.append(self.comm.create_channel(f'model_out_{i}', self.num_partitions - 1))
            log_channels.append(self.comm.create_channel(f'log_{i}', self.num_partitions - 1))
        self.comm.create_channel_mux('log', *log_channels)
        self.profiler_delegate_worker = ProfilerDelegateWorker(DistributedQueue(None, self.comm.rank, 'log'))

        # barrier to wait other workers
        self.comm.wait_ready()

        # initialize local worker
        lst_local_context = TaskContext.create_local_contexts(self.num_partitions, self.num_ubatches, self.num_bwd_ubatches, self.comm, self.config)
        self.local_worker = LocalWorker(lst_local_context)
        

        for i in range(0, self.num_partitions):
            self.comm.send(f'init_config_{i}', self.args)
            self.comm.send(f'init_config_{i}', self.config)
            self.comm.send(f'init_config_{i}', self.extra_args)
            self.comm.send(f'init_config_{i}', optimizer_pg_param_map[i])
            self.comm.send(f'init_model_{i}', (self.partitions_online[i], self.partitions_target[i]))

        # start profiler
        profile_init()

        # start pipeline
        self.start_pipeline()

    def join_workers(self):
        Log.i("Waiting until workers finish their jobs...")
        for worker in self.gpu_workers:
            worker.process.join()

    @property
    def is_primary(self):
        return self.rank == 0

    def spawn_new_gpu_workers(self, partition_id: int):
        """Spawn a new GPU worker for partition `partition_id`."""
        return GpuWorker(partition_id,
                         self.num_ubatches,
                         self.num_bwd_ubatches,
                         CommunicatorParam(self.args.ip, self.total_world_size, partition_id), 
                         self.optimizer, 
                         self.momentum,
                         self.loss_fn,
                         self.target_update_fn)
 
    def _thread_scheduler(self):
        self.running = PipelineRunState.RUNNING
        self.processed_batch_idx = 1
        set_terminated_partition = set()
        gpipe_emulation_enabled = self.config['gpipe_emulation']['enabled']
        num_skip_initial_staleness = self.config['optimizer']['num_skip_initial_staleness']
        epoch = 0

        for schedules in self.task_scheduler.schedule_generator():
            if len(set_terminated_partition) == self.num_partitions:
                Log.v("Finishing scheduling loop, since termination signal is sent to everyone")
                self.running = PipelineRunState.STOPPED
            
            if self.running == PipelineRunState.STOPPED:
                Log.v("Stopping Thread Scheduler")
                return

            for sched in schedules:
                if sched is not None:
                    is_target = sched.is_target
                    terminate = False
                    if self.running != PipelineRunState.RUNNING and sched.batch_idx > self.last_batch_idx:
                        terminate = True
                        
                    if terminate:
                        if sched.j not in set_terminated_partition:
                            task_terminate = GpuTask(TaskType.TASK_TERMINATE, sched.batch_idx, sched.view_idx, sched.i, sched.j, is_target)
                            Log.v(f"Issuing termination to partition {sched.j}")
                            self.schedule_task(task_terminate)
                            set_terminated_partition.add(sched.j)
                        continue
                    
                    stream_compute = StreamDescriptor(sched.j, StreamType.STREAM_DEFAULT_COMPUTE)
                    stream_from = StreamDescriptor(sched.j, StreamType.STREAM_COPY_BATCH_FROM)

                    if sched.optimize:
                        if sched.i != 0:
                            continue
                        o = self.did_epoch_terminate(sched.batch_idx)
                        new_momentum, new_lr = None, None
                        last_optim_partition = self.num_partitions - 1
                        if gpipe_emulation_enabled or sched.batch_idx <= num_skip_initial_staleness:
                            last_optim_partition = 0
                        if sched.batch_idx in self.scheduled_momentum_update:
                            new_momentum = self.scheduled_momentum_update[sched.batch_idx]
                            if sched.j == last_optim_partition:
                                self.scheduled_momentum_update.pop(sched.batch_idx)
                        if sched.batch_idx in self.scheduled_lr_update:
                            new_lr = self.scheduled_lr_update[sched.batch_idx]
                            if sched.j == last_optim_partition:
                                self.scheduled_lr_update.pop(sched.batch_idx)
                        task_compute_optimize_gpu = GpuTask(TaskType.TASK_COMPUTE_OPTIMIZE_GPU, sched.batch_idx, sched.view_idx, \
                                                            sched.i, sched.j, is_target, optimizer_step=o, new_momentum=new_momentum, new_lr=new_lr)
                        self.schedule_task(task_compute_optimize_gpu)
                        if o:
                            epoch += 1
                        continue

                    if self.tspipe_mode == TSPipeMode.SUPERVISED_MOMENTUM:
                        view_bool = True if sched.view_idx == 1 else False
                        if view_bool != sched.is_target:
                            # print(f"Skipping transmission of {sched}")
                            if sched.model_update_ver is not None and sched.model_update_ver > 0:
                                task_model_copy = GpuTask(TaskType.TASK_COPY_MODEL, sched.batch_idx, sched.view_idx, sched.i, sched.j, is_target, 
                                dst_stream=stream_compute, src_stream=stream_copy_dst)
                                self.schedule_task(task_model_copy)
                            continue
                        else:
                            pass
                            # print(f"Allowing transmission of {sched}")

                    if sched.backprop:
                        is_target = False
                        task_backward = GpuTask(TaskType.TASK_COMPUTE_BACKWARD, sched.batch_idx, sched.view_idx, \
                                                sched.i, sched.j, is_target, src_stream=stream_compute, wait_stream=stream_from, asymmetric=(self.tspipe_mode==TSPipeMode.SUPERVISED_MOMENTUM))
                        num_bwd_batch = self.num_bwd_ubatches if not gpipe_emulation_enabled and sched.batch_idx > num_skip_initial_staleness else self.num_ubatches
                        if self.tspipe_mode == TSPipeMode.SELF_SUPERVISED_MOMENTUM and sched.j == self.num_partitions - 1 and sched.i == num_bwd_batch - 1 and sched.view_idx == 1 or \
                           self.tspipe_mode == TSPipeMode.SUPERVISED_MOMENTUM      and sched.j == self.num_partitions - 1 and sched.i == num_bwd_batch - 1:
                            task_loss = GpuTask(TaskType.TASK_COMPUTE_LOSS, sched.batch_idx, sched.view_idx, sched.i, sched.j, is_target, src_stream=stream_compute, 
                                                asymmetric=self.tspipe_mode==TSPipeMode.SUPERVISED_MOMENTUM, epoch=epoch, scatter_gather_fn=self.scatter_gather_fn)
                            self.schedule_task(task_loss)
                        self.schedule_task(task_backward)

                        if sched.j > 0:
                            task_copy_grad_out = GpuTask(TaskType.TASK_COPY_GRAD_OUT, sched.batch_idx, sched.view_idx, \
                                                        sched.i, sched.j, is_target, src_stream=stream_compute, wait_stream=stream_from, asymmetric=(self.tspipe_mode==TSPipeMode.SUPERVISED_MOMENTUM))
                            self.schedule_task(task_copy_grad_out)

                            stream_compute = StreamDescriptor(sched.j-1, StreamType.STREAM_DEFAULT_COMPUTE)
                            stream_copy_dst = StreamDescriptor(sched.j-1, StreamType.STREAM_COPY_BATCH_TO)
                            stream_copy_src = StreamDescriptor(sched.j, StreamType.STREAM_COPY_BATCH_FROM)
                            task_grad_copy = GpuTask(TaskType.TASK_COPY_GRAD, sched.batch_idx, sched.view_idx, sched.i, sched.j-1, is_target, 
                                                     dst_stream=stream_copy_dst, src_stream=stream_copy_src, wait_stream=stream_compute)
                            self.schedule_task(task_grad_copy)
                    else:
                        stream_copy_dst = StreamDescriptor(sched.j, StreamType.STREAM_COPY_BATCH_TO)
                        stream_copy_src = StreamDescriptor(sched.prev_partition, StreamType.STREAM_COPY_BATCH_FROM)
                        stream_copy_to_next = StreamDescriptor(sched.j, StreamType.STREAM_COPY_BATCH_FROM) 
                        task_compute = GpuTask(TaskType.TASK_COMPUTE_FORWARD, sched.batch_idx, sched.view_idx, sched.i, sched.j, is_target, 
                                               dst_stream=stream_copy_dst, src_stream=stream_compute, wait_stream=stream_copy_to_next, scatter_gather_fn=self.scatter_gather_fn)
                        task_batch_copy_out = GpuTask(TaskType.TASK_COPY_BATCH_OUT, sched.batch_idx, sched.view_idx, sched.i, sched.j, is_target, asymmetric=(self.tspipe_mode==TSPipeMode.SUPERVISED_MOMENTUM))
                        if sched.model_update_ver is not None and sched.model_update_ver > 0:
                            task_model_copy = GpuTask(TaskType.TASK_COPY_MODEL, sched.batch_idx, sched.view_idx, sched.i, sched.j, is_target, 
                            dst_stream=stream_compute, src_stream=stream_copy_dst)
                            self.schedule_task(task_model_copy)

                        task_batch_copy = GpuTask(TaskType.TASK_COPY_BATCH, sched.batch_idx, sched.view_idx, sched.i, sched.j, is_target, 
                                                  dst_stream=stream_copy_dst, src_stream=stream_copy_src, wait_stream=stream_compute)
                        self.schedule_task(task_batch_copy)
                        self.schedule_task(task_compute)
                        if sched.j < self.num_partitions - 1:
                            self.schedule_task(task_batch_copy_out)

    def did_epoch_terminate(self, batch_id: int) -> bool:
        return batch_id in self.batch_q.epoch_boundaries

    def schedule_task(self, gpu_task: GpuTask):
        self.highest_scheduled_batch_id = max(self.highest_scheduled_batch_id, gpu_task.batch_id)
        if gpu_task.task_type == TaskType.TASK_TERMINATE:
            if gpu_task.partition_id is not None and gpu_task.partition_id >= 0:
                self.comm.send(f'task_{gpu_task.partition_id}', gpu_task)
            else:
                self.local_worker.schedule(deepcopy(gpu_task))
        elif gpu_task.task_type == TaskType.TASK_COPY_BATCH or gpu_task.task_type == TaskType.TASK_COPY_BATCH_OUT:
            self.comm.send(f'task_{gpu_task.partition_id}', gpu_task)
        elif gpu_task.task_type == TaskType.TASK_COPY_GRAD or gpu_task.task_type == TaskType.TASK_COPY_MODEL:
            self.comm.send(f'task_{gpu_task.partition_id}', gpu_task)
        elif gpu_task.task_type == TaskType.TASK_FEED_BATCH:
            self.local_worker.schedule(gpu_task)
        else:
            self.comm.send(f'task_{gpu_task.partition_id}', gpu_task)

    def start_pipeline(self) -> None:
        """Starts pipeline that schedules computing tasks to each GPU."""
        self.thread = Thread(
            target=self._thread_scheduler,
            args=[],
            daemon=True,
        )
        self.thread.name='SchedulerThread'
        self.thread.start()

    def feed(self, view_1: TensorOrTensors, view_2: TensorOrTensors, view_target: Optional[TensorOrTensors] = None) -> Optional[torch.Tensor]:
        """Feed new dataset to pipeline."""
        assert self.running != PipelineRunState.PENDING_STOP
        assert self.rank == 0, "Must be primary to feed batch"
        
        # sanity check
        if torch.is_tensor(view_1):
            batch_ops.check(view_1)
        if torch.is_tensor(view_2):
            batch_ops.check(view_2)
        
        # split into multiple batches for gradient accumulation
        # only apply gradient accumulation if microbatch size is geq than 4
        assert self.scatter_gather_fn.batch_size(view_1) == self.scatter_gather_fn.batch_size(view_2)
        gradient_accumulation = self.config['optimizer']['gradient_accumulation']
        if self.scatter_gather_fn.batch_size(view_1) >= 4 * self.num_ubatches * gradient_accumulation:
            minibatches_view_m1 = self.scatter_gather_fn.scatter(view_1, gradient_accumulation)
            minibatches_view_m2 = self.scatter_gather_fn.scatter(view_2, gradient_accumulation)
        else:
            minibatches_view_m1 = self.scatter_gather_fn.scatter(view_1, 1)
            minibatches_view_m2 = self.scatter_gather_fn.scatter(view_2, 1)
        
        # start the pipeline in another thread if not running
        if self.running == PipelineRunState.STOPPED:
            self.start_pipeline()

        # inject batches to batch_q and wait for the loss
        for view_m1, view_m2 in zip(minibatches_view_m1, minibatches_view_m2):
            batches_1 = self.scatter_gather_fn.scatter(view_m1.tensor_or_tensors, self.num_ubatches)
            batches_2 = self.scatter_gather_fn.scatter(view_m2.tensor_or_tensors, self.num_ubatches)

            bid = self.batch_q.next_id
            with Operation('feed_api', batch_idx=bid):
                if self.tspipe_mode == TSPipeMode.SELF_SUPERVISED_MOMENTUM:
                    batches_lists = [batches_1, batches_2, batches_2, batches_1] # no clone needed here
                    asymmetric = False
                    assert view_target is None, "Self-supervised learning does not support labels."
                    batch_view_target = None
                elif self.tspipe_mode == TSPipeMode.SUPERVISED_MOMENTUM:
                    batches_lists = [batches_1, batches_2] # no clone needed here
                    asymmetric = True
                    assert view_target is not None, "Supervised learning requires labels."
                    batch_view_target = Batch(view_target)
                else:
                    assert False, "Invalid Mode"
                batch_idx = self.batch_q.get_new_batch_id()
                self.last_batch_idx = batch_idx
                self.schedule_task(GpuTask(TaskType.TASK_FEED_BATCH, batch_idx, 0, 0, 0, 0, batch_list=batches_lists, asymmetric=asymmetric, label_batch=batch_view_target))
            loss = self.wait_forward()

        # loss may be None for first few initial iterations
        return loss

    def feed_epoch(self):
        print(f"epoch will end at batch {self.last_batch_idx+1}")
        self.batch_q.report_epoch_boundary()


    def stop(self) -> None:
        assert self.rank == 0

        Log.v("Issued stop here!")
        self.running = PipelineRunState.PENDING_STOP
        self.batch_q.stop()

        task_terminate = GpuTask(TaskType.TASK_TERMINATE, 0, 0, 0, -1, False)
        Log.v(f"Issuing termination to CPU partition")
        self.schedule_task(task_terminate)

        # wait for scheduler thread stop
        self.thread.join()

        for idx, channel in enumerate(self.model_out_channels):
            print(channel.name)
            new_state_dict = channel.recv()
            print(f"got new_state_dict f{get_shape(new_state_dict)}")
            self.partitions_target[idx].load_state_dict(new_state_dict)

        print("Terminating local_worker")
        self.local_worker.stop()

        print("Terminating Log queue")
        self.profiler_delegate_worker.join()
        
        print("Terminating GPU workers")
        for w in self.gpu_workers:
            w.join()
        
        print("Terminating comm")
        self.comm.finish(False)
        sleep(10)

    def wait_forward(self) -> Optional[float]:
        """Blocks the current thread until the next computation for loss is complete,
        provided batch feeds are sufficient enough to fully utilize the pipeline.
        Immediately returns None if the batch feeds are insufficient. (i.e., in the beginning of training)

        Returns:
            Computed loss, or None
        """
        try:
            forward_batch_id, loss = self.forward_out_queue.get_nowait()
            self.forward_complete_batch_idx = forward_batch_id
            return loss
        except Empty:
            pass

        # if batch feeds are insufficient, return immediately
        if self.batch_q.next_id < self.forward_complete_batch_idx + 8:
            Log.d(f"Wait_forward IMM2: Next batch to arrive {self.batch_q.next_id}, Last enqueued batch {self.processed_batch_idx}, Last forward complete batch {self.forward_complete_batch_idx}")
            return None

        # wait for the loss computation result
        forward_batch_id, loss = self.forward_out_queue.get()
        self.forward_complete_batch_idx = forward_batch_id
        return loss

    def update_momentum(self, new_momentum: float):
        assert self.batch_q.next_id > self.highest_scheduled_batch_id, f"Batch {self.highest_scheduled_batch_id} has been already scheduled. Trying to schedule {self.batch_q.next_id}"
        self.scheduled_momentum_update[self.batch_q.next_id] = new_momentum
        Log.i(f"Scheduling momentum update at batch {self.batch_q.next_id} to {new_momentum}")

    def update_lr(self, new_lr: Union[float, List[float]]):
        assert self.batch_q.next_id > self.highest_scheduled_batch_id, f"Batch {self.highest_scheduled_batch_id} has been already scheduled. Trying to schedule {self.batch_q.next_id}"
        self.scheduled_lr_update[self.batch_q.next_id] = new_lr
        Log.i(f"Scheduling lr update at batch {self.batch_q.next_id} to {new_lr}")

    def split_module(self, balance_online: Iterable[int], balance_target: Iterable[int]) -> Tuple[List[torch.nn.Sequential], List[torch.nn.Sequential]]:
        """Splits `self.module_online`+`self.module_predictor` and `self.module_target`,
         and stores them into `self.partitions_online` and `self.partitions_target`.

        Returns: 
            A tuple of (`List[model]` for online module, `List[model]` for target module)
            
        Raises:
            BalanceError:
                wrong balance
        """
        module_online, module_target, module_predictor = self.module_online, self.module_target, self.module_predictor
        balance_online = list(balance_online)
        balance_target = list(balance_target)

        if len(module_online) + (len(module_predictor) if module_predictor is not None else 0) != sum(balance_online):
            raise BalanceError('online module and sum of balance have different length '
                            f'(module: {len(module_online)}, sum of balance: {sum(balance_online)})')

        if len(module_target) != sum(balance_target):
            raise BalanceError('target module and sum of balance have different length '
                            f'(module: {len(module_target)}, sum of balance: {sum(balance_target)})')

        if any(x <= 0 for x in balance_online):
            raise BalanceError(f'all balance numbers must be positive integer (balance: {balance_online})')
        if any(x <= 0 for x in balance_target):
            raise BalanceError(f'all balance numbers must be positive integer (balance: {balance_target})')


        def split(module_children: Iterable[torch.nn.Module], balance: List[int]) -> List[torch.nn.Sequential]:
            j = 0
            partitions = []
            layers: OrderedDict[str, torch.nn.Module] = OrderedDict()
            for name, layer in module_children:
                layer_input: torch.nn.Module = layer

                if len(layers) == 0:
                    # make this layer as leaf
                    for param in layer_input.parameters():
                        param.detach_()
                        param.requires_grad = True
                        assert param.is_leaf
                    
                layers[name] = layer_input

                if len(layers) == balance[j]:
                    # Group buffered layers as a partition.
                    partition = torch.nn.Sequential(layers)
                    partitions.append(partition)

                    # Prepare for the next partition.
                    layers.clear()
                    j += 1
            print([len(part) for part in partitions])
            return cast(List[torch.nn.Sequential], partitions)

        if module_predictor is not None:
            partition_online = split(chain(module_online.named_children(), module_predictor.named_children()), balance_online)
        else:
            partition_online = split(module_online.named_children(), balance_online)
        partition_target = split(module_target.named_children(), balance_target)

        return partition_online, partition_target
