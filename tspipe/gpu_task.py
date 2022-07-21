"""GPU Task"""
import time
from enum import Enum
from functools import partial
from itertools import chain
from queue import Empty
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import Tensor

from tspipe.batch_ops import Batch, BatchList, Microbatch, ScatterGatherFn
from tspipe.gpu_context import (GpuTaskContext, LocalTaskContext, ParamStorage,
                                StreamDescriptor, StreamType, TaskContext)
from tspipe.logger import Log
from tspipe.model_wrapper import TSPipeModelOutput
from tspipe.profiler import Operation, profile_semantic
from tspipe.utils import (find_path_tensor_requires_grad, get_bytes,
                          get_device, record_stream,
                          traverse_object_tensor_foreach, traverse_path_apply,
                          traverse_tensor_map, use_stream, wait_stream)

__all__ = ['TaskType', 'GpuTask', 'GpuTaskContext', 'StreamType', 'StreamDescriptor',
           'TaskContext', 'ParamStorage', 'slice_parameters']


class TaskType(Enum):
    TASK_COMPUTE_FORWARD = 1
    TASK_COMPUTE_BACKWARD = 2
    TASK_COMPUTE_LOSS = 3
    TASK_COPY_BATCH = 4
    TASK_COPY_MODEL = 5
    TASK_COPY_BATCH_OUT = 6
    TASK_COPY_GRAD = 8
    TASK_COPY_GRAD_OUT = 9
    TASK_FEED_BATCH = 11
    TASK_TERMINATE = 12
    TASK_COMPUTE_OPTIMIZE_GPU = 13
    TASK_FEED_TARGET = 14


def null_print(*args, **kwargs):
    pass


cond_debug_print = null_print  # Log.d
cond_info_print = null_print   # Log.d

global_tensor_uuid_count = 0


def issue_tensor_uuid() -> int:
    global global_tensor_uuid_count
    global_tensor_uuid_count += 1
    return global_tensor_uuid_count


def model_id_to_use(batch_id: int, is_target: bool, device_id: int,
                    num_skip_staleness: int, async_param_update_disable: bool) -> int:
    if num_skip_staleness is None or batch_id <= num_skip_staleness + 1:
        return max(batch_id - 1, 0)

    if is_target or async_param_update_disable:
        return max(batch_id - 2, num_skip_staleness)
    else:
        return max(batch_id - 1, num_skip_staleness)


def slice_parameters(given_dev_allocation: List[int], lst: List[Optional[Tensor]]) -> List[List[Optional[Tensor]]]:
    # do not modify given input
    device_allocation = [d for d in given_dev_allocation]

    result = [[] for _ in range(len(device_allocation))]
    dev_id = 0
    for tensor in lst:
        while device_allocation[0] == 0:
            device_allocation.pop(0)
            dev_id += 1
        if len(device_allocation) == 0:
            break

        result[dev_id].append(tensor)
        device_allocation[0] -= 1
        assert device_allocation[0] >= 0
    assert [len(p) for p in result] == given_dev_allocation
    return result


def compute_loss(ctx: 'GpuTaskContext', task: 'GpuTask'):
    with Operation('loss', task.batch_id, task.view_id, task.ubatch_id, task.is_target, None, task.partition_id):
        batch_online_view_1 = [None for _ in range(ctx.num_ubatch)]
        batch_target_view_2 = [None for _ in range(ctx.num_ubatch)]
        batch_online_view_2 = [None for _ in range(ctx.num_ubatch)]
        batch_target_view_1 = [None for _ in range(ctx.num_ubatch)]

        gradient_store: Dict[str, Tuple[int, Dict[Tuple, torch.Tensor]]] = {}
        num_bwd_ubatch = ctx.num_bwd_ubatch if not ctx.config['gpipe_emulation']['enabled'] and \
            task.batch_id > ctx.config['optimizer']['num_skip_initial_staleness'] else ctx.num_ubatch
        assert num_bwd_ubatch % ctx.num_ubatch == 0, "For now, ctx.num_bwd_ubatch should be muptiple of ctx.num_ubatch"

        ubatches_to_collect = (ctx.num_ubatch * 2) if task.asymmetric else (ctx.num_ubatch * 2 * 2)

        for _ in range(ubatches_to_collect):
            ubatch = ctx.queue_loss_compute_in.get()
            ctx.device_state.in_memory_ubatch_count -= 1
            ctx.device_state.in_memory_ubatch_bytes -= get_bytes(ubatch.data.value)
            assert ubatch.batch_id == task.batch_id

            batch = ubatch.data.detach()

            # register hook to collect gradient
            if not ubatch.is_target:
                bwd_multiple = num_bwd_ubatch // ctx.num_ubatch
                sc_ubatches = task.scatter_gather_fn.scatter(batch.value, bwd_multiple)

                for idx, sc_ubatch in enumerate(sc_ubatches):
                    bwd_ubatch_id = bwd_multiple * ubatch.ubatch_id + idx

                    def hook_save_gradient(grad: Tensor, batch_id: int, view_id: int, bwd_ubatch_id: int, path: Tuple,
                                           gradient_store=gradient_store):
                        key = f"{view_id}_{bwd_ubatch_id}"
                        if key not in gradient_store:
                            gradient_store[key] = (batch_id, {})
                        gradient_store[f"{view_id}_{bwd_ubatch_id}"][1][path] = grad

                    if isinstance(sc_ubatch.value, torch.Tensor):
                        sc_ubatch.value.requires_grad = True
                        sc_ubatch.value.register_hook(partial(hook_save_gradient,
                                                              batch_id=ubatch.batch_id, view_id=ubatch.view_id,
                                                              bwd_ubatch_id=bwd_ubatch_id, path=tuple()))
                    elif isinstance(sc_ubatch.value, TSPipeModelOutput):
                        def set_hook(t: torch.Tensor, path: Tuple, ubatch=ubatch):
                            if 'logits' in path:
                                assert not t.requires_grad and t.is_leaf
                                t.requires_grad = True
                                t.register_hook(partial(hook_save_gradient,
                                                        batch_id=ubatch.batch_id, view_id=ubatch.view_id,
                                                        bwd_ubatch_id=bwd_ubatch_id, path=path))
                        for path in sc_ubatch.value.lst_require_grad_tensor_path:
                            traverse_path_apply(sc_ubatch.value.data, path, partial(set_hook, path=path))
                    else:
                        raise ValueError("Models must be wrapped with TSPipeModelWrapper " +
                                         "if their output is not a Tensor.")
                    del hook_save_gradient

                # Scatter and Gather
                batch = Batch(task.scatter_gather_fn.gather(sc_ubatches))
                del sc_ubatch
                del sc_ubatches

            if ubatch.view_id == 0:
                if ubatch.is_target:
                    batch_target_view_1[ubatch.ubatch_id] = batch
                else:
                    batch_online_view_1[ubatch.ubatch_id] = batch
            else:
                if ubatch.is_target:
                    batch_target_view_2[ubatch.ubatch_id] = batch
                else:
                    batch_online_view_2[ubatch.ubatch_id] = batch

        del batch
        del ubatch
        ctx.device_state.print_mem_stat()

        if task.asymmetric:
            target_ubatch = ctx.queue_copy_label_in.get()
            assert target_ubatch.batch_id == task.batch_id

            batch_online_view_1 = task.scatter_gather_fn.gather(batch_online_view_1)
            batch_target_view_2 = task.scatter_gather_fn.gather(batch_target_view_2)

            if isinstance(batch_online_view_1, TSPipeModelOutput):
                batch_online_view_1 = batch_online_view_1.data
            if isinstance(batch_target_view_2, TSPipeModelOutput):
                batch_target_view_2 = batch_target_view_2.data

            loss: Tensor = ctx.loss_fn(batch_online_view_1, batch_target_view_2, target_ubatch.data.value,
                                       ctx.args, ctx.extra_args, task.epoch)
            ctx.queue_loss_out.put((task.batch_id, loss.detach().cpu().item()))
            Log.d("Epoch", task.epoch)

            del batch_online_view_1
            del batch_target_view_2
            del target_ubatch

        else:
            batch_online_view_1 = task.scatter_gather_fn.gather(batch_online_view_1)
            batch_target_view_2 = task.scatter_gather_fn.gather(batch_target_view_2)
            batch_online_view_2 = task.scatter_gather_fn.gather(batch_online_view_2)
            batch_target_view_1 = task.scatter_gather_fn.gather(batch_target_view_1)

            if isinstance(batch_online_view_1, TSPipeModelOutput):
                batch_online_view_1 = batch_online_view_1.data
            if isinstance(batch_target_view_2, TSPipeModelOutput):
                batch_target_view_2 = batch_target_view_2.data
            if isinstance(batch_online_view_2, TSPipeModelOutput):
                batch_online_view_2 = batch_online_view_2.data
            if isinstance(batch_target_view_1, TSPipeModelOutput):
                batch_target_view_1 = batch_target_view_1.data

            loss: Tensor = ctx.loss_fn(batch_online_view_1, batch_online_view_2,
                                       batch_target_view_1, batch_target_view_2, ctx.args, ctx.extra_args)
            ctx.queue_loss_out.put((task.batch_id, loss.detach().cpu().item()))

        # calculate backward for loss here
        stream = ctx.find_stream(task.src_stream)
        with use_stream(stream):
            stream.synchronize()
            # grad_tensors is set to None, as performing backward regarding scalar tensor assumes its gradient to 1.
            torch.autograd.backward(loss, grad_tensors=None)
        stream.synchronize()

        # enqueue in reversed order
        if task.asymmetric:
            view_lst = [0]
        else:
            view_lst = [1, 0]

        for view_id in view_lst:
            for bwd_ubatch_id in reversed(range(num_bwd_ubatch)):
                ubatch_grad = gradient_store[f"{view_id}_{bwd_ubatch_id}"]
                ctx.queue_grad_copy.put(BatchList(batch_id=ubatch_grad[0], grad=ubatch_grad[1]))
        del loss
        del gradient_store
        del ubatch_grad


def compute_backward(ctx: 'GpuTaskContext', task: 'GpuTask'):
    if ctx.num_ubatch > 1:
        activations = ctx.activation.pop(task.batch_id, task.view_id, task.ubatch_id)[()]
        ctx.device_state.in_memory_activation_bytes -= get_bytes(activations)
        bl_grad: BatchList = ctx.queue_grad_copy.get()
    else:
        bl_grad: BatchList = ctx.queue_grad_copy.get()
        activations = ctx.activation.pop(task.batch_id, task.view_id, task.ubatch_id)[()]
        ctx.device_state.in_memory_activation_bytes -= get_bytes(activations)

    activation = activations.data if isinstance(activations, TSPipeModelOutput) else activations

    assert task.batch_id == bl_grad.batch_id
    if isinstance(bl_grad.grad, dict):
        gradients = bl_grad.grad
    elif isinstance(bl_grad.grad, torch.Tensor):
        gradients = {tuple(): bl_grad.grad}

    with Operation('backward', task.batch_id, task.view_id, task.ubatch_id, task.is_target, None, task.partition_id):
        stream = ctx.find_stream(task.src_stream)
        tx_stream = ctx.find_stream(task.wait_stream)
        with use_stream(stream):
            stream.synchronize()
            # Reset grad for each param when necessary
            if ctx.num_ubatch == task.ubatch_id + 1 and (task.asymmetric or task.view_id == 1):
                for param in ctx.partition_online.parameters():
                    param.grad = None
            for path, gradient in gradients.items():
                def do_backward(activation: torch.Tensor, gradient: torch.Tensor):
                    torch.autograd.backward(activation, grad_tensors=gradient)
                traverse_path_apply(activation, path, partial(do_backward, gradient=gradient))
        stream.synchronize()
        tx_stream.synchronize()
        del gradient
        del gradients

        if ctx.gradient.has(task.batch_id, task.view_id, task.ubatch_id):
            output_grad = ctx.gradient.pop(task.batch_id, task.view_id, task.ubatch_id)
            for t in output_grad.values():
                ctx.device_state.in_memory_gradient_bytes -= get_bytes(t)
            assert all((not g.requires_grad) for g in output_grad.values())
            # emit the grad to next partition
            bl_output = BatchList(task.batch_id, grad=output_grad)
            del output_grad
        else:
            bl_output = BatchList(task.batch_id, None)

    if task.partition_id > 0:
        ctx.queue_grad_pending.put(bl_output)
        ctx.device_state.backward_complete_ubatch_count += 1

    del activations
    del activation
    del bl_output
    del bl_grad

    ctx.device_state.print_mem_stat()


def copy_grad_out(ctx: 'GpuTaskContext', task: 'GpuTask'):

    bl_output = ctx.queue_grad_pending.get()
    tx_stream = ctx.find_stream(task.wait_stream)
    if ctx.queue_grad_out is not None:
        with Operation('cp_tx_grad', task.batch_id, task.view_id,
                       task.ubatch_id, task.is_target, None, task.partition_id), \
             Operation('cp_rx_grad', task.batch_id, task.view_id,
                       task.ubatch_id, task.is_target, None, task.partition_id - 1):
            with use_stream(tx_stream):
                tx_stream.synchronize()
                ctx.queue_grad_out.put(bl_output)
            tx_stream.synchronize()


def copy_grad_out_condition(ctx: 'GpuTaskContext', task: 'GpuTask'):
    if not ctx.config['gpipe_emulation']['enabled']:
        return True

    if task.asymmetric:
        return True

    # should be updated when pipeline_schedule updates
    which_ubatch_am_i = ctx.num_ubatch * 2 - 1
    which_ubatch_am_i -= task.view_id * ctx.num_ubatch
    which_ubatch_am_i -= task.ubatch_id
    nth_chunk = which_ubatch_am_i // 2
    num_ubatch_per_batch = ctx.num_ubatch * 2
    if ctx.device_state.backward_complete_ubatch_count >= \
            num_ubatch_per_batch * (task.batch_id - 1) + (nth_chunk + 1) * 2:
        return True
    return False


def compute_optimize_gpu(ctx: 'GpuTaskContext', task: 'GpuTask'):

    def update_expected_online_model_id():
        ctx.device_state.expected_online_model_id = \
            model_id_to_use(
                task.batch_id + 1, task.is_target, task.device_id,
                ctx.config['optimizer']['num_skip_initial_staleness']
                if not ctx.config['gpipe_emulation']['enabled'] else None,
                ctx.config['async_param_update_emulation']['enabled'])
        cond_info_print(f"Update expected_online_model_id here! now {ctx.device_state.expected_online_model_id} {task}")

    # pre-process momentum / LR update
    if task.new_momentum is not None:
        Log.i(f"Momentum is updated at batch {task.batch_id}: {ctx.momentum} -> {task.new_momentum}")
        ctx.momentum = task.new_momentum
    if task.new_lr is not None:
        if isinstance(task.new_lr, list) or isinstance(task.new_lr, tuple):
            for param_group, lr in zip(ctx.optimizer.param_groups, task.new_lr):
                Log.v(f"LR is updated at batch {task.batch_id}: {param_group['lr']} -> {lr}")
                param_group['lr'] = lr
        else:
            for param_group in ctx.optimizer.param_groups:
                Log.v(f"LR is updated at batch {task.batch_id}: {param_group['lr']} -> {task.new_lr}")
                param_group['lr'] = task.new_lr

    with Operation('optimize', task.batch_id, task.view_id, task.ubatch_id, task.is_target, None, task.partition_id):
        online_prev_param = list(ctx.partition_online.parameters())
        target_prev_param = list(ctx.partition_target.parameters())

        if ctx.config['async_param_update_emulation']['enabled']:
            ctx.params_online.discard_below(task.batch_id - 2)
            ctx.params_target.discard_below(task.batch_id - 2)
        else:
            ctx.params_online.discard_below(task.batch_id - 1)
            ctx.params_target.discard_below(task.batch_id - 1)
        ctx.device_state.in_memory_model_bytes = \
            sum(get_bytes(t)
                for p in chain(ctx.params_online.storage.values(), ctx.params_target.storage.values())
                for t in p)

        # skip optimization on gradient accumulation / skip_optimizer is active
        should_skip_optimization = ctx.config['optimizer']['skip_optimizer'] or \
            ((ctx.num_grad_accumulation + 1) % ctx.config['optimizer']['gradient_accumulation'] != 0 and
             not task.optimizer_step)

        if should_skip_optimization:
            if ctx.config['optimizer']['gradient_accumulation'] > 1:
                ctx.num_grad_accumulation += 1
            else:
                # free out gradients, as we don't need this
                ctx.optimizer.zero_grad()

            ctx.params_online.push(task.batch_id, online_prev_param)
            ctx.params_target.push(task.batch_id, target_prev_param)
            ctx.device_state.in_memory_model_bytes = \
                sum(get_bytes(t)
                    for p in chain(ctx.params_online.storage.values(), ctx.params_target.storage.values())
                    for t in p)

            update_expected_online_model_id()
            return

        compute_stream = ctx.get_stream(StreamType.STREAM_DEFAULT_COMPUTE)
        with use_stream(compute_stream):
            # If using gradient accumulation, average out gradients
            if ctx.config['optimizer']['gradient_accumulation'] > 1:
                for pg in ctx.optimizer.param_groups:
                    param: torch.Tensor
                    for param in pg['params']:
                        if param.grad is not None:
                            param.grad /= (ctx.num_grad_accumulation + 1)  # average out gradient
                ctx.num_grad_accumulation = 0

            # step optimizer
            ctx.optimizer.step()
            ctx.optimizer.zero_grad()
            online_new_param = [t.detach().clone() for t in online_prev_param]

        compute_stream.synchronize()
        del online_prev_param  # not valid anymore

        if ctx.update_target_fn is None:
            target_new_param = target_prev_param
        else:
            target_new_param = ctx.update_target_fn(ctx.momentum, online_new_param, target_prev_param)
        assert all((p.is_cuda if p is not None else True) for p in chain(online_new_param, target_prev_param))

        ctx.params_online.push(task.batch_id, online_new_param)
        ctx.params_target.push(task.batch_id, target_new_param)
        ctx.device_state.in_memory_model_bytes = \
            sum(get_bytes(t)
                for p in chain(ctx.params_online.storage.values(), ctx.params_target.storage.values())
                for t in p)

        # Update expected_online_model_id here
        update_expected_online_model_id()


def update_target_model_id_after_forward_pass(ctx: 'GpuTaskContext', task: 'GpuTask'):
    if task.is_target and task.view_id == 1 and task.ubatch_id + 1 == ctx.num_ubatch:
        ctx.device_state.expected_target_model_id = \
            model_id_to_use(task.batch_id + 1, task.is_target, task.device_id,
                            ctx.config['optimizer']['num_skip_initial_staleness']
                            if not ctx.config['gpipe_emulation']['enabled'] else None,
                            ctx.config['async_param_update_emulation']['enabled'])
        cond_info_print("Update expected_target_model_id here! " +
                        f"now {ctx.device_state.expected_target_model_id} {task.batch_id}")


def compute_forward(ctx: 'GpuTaskContext', task: 'GpuTask'):
    # Pull previous compute results
    ubatch: Microbatch = ctx.queue_compute_in.get()
    ctx.device_state.in_memory_ubatch_bytes -= get_bytes(ubatch.data.value)
    if ubatch.data is None:
        return
    assert task.batch_id == ubatch.batch_id and task.ubatch_id == ubatch.ubatch_id and \
        task.is_target == ubatch.is_target and task.view_id == task.view_id, f"{task} must match {ubatch}"
    batch = ubatch.data

    model_ver = ctx.device_state.current_target_model_id if task.is_target else ctx.device_state.current_online_model_id
    cond_info_print(f"Computing forward for batch_id {task.batch_id} target={task.is_target} model_ver={model_ver}")
    if batch.atomic and batch.tensor.device.index != ctx.cb_partition_id_to_internal_gpu_id(task.partition_id):
        assert False, f"Warning: Synchronization Error - Device mismatch {batch.tensor.device} != {task.partition_id}"

    num_bwd_ubatch = ctx.num_bwd_ubatch \
        if not ctx.config['gpipe_emulation']['enabled'] and \
        task.batch_id > ctx.config['optimizer']['num_skip_initial_staleness'] else ctx.num_ubatch
    assert num_bwd_ubatch % ctx.num_ubatch == 0
    bwd_batch_split = num_bwd_ubatch // ctx.num_ubatch

    def hook_save_gradient(grad: Tensor, nano_idx: int, path: Tuple):
        num_multiplier = num_bwd_ubatch // ctx.num_ubatch
        bwd_ubatch_id = task.ubatch_id * num_multiplier + nano_idx
        ubatch = Microbatch(task.batch_id, task.view_id, bwd_ubatch_id, task.is_target, Batch(grad))
        ctx.gradient.push(ubatch, path)
        ctx.device_state.in_memory_gradient_bytes += get_bytes(ubatch.data.value)

    # check validity for atomic batch
    if batch.atomic:
        assert batch.tensor.device.index == task.device_id
        if task.is_target:
            assert all(t.device.index == task.device_id for t in ctx.partition_target.parameters())
        else:
            assert all(t.device.index == task.device_id for t in ctx.partition_online.parameters())

    compute_stream = ctx.get_stream(StreamType.STREAM_DEFAULT_COMPUTE)
    copy_dst_stream = ctx.get_stream(StreamType.STREAM_COPY_BATCH_TO)
    wait_stream(compute_stream, copy_dst_stream)

    profile_semantic(task.batch_id, task.view_id, task.ubatch_id, task.is_target, None, task.partition_id, 'compute')
    lst_activation_batch = []
    with use_stream(compute_stream):
        batch_lst = task.scatter_gather_fn.scatter(batch.tensor_or_tensors, bwd_batch_split)
        for nano_idx, b in enumerate(batch_lst):
            if task.is_target:
                with torch.no_grad():
                    batch_out = b.call(ctx.partition_target)
            else:
                def set_grad(t: torch.Tensor, path: Optional[Tuple]):
                    if t.is_floating_point() or t.is_complex():
                        t.requires_grad = True
                        t.register_hook(partial(hook_save_gradient, nano_idx=nano_idx, path=path))

                if isinstance(b.value, torch.Tensor):
                    set_grad(b.value, tuple())
                elif isinstance(b.value, TSPipeModelOutput):
                    for path in b.value.lst_require_grad_tensor_path:
                        traverse_path_apply(b.value.data, path, partial(set_grad, path=path))
                else:
                    assert task.partition_id == 0, \
                        "Models must be wrapped with TSPipeModelWrapper if their output is not a Tensor."
                    traverse_object_tensor_foreach(b.value, set_grad)

                batch_out = b.call(ctx.partition_online)

                if isinstance(batch_out.value, torch.Tensor):
                    pass
                elif isinstance(batch_out.value, TSPipeModelOutput):
                    # prepare lst_require_grad_tensor_path for forward op at the next gpu
                    batch_out.value.lst_require_grad_tensor_path = find_path_tensor_requires_grad(batch_out.value.data)
                else:
                    raise ValueError("Models must be wrapped with TSPipeModelWrapper if their output is not a Tensor.")

            lst_activation_batch.append(batch_out)
            del batch_out
        compute_stream.synchronize()
        del batch_lst
        del b
    del batch
    profile_semantic(task.batch_id, task.view_id, task.ubatch_id, task.is_target, None, task.partition_id,
                     'compute_finish')

    result_ubatch = Microbatch(task.batch_id, task.view_id, task.ubatch_id, task.is_target,
                               Batch(task.scatter_gather_fn.gather(lst_activation_batch)))
    ctx.device_state.in_memory_ubatch_bytes += get_bytes(result_ubatch.data.value)
    ctx.queue_compute_out.put(result_ubatch)

    if not task.is_target:
        for idx, sc_batch in enumerate(lst_activation_batch):
            bwd_ubatch_id = task.ubatch_id * bwd_batch_split + idx
            activation = Microbatch(task.batch_id, task.view_id, bwd_ubatch_id, task.is_target, sc_batch)
            ctx.activation.push(activation)
            ctx.device_state.in_memory_activation_bytes += get_bytes(activation.data.value)
    del lst_activation_batch
    ctx.device_state.print_mem_stat()
    # for last partition, queue_compute_out will directly send ubatch to loss calculation
    if task.partition_id == ctx.num_partitions - 1:
        update_target_model_id_after_forward_pass(ctx, task)

    # for emulating gpipe
    ctx.device_state.forward_complete_ubatch_count += 1


def compute_forward_condition(ctx: 'GpuTaskContext', task: 'GpuTask'):
    num_skip_initial_staleness = ctx.config['optimizer']['num_skip_initial_staleness'] \
        if not ctx.config['gpipe_emulation']['enabled'] else None
    if task.is_target:
        return ctx.device_state.current_target_model_id == \
            model_id_to_use(task.batch_id, task.is_target, task.device_id, num_skip_initial_staleness,
                            ctx.config['async_param_update_emulation']['enabled'])
    else:
        return ctx.device_state.current_online_model_id == \
            model_id_to_use(task.batch_id, task.is_target, task.device_id, num_skip_initial_staleness,
                            ctx.config['async_param_update_emulation']['enabled'])


def feed_batch(ctx: 'LocalTaskContext', task: 'GpuTask'):

    with Operation('feed_batch', task.batch_id, task.view_id, task.ubatch_id, task.is_target, None, task.partition_id):
        batches_list = task.batch_list

        if task.asymmetric:
            for view_idx in [1, 0]:
                batch_to_feed: List[Batch] = batches_list[view_idx]
                for _idx in range(ctx.num_ubatch):
                    ubatch = Microbatch(task.batch_id, view_idx, _idx, view_idx == 1, batch_to_feed[_idx])
                    ctx.lst_queue_batch_feed_out[_idx].put(ubatch)
            ubatch_label = Microbatch(task.batch_id, 0, 0, False, task.label_batch)
            ctx.queue_label_feed_out.put(ubatch_label)

        else:
            def target_idx(is_target, view_idx):
                if is_target:
                    return 3 if view_idx == 0 else 1
                else:
                    return 0 if view_idx == 0 else 2

            for is_target in [True, False]:
                for view_idx in [0, 1]:
                    batch_to_feed: List[Batch] = batches_list[target_idx(is_target, view_idx)]
                    for _idx in range(ctx.num_ubatch):
                        assert batch_to_feed[_idx].tensor.is_shared() or batch_to_feed[_idx].tensor.is_pinned()
                        ubatch = Microbatch(task.batch_id, view_idx, _idx, is_target, batch_to_feed[_idx])
                        ctx.lst_queue_batch_feed_out[_idx].put(ubatch)


def copy_batch_out(ctx: 'GpuTaskContext', task: 'GpuTask'):
    ubatch = ctx.queue_compute_out.get()
    # NCCL Backend will start copying here
    with Operation('cp_rx_batch', task.batch_id, task.view_id,
                   task.ubatch_id, task.is_target, None, task.partition_id+1),\
         Operation('cp_tx_batch', task.batch_id, task.view_id,
                   task.ubatch_id, task.is_target, None, task.partition_id):
        ctx.device_state.in_memory_ubatch_count -= 1
        ctx.device_state.in_memory_ubatch_bytes -= get_bytes(ubatch.data.value)
        ctx.queue_copy_curr_out.put(ubatch)  # TODO: Possible blocking here
        ctx.device_state.print_mem_stat()

    update_target_model_id_after_forward_pass(ctx, task)


def copy_batch_out_condition(ctx: 'GpuTaskContext', task: 'GpuTask'):
    if not ctx.config['gpipe_emulation']['enabled']:
        return True

    if task.asymmetric:
        which_ubatch_am_i = 0 if task.is_target else (ctx.num_ubatch)
        which_ubatch_am_i += task.ubatch_id
        nth_chunk = which_ubatch_am_i // 2
        num_ubatch_per_batch = ctx.num_ubatch * 2

        if ctx.device_state.forward_complete_ubatch_count >= \
                num_ubatch_per_batch * (task.batch_id - 1) + (nth_chunk + 1) * 2:
            return True
        return False

    # should be updated when pipeline_schedule updates
    which_ubatch_am_i = 0 if task.is_target else (ctx.num_ubatch * 2)
    which_ubatch_am_i += task.view_id * ctx.num_ubatch
    which_ubatch_am_i += task.ubatch_id
    nth_chunk = which_ubatch_am_i // 4
    num_ubatch_per_batch = (ctx.num_ubatch * 4)
    if ctx.device_state.forward_complete_ubatch_count >= \
            num_ubatch_per_batch * (task.batch_id - 1) + (nth_chunk + 1) * 4:
        return True
    return False


def copy_batch_condition(ctx: 'GpuTaskContext', task: 'GpuTask'):
    if task.partition_id == ctx.num_partitions - 1:
        return True

    if ctx.config['gpipe_emulation']['enabled']:
        return True

    return ctx.device_state.in_memory_ubatch_count < 2


def copy_batch(ctx: 'GpuTaskContext', task: 'GpuTask'):
    if task.partition_id != 0:
        ubatch_curr = ctx.queue_copy_curr_in.get()
    else:
        try:
            ubatch_curr = ctx.queue_copy_curr_in.get_nowait()
        except Empty:
            wait_start = time.time()
            ubatch_curr = ctx.queue_copy_curr_in.get()
            Log.d(f"Waited for new ubatch for {(time.time() - wait_start)*1000:.3} ms..")
    assert ubatch_curr.data is not None
    assert task.batch_id == ubatch_curr.batch_id and task.ubatch_id == ubatch_curr.ubatch_id

    gpu_rx_stream = ctx.get_stream(StreamType.STREAM_COPY_BATCH_TO)
    gpu_compute_stream = ctx.get_stream(StreamType.STREAM_DEFAULT_COMPUTE)

    # implement tensor traversal
    input_ubatch_val = ubatch_curr.data.value
    cpu_to_gpu_copy = False

    def _internal_tensor_copy(t: torch.Tensor):
        if t.is_cuda:
            # this was already copied to me using NCCL backend!
            ctx.device_state.in_memory_ubatch_bytes += get_bytes(t)
            return t
        else:
            # CPU -> GPU copy here
            with use_stream(gpu_rx_stream):
                gpu_rx_stream.synchronize()
                x = t
                y = x.detach().to(get_device(gpu_rx_stream))
            record_stream(y, gpu_compute_stream)
            wait_stream(gpu_compute_stream, gpu_rx_stream)
            gpu_rx_stream.synchronize()
            output = y
            ctx.device_state.in_memory_ubatch_bytes += get_bytes(output)
            assert not y.requires_grad, "Output should not require grad!"
            return output.detach()

    output = traverse_tensor_map(input_ubatch_val, _internal_tensor_copy)
    new_ubatch = Microbatch(ubatch_curr.batch_id, ubatch_curr.view_id, ubatch_curr.ubatch_id, ubatch_curr.is_target,
                            Batch(output))
    ctx.queue_copy_out.put(new_ubatch)
    if cpu_to_gpu_copy:
        profile_semantic(task.batch_id, task.view_id, task.ubatch_id, task.is_target, None, task.partition_id,
                         'cp_rx_batch_finish')

    ctx.device_state.in_memory_ubatch_count += 1
    ctx.device_state.print_mem_stat()


def copy_grad(ctx: 'GpuTaskContext', task: 'GpuTask'):
    """Dequeue from `ctx.queue_grad_in`, copy tensor to `task.dst_stream`, and Enqueue to `ctx.queue_grad_out`.
    """
    # this was already copied to me using NCCL backend!
    batch_lst: BatchList = ctx.queue_grad_in.get()
    assert task.batch_id == batch_lst.batch_id
    ctx.queue_grad_copy.put(batch_lst)


def copy_model_condition(ctx: 'GpuTaskContext', task: 'GpuTask'):
    cond_debug_print(f"CopyModelCondition Check task={task} " +
                     f"expected_target_model_id={ctx.device_state.expected_target_model_id} " +
                     f"expected_online_model_id={ctx.device_state.expected_online_model_id} ")
    num_skip_initial_staleness = ctx.config['optimizer']['num_skip_initial_staleness'] \
        if not ctx.config['gpipe_emulation']['enabled'] else None
    if task.is_target:
        model_id = model_id_to_use(task.batch_id, task.is_target, task.device_id, num_skip_initial_staleness,
                                   ctx.config['async_param_update_emulation']['enabled'])
        return ctx.device_state.expected_target_model_id == model_id and ctx.params_target.has(model_id)
    else:
        model_id = model_id_to_use(task.batch_id, task.is_target, task.device_id, num_skip_initial_staleness,
                                   ctx.config['async_param_update_emulation']['enabled'])
        return ctx.device_state.expected_online_model_id == model_id and ctx.params_online.has(model_id)


def copy_model(ctx: 'GpuTaskContext', task: 'GpuTask'):
    if task.is_target:
        model_batch_id = ctx.device_state.expected_target_model_id
        parameters: Optional[Iterable[Tensor]] = ctx.params_target.peek(model_batch_id)
    else:
        model_batch_id = ctx.device_state.expected_online_model_id
        parameters: Optional[Iterable[Tensor]] = ctx.params_online.peek(model_batch_id)

    model = ctx.partition_target if task.is_target else ctx.partition_online

    if parameters is not None:
        src_stream = ctx.get_stream(StreamType.STREAM_COPY_BATCH_TO)
        profile_semantic(task.batch_id, task.view_id, task.ubatch_id, task.is_target, None,
                         task.partition_id, 'update_model')
        with use_stream(src_stream):
            for param, new_param in zip(model.parameters(), parameters):
                with torch.no_grad():
                    param.copy_(new_param, non_blocking=True)
        src_stream.synchronize()

    if ctx.config['train']['save_model_every_iter'] > 0:
        if task.batch_id % ctx.config['train']['save_model_every_iter'] == 0 and task.is_target:
            Log.i(f"Saving model at batch {task.batch_id}")
            destination = f"{ctx.config['__artifact_dir']}/model_batch{task.batch_id}_part{task.partition_id}.pt"
            torch.save({
                'online_network_state_dict': ctx.partition_online.state_dict(),
                'target_network_state_dict': model.state_dict(),
                'optimizer_state_dict': ctx.optimizer.state_dict()
            }, destination)

    if task.is_target:
        ctx.device_state.current_target_model_id = model_batch_id
        # print(f"current_target_model_id is now {model_batch_id}")
    else:
        ctx.device_state.current_online_model_id = model_batch_id
        # print(f"current_online_model_id is now {model_batch_id}")
    if parameters is not None:
        profile_semantic(task.batch_id, task.view_id, task.ubatch_id, task.is_target, None, task.partition_id,
                         'update_model_finish')
    del parameters


def terminate(ctx: 'GpuTaskContext', task: 'GpuTask'):
    Log.i(f"Terminating this {ctx.worker}")


class GpuTask():
    def __init__(self, task_type: TaskType, batch_id: int, view_id: Optional[int], ubatch_id: Optional[int],
                 partition_id: Optional[int], is_target: bool, dst_stream: Optional[StreamDescriptor] = None,
                 src_stream: Optional[StreamDescriptor] = None, wait_stream: Optional[StreamDescriptor] = None,
                 new_model_ver: Optional[Any] = None, device_id: Optional[int] = None,
                 batch_list: Optional[Iterable[Iterable[Batch]]] = None, optimizer_step: Optional[bool] = None,
                 new_lr: Optional[Union[float, List[float]]] = None, new_momentum: Optional[float] = None,
                 asymmetric: Optional[bool] = False, label_batch: Optional[Iterable[Batch]] = None,
                 epoch: Optional[int] = None, scatter_gather_fn: Optional[ScatterGatherFn] = None):
        self.task_type = task_type
        self.batch_id = batch_id
        self.view_id = view_id
        self.ubatch_id = ubatch_id
        self.partition_id = partition_id
        self.is_target = is_target
        self.dst_stream = dst_stream
        self.src_stream = src_stream
        self.wait_stream = wait_stream
        self.new_model_ver = new_model_ver
        self.device_id = device_id

        self.scheduled = False
        self.completed = False
        self.worker = None
        self.batch_list = batch_list
        self.optimizer_step = optimizer_step
        self.new_lr = new_lr
        self.new_momentum = new_momentum
        self.asymmetric = asymmetric
        self.label_batch = label_batch
        self.epoch = epoch
        self.scatter_gather_fn = scatter_gather_fn

    def __repr__(self):
        return f"Task<{self.task_type.name} batch_id={self.batch_id} view_id={self.view_id} " + \
               f"ubatch_id={self.ubatch_id} partition_id={self.partition_id} is_target={self.is_target}" + \
               f"{f' new_lr={self.new_lr}' if self.new_lr else ''}" + \
               f"{f' new_momentum={self.new_momentum}' if self.new_momentum else ''}>"

    def schedule(self, worker):
        # Scheduler thread will call this
        assert not self.scheduled, "Cannot re-schedule already scheduled task."
        self.scheduled = True

        if worker.device is not None:
            self.device_id = worker.device.index
        else:
            self.device_id = -1

    # (Callback, Condition_Callback, Stashable)
    task_fn_map = {
        TaskType.TASK_COMPUTE_FORWARD:  (compute_forward,   compute_forward_condition),  # noqa: E202
        TaskType.TASK_COMPUTE_BACKWARD: (compute_backward,  None                     ),  # noqa: E202
        TaskType.TASK_COMPUTE_LOSS:     (compute_loss,      None                     ),  # noqa: E202
        TaskType.TASK_COPY_BATCH:       (copy_batch,        copy_batch_condition     ),  # noqa: E202
        TaskType.TASK_COPY_BATCH_OUT:   (copy_batch_out,    copy_batch_out_condition ),  # noqa: E202
        TaskType.TASK_COPY_MODEL:       (copy_model,        copy_model_condition     ),  # noqa: E202
        TaskType.TASK_COPY_GRAD:        (copy_grad,         None                     ),  # noqa: E202
        TaskType.TASK_COPY_GRAD_OUT:    (copy_grad_out, copy_grad_out_condition  ),      # noqa: E202
        TaskType.TASK_FEED_BATCH:       (feed_batch,        None                     ),  # noqa: E202
        TaskType.TASK_TERMINATE:        (terminate, None),
        TaskType.TASK_COMPUTE_OPTIMIZE_GPU: (compute_optimize_gpu,  None             ),  # noqa: E202
    }

    def check_precondition(self, ctx: Union[GpuTaskContext, LocalTaskContext]) -> bool:
        precondition: Callable = GpuTask.task_fn_map[self.task_type][1]
        if precondition is None:
            return True
        return precondition(ctx, self)

    def run(self, ctx: GpuTaskContext):
        return GpuTask.task_fn_map[self.task_type][0](ctx, self)

    @staticmethod
    def dict_key(batch_id: int, view_id: int, ubatch_id: int, partition_id: int, is_target: bool) -> str:
        return '_'.join(str(a) for a in [batch_id, view_id, ubatch_id, partition_id, is_target])

    @property
    def key(self) -> str:
        return GpuTask.dict_key(self.batch_id, self.view_id, self.ubatch_id, self.partition_id, self.is_target)

    def __eq__(self, o: object) -> bool:
        return self.key == o.key
