
from typing import Dict, Generator, Iterable, List, Optional

from tspipe.batch_ops import BatchQueue


class PipelineSchedule():
    # (i, j, prev_partition, is_target, batch_idx, view_idx, dependencies, model_update_ver, backprop)
    def __init__(self, legacy_tuple=None, terminate=False):
        if legacy_tuple is not None:
            i, j, prev_partition, is_target, batch_idx, view_idx, dependencies, model_update_ver, backprop, optimize \
                = legacy_tuple

            self.i = i
            self.j = j
            self.prev_partition = prev_partition
            self.is_target = is_target
            self.batch_idx = batch_idx
            self.view_idx = view_idx
            self.dependencies = dependencies
            self.model_update_ver = model_update_ver
            self.backprop = backprop
            self.optimize = optimize
        self.terminate = terminate

    def get_tuple(self):
        return (
            self.i,
            self.j,
            self.prev_partition,
            self.is_target,
            self.batch_idx,
            self.view_idx,
            self.dependencies,
            self.model_update_ver,
            self.backprop,
            self.optimize)

    @property
    def ubatch_idx(self):
        return self.i

    @property
    def partition_idx(self):
        return self.j

    def __repr__(self) -> str:
        return f"PipelineSchedule<{self.__dict__}>"


def schedule_without_pipeline(devices, batch_index, optimize_len, gpipe_num_ubatch=None) \
        -> Iterable[Iterable[Optional[PipelineSchedule]]]:
    # (i, j, prev_partition, is_target, batch_idx, view_idx, dependencies, model_update_ver, backprop)

    num_devices = devices
    # num_batches = int((3 * (num_devices - 1) - optimize_len) / 2)
    if gpipe_num_ubatch is not None:
        num_batches = gpipe_num_ubatch
    else:
        num_batches = num_devices - 1

    schedule = []
    micro_idx = [0 for i in range(num_devices)]
    is_target = [0 for i in range(num_devices)]
    view_idx = [0 for i in range(num_devices)]
    forward = [True for i in range(num_devices)]
    # Forward
    for bat_id in range(4 * num_batches + (num_devices - 1)):
        sched = []
        for dev_id in range(num_devices):
            if micro_idx[dev_id] >= num_batches:
                micro_idx[dev_id] -= num_batches
            prev = dev_id - 1 if dev_id > 0 else None
            if is_target[dev_id] >= num_batches * 4:
                is_target[dev_id] -= 4 * num_batches
                forward[dev_id] = False
            is_target_bool = is_target[dev_id] < 2 * num_batches
            if view_idx[dev_id] >= num_batches * 2:
                view_idx[dev_id] -= 2 * num_batches
            view_idx_int = 1 if view_idx[dev_id] >= num_batches else 0
            if micro_idx[dev_id] == 0 and view_idx_int == 0:
                model_update_ver = batch_index - 1
            else:
                model_update_ver = None
            if dev_id <= bat_id and forward[dev_id]:
                #            (i,         j,     prev_partition, is_target,      batch_idx,   view_idx, dependencies, model_update_ver, backprop, optimize) # noqa :E203
                block = tuple([micro_idx[dev_id], dev_id, prev, is_target_bool, batch_index, view_idx_int, None, model_update_ver, False, False]) # noqa :E203
                micro_idx[dev_id] = micro_idx[dev_id] + 1
                view_idx[dev_id] = view_idx[dev_id] + 1
                is_target[dev_id] = is_target[dev_id] + 1
            else:
                block = None
            sched.append(PipelineSchedule(block) if block is not None else None)
        schedule.append(sched)
    grad_micro_idx = [num_batches - 1 for i in range(num_devices)]
    grad_view_idx = [num_batches * 2 - 1 for i in range(num_devices)]
    backward_start = num_devices - 1

    # Backward
    for bat_id in range(2 * num_batches + (num_devices - 1) + 1):
        sched = []
        for dev_id in range(num_devices):
            if grad_micro_idx[dev_id] < 0:
                grad_micro_idx[dev_id] += num_batches
            prev = dev_id + 1 if dev_id < num_devices - 1 else None
            grad_view_idx_int = 1 if grad_view_idx[dev_id] >= num_batches else 0
            if dev_id >= backward_start and grad_view_idx[dev_id] >= 0:
                block = tuple([grad_micro_idx[dev_id], dev_id, prev, False, batch_index,
                               grad_view_idx_int, None, None, True, False])
                grad_micro_idx[dev_id] = grad_micro_idx[dev_id] - 1
                grad_view_idx[dev_id] = grad_view_idx[dev_id] - 1
            else:
                if grad_view_idx[dev_id] == -1:
                    block = tuple([0, dev_id, None, False, batch_index, 0, None, None, False, True])
                    grad_view_idx[dev_id] -= 1
                else:
                    block = None
            sched.append(PipelineSchedule(block) if block is not None else None)
        schedule.append(sched)
        backward_start -= 1
    return schedule


def schedule_start(devices, batch_index, micro_idx, is_target, view_idx, optimize_len, skip=False) \
        -> Iterable[Iterable[Optional[PipelineSchedule]]]:
    num_devices = devices
    num_batches = num_devices - 1
    start = []
    for bat_id in range(2 * num_batches):
        schedule = []
        for dev_id in range(num_devices):

            if micro_idx[dev_id] >= num_batches:
                micro_idx[dev_id] -= num_batches

            prev = dev_id - 1 if dev_id > 0 else None

            if is_target[dev_id] >= num_batches * 4:
                is_target[dev_id] -= num_batches * 4

            is_target_bool = is_target[dev_id] < 2 * num_batches

            if view_idx[dev_id] >= num_batches * 2:
                view_idx[dev_id] -= num_batches * 2

            view_idx_int = 1 if view_idx[dev_id] >= num_batches else 0
            if not skip:
                if micro_idx[dev_id] == 0 and view_idx_int == 0 and is_target_bool:
                    model_update_ver = batch_index - 1
                else:
                    model_update_ver = None
            else:
                model_update_ver = None

            if dev_id <= bat_id:
                block = tuple([micro_idx[dev_id], dev_id, prev, is_target_bool, batch_index, view_idx_int, None,
                               model_update_ver, False, False])
                micro_idx[dev_id] = micro_idx[dev_id] + 1
                view_idx[dev_id] = view_idx[dev_id] + 1
                is_target[dev_id] = is_target[dev_id] + 1
            else:
                block = None
            schedule.append(PipelineSchedule(block) if block is not None else None)
        start.append(schedule)
    return start


def schedule_repeat(devices, batch_index, micro_idx, is_target, view_idx, skip_target=False, skip_online=False) \
        -> Iterable[Iterable[Optional[PipelineSchedule]]]:
    # (i, j, prev_partition, is_target, batch_idx, view_idx, dependencies, model_update_ver, backprop)
    num_devices = devices
    num_batches = num_devices - 1
    repeat = []

    backprop = [-1 for i in range(num_devices)]
    propagate = [False for i in range(num_devices)]
    grad_view_idx = [0 for i in range(num_devices)]
    grad_micro_idx = [num_batches - 1 for i in range(num_devices)]
    batch_count = []
    for i in range(num_devices):
        batch_count.append(-i)

    for i in range(num_batches * 6):
        schedule = []
        dependency = None
        for dev_id in range(num_devices):
            if backprop[dev_id] == -1:

                if micro_idx[dev_id] >= num_batches:
                    micro_idx[dev_id] -= num_batches

                # previous Partitions
                prev = dev_id - 1 if dev_id > 0 else None

                if is_target[dev_id] >= num_batches * 4:
                    is_target[dev_id] -= num_batches * 4

                is_target_bool = is_target[dev_id] < 2 * num_batches

                if view_idx[dev_id] >= num_batches * 2:
                    view_idx[dev_id] -= num_batches * 2

                view_idx_int = 1 if view_idx[dev_id] >= num_batches else 0

                # Dependency
                if dev_id == num_devices - 1 and is_target[dev_id] == 4*num_batches - 1:
                    dependency = batch_index
                    backprop[dev_id] = 2*num_batches - 1

                # Model update Version
                if skip_target:
                    if micro_idx[dev_id] == 0 and not is_target_bool and batch_count[dev_id] == 0:
                        model_update_ver = batch_index - 1
                    else:
                        model_update_ver = None
                elif skip_online:
                    if micro_idx[dev_id] == 0 and is_target_bool and batch_count[dev_id] == 2*num_batches:
                        model_update_ver = batch_index - 1
                    else:
                        model_update_ver = None
                else:
                    if micro_idx[dev_id] == 0 and batch_count[dev_id] == 0:
                        model_update_ver = batch_index - 2
                    elif micro_idx[dev_id] == 0 and batch_count[dev_id] == 2*num_batches:
                        model_update_ver = batch_index - 1
                    else:
                        model_update_ver = None

                if batch_count[dev_id] >= 2*num_batches:
                    block = tuple([micro_idx[dev_id], dev_id, prev, is_target_bool, batch_index + 1,
                                   view_idx_int, dependency, model_update_ver, False])
                else:
                    block = tuple([micro_idx[dev_id], dev_id, prev, is_target_bool, batch_index,
                                   view_idx_int, dependency, model_update_ver, False])

                micro_idx[dev_id] = micro_idx[dev_id] + 1
                view_idx[dev_id] = view_idx[dev_id] + 1
                is_target[dev_id] = is_target[dev_id] + 1
                batch_count[dev_id] = batch_count[dev_id] + 1
            else:

                if grad_micro_idx[dev_id] < 0:
                    grad_micro_idx[dev_id] += num_batches

                prev = dev_id + 1 if dev_id < num_devices - 1 else None

                if dev_id > 0 and not propagate[dev_id]:
                    backprop[dev_id - 1] = 2*num_batches - 1
                    propagate[dev_id] = True

                grad_view_idx_int = 1 if grad_view_idx[dev_id] < num_batches else 0

                block = tuple([grad_micro_idx[dev_id], dev_id, prev, False, batch_index,
                               grad_view_idx_int, None, None, True])

                backprop[dev_id] = backprop[dev_id] - 1
                grad_micro_idx[dev_id] = grad_micro_idx[dev_id] - 1
                grad_view_idx[dev_id] = grad_view_idx[dev_id] + 1

            schedule.append(PipelineSchedule(block) if block is not None else None)
        repeat.append(schedule)
    return repeat


def new_schedule_start(devices, batch_index, micro_idx, is_target, view_idx, optimize, batch_count, optimize_len):
    num_devices = devices
    num_batches = num_devices - 1
    repeat = []

    backprop = [-1 for i in range(num_devices)]
    propagate = [0 for i in range(num_devices)]
    grad_view_idx = [0 for i in range(num_devices)]
    grad_micro_idx = [2 * num_batches - 1 for i in range(num_devices)]
    for bat_id in range(8 * num_batches + optimize_len):
        schedule = []
        dependency = None
        for dev_id in range(num_devices):
            if backprop[dev_id] == -1:

                if optimize[dev_id] > 0:

                    if optimize[dev_id] == optimize_len:
                        block = tuple([0, dev_id, 0, False, batch_index, 0, None, None, False, True])
                    optimize[dev_id] -= 1
                else:
                    if micro_idx[dev_id] >= num_batches:
                        micro_idx[dev_id] -= num_batches

                    # previous Partitions
                    prev = dev_id - 1 if dev_id > 0 else None

                    if is_target[dev_id] >= num_batches * 4:
                        is_target[dev_id] -= num_batches * 4

                    is_target_bool = is_target[dev_id] < 2 * num_batches

                    if view_idx[dev_id] >= num_batches * 2:
                        view_idx[dev_id] -= num_batches * 2

                    view_idx_int = 1 if view_idx[dev_id] >= num_batches else 0

                    # Dependency
                    if dev_id == num_devices - 1 and is_target[dev_id] == 4*num_batches - 1:
                        dependency = batch_index
                        backprop[dev_id] = 4*num_batches - 1

                    # Model Update Version
                    if micro_idx[dev_id] == 0 and batch_count[dev_id] % (2*num_batches) == 0:
                        model_update_ver = batch_index - 1
                    else:
                        model_update_ver = None

                    if batch_count[dev_id] >= 2*num_batches:
                        block = tuple([micro_idx[dev_id], dev_id, prev, is_target_bool, batch_index + 1,
                                       view_idx_int, dependency, model_update_ver, False, False])
                    else:
                        block = tuple([micro_idx[dev_id], dev_id, prev, is_target_bool, batch_index,
                                       view_idx_int, dependency, model_update_ver, False, False])

                    micro_idx[dev_id] = micro_idx[dev_id] + 1
                    view_idx[dev_id] = view_idx[dev_id] + 1
                    is_target[dev_id] = is_target[dev_id] + 1
                    batch_count[dev_id] = batch_count[dev_id] + 1

                    if batch_count[dev_id] >= 4*num_batches and optimize[dev_id] == 0:
                        optimize[dev_id] = optimize_len
                        batch_count[dev_id] = 0
            else:
                if grad_micro_idx[dev_id] < 0:
                    grad_micro_idx[dev_id] += 2 * num_batches

                prev = dev_id + 1 if dev_id < num_devices - 1 else None

                if grad_view_idx[dev_id] >= num_batches * 4:
                    grad_view_idx[dev_id] = 0

                if dev_id > 0 and propagate[dev_id] < 1:
                    propagate[dev_id] += 1
                    if propagate[dev_id] == 1:
                        backprop[dev_id - 1] = 4*num_batches - 1

                grad_view_idx_int = 1 if grad_view_idx[dev_id] < 2 * num_batches else 0

                block = tuple([grad_micro_idx[dev_id], dev_id, prev, False, batch_index,
                               grad_view_idx_int, None, None, True, False])

                backprop[dev_id] = backprop[dev_id] - 1
                grad_micro_idx[dev_id] = grad_micro_idx[dev_id] - 1
                grad_view_idx[dev_id] = grad_view_idx[dev_id] + 1

            schedule.append(PipelineSchedule(block) if block is not None else None)
        repeat.append(schedule)
    return repeat


def new_schedule_repeat(devices, batch_index, micro_idx, is_target, view_idx, optimize, batch_count, optimize_len):
    num_devices = devices
    num_batches = num_devices - 1
    repeat = []

    backprop = [-1 for i in range(num_devices)]
    propagate = [0 for i in range(num_devices)]
    grad_view_idx = [0 for i in range(num_devices)]
    grad_micro_idx = [2 * num_batches - 1 for i in range(num_devices)]
    reset = False
    for bat_id in range(8 * num_batches + optimize_len):
        schedule = []
        dependency = None
        for dev_id in range(num_devices):
            if backprop[dev_id] == -1:
                if optimize[dev_id] > 0:
                    if optimize[dev_id] == optimize_len:
                        if reset:
                            block = tuple([0, dev_id, 0, False, batch_index, 0, None, None, False, True])
                        else:
                            block = tuple([0, dev_id, 0, False, batch_index - 1, 0, None, None, False, True])
                    else:
                        block = None
                    optimize[dev_id] -= 1
                else:
                    if micro_idx[dev_id] >= num_batches:
                        micro_idx[dev_id] -= num_batches
                        if dev_id == num_devices - 1:
                            reset = True

                    # previous Partitions
                    prev = dev_id - 1 if dev_id > 0 else None

                    if is_target[dev_id] >= num_batches * 4:
                        is_target[dev_id] -= num_batches * 4

                    is_target_bool = is_target[dev_id] < 2 * num_batches

                    if view_idx[dev_id] >= num_batches * 2:
                        view_idx[dev_id] -= num_batches * 2

                    view_idx_int = 1 if view_idx[dev_id] >= num_batches else 0

                    # Dependency
                    if dev_id == num_devices - 1 and is_target[dev_id] == 4*num_batches - 1:
                        dependency = batch_index
                        backprop[dev_id] = 4*num_batches - 1

                    # Model Update Ver
                    if micro_idx[dev_id] == 0 and batch_count[dev_id] % (2*num_batches) == 0:
                        model_update_ver = batch_index - 1
                    else:
                        model_update_ver = None

                    if batch_count[dev_id] >= 2*num_batches and reset:
                        block = tuple([micro_idx[dev_id], dev_id, prev, is_target_bool, batch_index + 1,
                                       view_idx_int, dependency, model_update_ver, False, False])
                    else:
                        block = tuple([micro_idx[dev_id], dev_id, prev, is_target_bool, batch_index,
                                       view_idx_int, dependency, model_update_ver, False, False])

                    micro_idx[dev_id] = micro_idx[dev_id] + 1
                    view_idx[dev_id] = view_idx[dev_id] + 1
                    is_target[dev_id] = is_target[dev_id] + 1
                    batch_count[dev_id] = batch_count[dev_id] + 1

                    if batch_count[dev_id] >= 4*num_batches and optimize[dev_id] == 0:
                        optimize[dev_id] = optimize_len
                        batch_count[dev_id] = 0
            else:
                if grad_micro_idx[dev_id] < 0:
                    grad_micro_idx[dev_id] += 2 * num_batches

                prev = dev_id + 1 if dev_id < num_devices - 1 else None

                if grad_view_idx[dev_id] >= num_batches * 4:
                    grad_view_idx[dev_id] = 0

                if dev_id > 0 and propagate[dev_id] < 1:
                    propagate[dev_id] += 1
                    if propagate[dev_id] == 1:
                        backprop[dev_id - 1] = 4*num_batches - 1

                grad_view_idx_int = 1 if grad_view_idx[dev_id] < 2 * num_batches else 0

                block = tuple([grad_micro_idx[dev_id], dev_id, prev, False, batch_index,
                               grad_view_idx_int, None, None, True, False])

                backprop[dev_id] = backprop[dev_id] - 1
                grad_micro_idx[dev_id] = grad_micro_idx[dev_id] - 1
                grad_view_idx[dev_id] = grad_view_idx[dev_id] + 1

            schedule.append(PipelineSchedule(block) if block is not None else None)
        repeat.append(schedule)
    return repeat


def schedule_generator_mp(devices, batch_idx):
    n = devices
    m = 1
    batch_index = batch_idx
    # (i, j, prev_partition, is_target, batch_index,
    #  view_idx, dependencies, model_update_ver, backprop)
    repeat = []
    is_target = [1 for i in range(n)]
    view_idx = [0 for i in range(n)]
    batch_count = [0 for i in range(n)]
    grad_count = [0 for i in range(n)]
    propagate = [False for i in range(n)]
    back_view_idx = [1 for i in range(n)]
    forward = True
    for k in range(2*n + 4):
        schedule = []
        for j in range(n):
            if forward:
                prev = j - 1 if j > 0 else None

                if is_target[j] < - 2 * m:
                    is_target[j] = 1

                if view_idx[j] > 1:
                    view_idx[j] = 0

                if batch_count[j] == 4*m - 1 and j == n - 1:
                    forward = False
                    propagate[j] = True

                if batch_count[j] == 0:
                    model_update_ver = batch_index - 2
                elif batch_count[j] == 2 * m:
                    model_update_ver = batch_index - 2
                else:
                    model_update_ver = None

                if j <= k and batch_count[j] <= 4*m - 1:
                    block = tuple([0, j, prev, is_target[j] >= 0, batch_index, view_idx[j],
                                   None, model_update_ver, False])
                    is_target[j] -= 1
                    view_idx[j] += 1
                    batch_count[j] += 1
                else:
                    block = None
            else:
                prev = j + 1 if j < n - 1 else None

                if back_view_idx[j] > 1:
                    back_view_idx[j] = 0

                if propagate[j] and grad_count[j] < 2*m:
                    block = tuple([0, j, prev, False, batch_index, back_view_idx[j], None, None, True])
                    grad_count[j] += 1
                    if j != 0:
                        propagate[j-1] = True
                    back_view_idx[j] += 1
                else:
                    block = None
            schedule.append(block)
        repeat.append(PipelineSchedule(*schedule))
    return repeat


def sanity_check(sched_gen_func):
    def wrap(s):
        schedule_history = set()
        schedule_gen = sched_gen_func(s)
        for schedule in schedule_gen:
            for sched in schedule:
                if sched is None:
                    continue
                if sched in schedule_history:
                    raise ValueError(f"This task has been already scheduled : {sched}")
                else:
                    schedule_history.add(sched)
            yield schedule
    return wrap


class TSPipeScheduler():
    def __init__(self, config: Dict, input_batch_queue: BatchQueue, num_partitions: int):
        self.input_batch_queue = input_batch_queue
        self.num_partitions = num_partitions

        self.gpipe_emulation_enabled = config['gpipe_emulation']['enabled']
        if self.gpipe_emulation_enabled:
            self.num_skip_staleness = None
            self.gpipe_num_ubatch = config['gpipe_emulation']['num_ubatch']
        else:
            self.num_skip_staleness = config['optimizer']['num_skip_initial_staleness']
            self.gpipe_num_ubatch = None

    def throttle_until_batch_available(self, lst_sched: Iterable[Optional[PipelineSchedule]]):
        # throttle scheduling
        for subsched in lst_sched:
            if subsched is not None:
                self.input_batch_queue.wait_batch(subsched.batch_idx)

    @sanity_check
    def schedule_generator(self) -> Generator[List[Optional[PipelineSchedule]], None, None]:
        """Generates schedules for each clock cycle."""
        # (i, j, prev_partition, is_target, batch_idx, view_idx, dependencies, model_update_ver, backprop)
        n = num_devices = self.num_partitions
        optimize_len = 1

        schedule = None
        if self.gpipe_emulation_enabled:
            batch_idx = 0
            while True:
                batch_idx += 1
                schedule = schedule_without_pipeline(n, batch_idx, optimize_len, gpipe_num_ubatch=self.gpipe_num_ubatch)
                for sched in schedule:
                    self.throttle_until_batch_available(sched)
                    yield sched

        for batch_idx in range(1, self.num_skip_staleness + 1):
            schedule = schedule_without_pipeline(n, batch_idx, optimize_len)
            for sched in schedule:
                self.throttle_until_batch_available(sched)
                yield sched
        batch_index = self.num_skip_staleness + 1

        micro_idx, view_idx, is_target, optimize = [0] * n, [0] * n, [0] * n, [0] * n
        batch_count = []
        for i in range(num_devices):
            batch_count.append(-i)
        schedule = schedule_start(n, batch_index, micro_idx, is_target, view_idx, optimize_len)
        for sched in schedule:
            self.throttle_until_batch_available(sched)
            yield sched
        schedule = new_schedule_start(n, batch_index, micro_idx, is_target, view_idx,
                                      optimize, batch_count, optimize_len)
        for sched in schedule:
            self.throttle_until_batch_available(sched)
            yield sched
        batch_index += 1
        while True:
            schedule = new_schedule_repeat(n, batch_index, micro_idx, is_target, view_idx,
                                           optimize, batch_count, optimize_len)
            batch_index += 1
            for sched in schedule:
                self.throttle_until_batch_available(sched)
                yield sched


if __name__ == "__main__":
    n = 4
    schedule = 0
    for batch_idx in range(1, 3):
        print()
        schedule = schedule_without_pipeline(n, batch_idx)
    print()
    micro_idx = [0 for i in range(n)]
    view_idx = [0 for i in range(n)]
    is_target = [0 for i in range(n)]
    optimize = [0 for i in range(n)]

    schedule = schedule_start(n, 3, micro_idx, is_target, view_idx)
    print()
    schedule = new_schedule_start(n, 3, micro_idx, is_target, view_idx, optimize)
    print()
    schedule = new_schedule_repeat(n, 4, micro_idx, is_target, view_idx, optimize)
    print()
    schedule = new_schedule_repeat(n, 5, micro_idx, is_target, view_idx, optimize)
