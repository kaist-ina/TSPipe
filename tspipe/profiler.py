from threading import Thread
from time import sleep, time_ns
from typing import Optional

from tspipe.communicator import DistributedQueue
from tspipe.logger import Log
from tspipe.multiprocessing import Queue

__all__ = ['TSPipeProfiler', 'profile_semantic', 'profile_init']


def timestamp():
    return time_ns() // 1_000_000


class TSPipeProfilerContainer:
    def __init__(self, profiler: Optional['TSPipeProfiler'] = None):
        self.profiler = profiler


current_profiler_container: Optional['TSPipeProfilerContainer'] = TSPipeProfilerContainer()
is_master_process = False


class TSPipeProfiler:
    def __init__(self, filename):
        # self.lock = Lock()
        self.filename = filename
        self.running = True
        self.profiler_message_queue = Queue()
        self.f = None
        self.thd = None

    def __enter__(self):
        global is_master_process
        current_profiler_container.profiler = self
        print("Profiler Activated.")
        is_master_process = True

    def __exit__(self, type, value, trace_back):
        if self.f is not None:
            self.f.close()
        current_profiler_container.profiler = None
        self.running = False
        if self.thd is not None:
            self.thd.join()

    def profile_semantic(self, ts, batch_idx, view_idx, ubatch_idx, is_target, src_partition, dst_partition, op_type):
        return ",".join(str(t) for t in [
            ts, batch_idx, view_idx, ubatch_idx, is_target, src_partition, dst_partition, op_type])


class RemoteTSPipeProfiler(TSPipeProfiler):
    def __init__(self, remote_queue):
        self.profiler_message_queue = remote_queue


def profile_semantic(*args):
    if current_profiler_container.profiler is not None:
        current_profiler_container.profiler.profiler_message_queue.put([timestamp(), *args])


def profile_inject(r):
    if current_profiler_container.profiler is not None:
        current_profiler_container.profiler.profiler_message_queue.put(r)


def profile_init():
    if current_profiler_container.profiler is not None:
        if is_master_process and current_profiler_container.profiler.f is None:
            self = current_profiler_container.profiler
            self.f = open(self.filename, 'w')

            def log_saver():
                Log.v("Waiting for logs...")
                while self.running:
                    # i = 0
                    while not self.profiler_message_queue.empty():
                        a = self.profiler_message_queue.get()
                        ts, args, kwargs = a[0], a[1:], {}
                        s = self.profile_semantic(ts, *args, **kwargs)
                        self.f.write(s+"\n")
                    self.f.flush()
                    sleep(0.25)
            self.thd = Thread(target=log_saver, args=())
            self.thd.name = 'ProfilerThread'
            self.thd.start()
            Log.v("Logger Started.")


def remote_profile_init(remote_queue):
    current_profiler_container.profiler = RemoteTSPipeProfiler(remote_queue)


class ProfilerDelegateWorker:
    def __init__(self, queue: 'DistributedQueue'):
        self.queue = queue

        def _thread_main():
            log_out_queue = queue
            while True:
                r = log_out_queue.get()
                if r is None:
                    print("Terminating Log Delegate Queue")
                    break
                profile_inject(r)

        self.thread = Thread(target=_thread_main)
        self.thread.name = 'ThreadLogDelegate'
        self.thread.start()

    def join(self):
        self.queue.put(None)
        self.thread.join()


class Operation:
    def __init__(self,
                 op_name: str,
                 batch_idx: Optional[int] = None,
                 view_idx: Optional[int] = None,
                 ubatch_idx: Optional[int] = None,
                 is_target: Optional[bool] = None,
                 src_partition: Optional[int] = None,
                 dst_partition: Optional[int] = None):
        self.op_name = op_name
        self.batch_idx = batch_idx
        self.view_idx = view_idx
        self.ubatch_idx = ubatch_idx
        self.is_target = is_target
        self.src_partition = src_partition
        self.dst_partition = dst_partition
        self.start_ts = None
        self.end_ts = None

    def __enter__(self):
        self.start_ts = timestamp()
        if current_profiler_container.profiler is not None:
            current_profiler_container.profiler.profiler_message_queue.put([
                self.start_ts, self.batch_idx, self.view_idx, self.ubatch_idx, self.is_target,
                self.src_partition, self.dst_partition, self.op_name,
            ])

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_ts = timestamp()
        if current_profiler_container.profiler is not None:
            current_profiler_container.profiler.profiler_message_queue.put([
                self.end_ts, self.batch_idx, self.view_idx, self.ubatch_idx, self.is_target,
                self.src_partition, self.dst_partition, self.op_name+"_finish",
            ])

