import builtins
import heapq
import threading
import warnings
from collections import OrderedDict
from enum import Enum
from queue import Queue as LocalQueue
from time import sleep
from typing import (Callable, DefaultDict, Dict, Generic, List, Optional,
                    Tuple, TypeVar)

import torch
import torch.distributed as dist
from torch.distributed import rpc

from tspipe.logger import Log
from tspipe.utils import use_device

PYTORCH_DISTRIBUTED_NCCL_PORT = 31101
PYTORCH_DISTRIBUTED_RPC_PORT = 31102


class TensorPlaceholder:
    def __init__(self, idx: int, tensor: torch.Tensor, src_rank: int, dtype: torch.dtype, checksum: int):
        self.i = idx
        self.s = tuple(tensor.shape)
        self.r = src_rank
        self.d = dtype
        self.c = checksum

    @property
    def shape(self) -> Tuple:
        return self.s

    @property
    def tensor_src_rank(self) -> int:
        return self.r

    @property
    def dtype(self) -> int:
        return self.d

    @property
    def idx(self) -> int:
        return self.i

    @property
    def checksum(self) -> int:
        return self.c

    def __repr__(self) -> str:
        return f"<TensorPlaceholder sz={self.shape} rank={self.tensor_src_rank} " + \
               f"dtype={self.dtype} checksum={self.checksum}>"


def object_decode(obj: object, cls, callback: Callable):
    def isinstance_namedtuple(obj) -> bool:
        return (
                isinstance(obj, tuple) and
                hasattr(obj, '_asdict') and
                hasattr(obj, '_fields')
        )
    if isinstance(obj, list):
        return [object_decode(itm, cls, callback) for itm in obj]
    if isinstance_namedtuple(obj):
        return obj.__class__._make(object_decode(itm, cls, callback) for itm in obj)
    if isinstance(obj, OrderedDict):
        return OrderedDict((k, object_decode(v, cls, callback)) for k, v in obj.items())
    if isinstance(obj, dict):
        return dict((k, object_decode(v, cls, callback)) for k, v in obj.items())
    if isinstance(obj, tuple):
        return tuple(object_decode(itm, cls, callback) for itm in obj)
    if isinstance(obj, cls):
        return callback(obj)
    if obj is None or type(obj).__name__ in dir(builtins) or isinstance(obj, Enum) or isinstance(obj, torch.dtype):
        return obj

    for key, value in obj.__dict__.items():
        obj.__dict__[key] = object_decode(value, cls, callback)

    return obj


def marshalize_object(rank: int, obj, device):
    lst_tensors = []
    lst_cksums = []
    num_tensors = [0]

    def encode_tensors(x: torch.Tensor):
        if not x.is_cuda:
            return x
        assert device is None or x.device == device
        tph = TensorPlaceholder(num_tensors[0], x, rank, x.dtype, 0)
        lst_tensors.append(x.detach())
        lst_cksums.append(tph.checksum)
        num_tensors[0] += 1
        return tph

    return object_decode(obj, torch.Tensor, encode_tensors), lst_tensors, lst_cksums


class Agreement:
    def __init__(self, src_rank: int, lst_packet_seq: List[int]):
        self.src_rank = src_rank
        self.lst_packet_seq = lst_packet_seq


class Packet:
    def __init__(self, channel=str, src_rank=None, seq_no=None, payload=None, nccl_seq_no=None, num_tensors=0):
        self.channel = channel
        self.s = src_rank
        self.p = payload
        self.t = num_tensors
        self.i = seq_no
        self.j = nccl_seq_no
        self.ls: Optional[List[int]] = None

    @property
    def payload(self):
        return self.p

    @property
    def num_tensors(self):
        return self.t

    @property
    def src_rank(self):
        return self.s

    @property
    def seq_no(self):
        return self.i

    @property
    def nccl_seq_no(self):
        return self.j

    def set_nccl_seq_no(self, seq):
        self.j = seq

    def set_checksum_list(self, lst: List[int]):
        self.ls = lst

    def __repr__(self) -> str:
        return f"{__class__.__name__} <src_rank={self.src_rank} num_tensors={self.num_tensors} " + \
               f"seq_no={self.seq_no} nccl_seq_no={self.nccl_seq_no}>"

    def set_payload(self, payload):
        self.p = payload

    def set_num_tensors(self, num_tensors):
        self.t = num_tensors

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, Packet):
            return False
        return self.seq_no == o.seq_no

    def __gt__(self, o: object) -> bool:
        if not isinstance(o, Packet):
            return False
        return self.seq_no > o.seq_no

    def __lt__(self, o: object) -> bool:
        if not isinstance(o, Packet):
            return False
        return self.seq_no < o.seq_no


class CommunicatorParam:
    def __init__(self, ip: str, world_size: int, rank: int, num_partition: Optional[int] = None):
        self.ip = ip
        self.world_size = world_size
        self.rank = rank

        self._num_part = self.world_size - 1 if num_partition is None else num_partition

    @property
    def num_partition(self):
        return self._num_part


class NcclTask:
    def __init__(self, packet: Packet, lst_tensor: Optional[List[torch.Tensor]] = None):
        self.packet = packet
        self.lst_tensor = lst_tensor
        self.tensor_idx: Optional[int] = None
        self.lst_tph: Optional[List[TensorPlaceholder]] = None

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, NcclTask):
            return False
        return self.packet.nccl_seq_no == o.packet.nccl_seq_no

    def __gt__(self, o: object) -> bool:
        if not isinstance(o, NcclTask):
            return False
        return self.packet.nccl_seq_no > o.packet.nccl_seq_no

    def __lt__(self, o: object) -> bool:
        if not isinstance(o, NcclTask):
            return False
        return self.packet.nccl_seq_no < o.packet.nccl_seq_no

    def __repr__(self) -> str:
        len_tensor = len(self.lst_tensor) if self.lst_tensor is not None else None
        return f"<NcclTask tensor={self.tensor_idx}/{len_tensor} packet={self.packet}>"


V = TypeVar('V')


class RecvQueue(Generic[V]):
    def __init__(self):
        self.rcv_nxt = 1
        self.q: List[V] = []

    def head(self) -> V:
        return self.q[0]

    def put(self, packet: V):
        heapq.heappush(self.q, packet)

    def get(self):
        return heapq.heappop(self.q)

    def __len__(self):
        return len(self.q)


class NcclNodeContext:
    def __init__(self, comm: 'Communicator'):
        self.comm = comm
        self.reordering_queue: RecvQueue[NcclTask] = RecvQueue()

        self.snd_nxt = 1
        self.nccl_lock = threading.Lock()
        self.send_cond = threading.Condition()
        self.agreement_q: LocalQueue[Agreement] = LocalQueue()
        self.recv_meta_q: LocalQueue[NcclTask] = LocalQueue()
        self.send_tensor_map: Dict[int, List[torch.Tensor]] = {}

    def enqueueTx(self, channel: 'Channel', obj):

        # generate NcclTask
        encoded, lst_tensors, lst_cksums = marshalize_object(self.comm.rank, obj, self.comm.device)

        # to ensure ordering of packet and tensor
        with self.send_cond:
            with channel.send_lock:
                packet = Packet(channel.name, self.comm.rank, channel.snd_nxt, encoded)
                channel.snd_nxt += 1
            packet.set_num_tensors(len(lst_tensors))
            packet.set_checksum_list(lst_cksums)
            ncclTask = NcclTask(packet, lst_tensors)

            if len(ncclTask.lst_tensor) == 0:
                return rpc.rpc_async(f'node{channel.dst_rank}', Communicator.rpc_enqueue_msg, (ncclTask.packet,))

            # print("Waiting for enqueueTx")
            # with self.send_cond:
            # print("Okay to enqueueTx")
            ncclTask.packet.set_nccl_seq_no(self.snd_nxt)
            # print(f"setting nccl no as {self.snd_nxt}")
            self.send_tensor_map[self.snd_nxt] = ncclTask.lst_tensor
            self.snd_nxt += 1
            return rpc.rpc_async(f'node{channel.dst_rank}', Communicator.rpc_enqueue_msg, (ncclTask.packet,))

    def routeRxPacket(self, ncclTask: NcclTask):
        packet = ncclTask.packet

        # dummy message
        if packet.channel == '':
            # print("Discarding dummy tensors")
            return

        if packet.channel not in self.comm.channel:
            self.comm.unknown_channel_messages[packet.channel].append(packet)
            return

        self.comm.channel[packet.channel].enque_recv(packet)

    def enqueueRx(self, ncclTask: NcclTask):
        if ncclTask.packet.num_tensors == 0:
            self.routeRxPacket(ncclTask)
            return

        with self.send_cond:
            ooo_q = self.reordering_queue
            added = False

            assert ncclTask.packet.num_tensors > 0

            if ooo_q.rcv_nxt == ncclTask.packet.nccl_seq_no:
                self.recv_meta_q.put(ncclTask)
                ooo_q.rcv_nxt += 1
                added = True

                while len(ooo_q) > 0 and ooo_q.rcv_nxt == ooo_q.head().packet.nccl_seq_no:
                    nt = ooo_q.get()
                    self.recv_meta_q.put(nt)
                    ooo_q.rcv_nxt += 1
            else:
                ooo_q.put(ncclTask)

            if added:
                self.send_cond.notify()


class Channel:
    def __init__(self, name: str, dst_rank: Optional[int] = None):
        self.name = name
        self.dst_rank = dst_rank
        self.snd_nxt = 1
        self.recv_q = LocalQueue()
        self.recv_lock = threading.Lock()
        self.send_lock = threading.Lock()
        self.ooo_q: RecvQueue[Packet] = RecvQueue()

    def __repr__(self) -> str:
        return f"<Channel name={self.name}>"

    def send(self, comm: 'Communicator', obj):
        assert self.dst_rank is not None, "Cannot send to read-only queue"

        if comm.nccl_per_node_ctx is None or self.dst_rank not in comm.nccl_per_node_ctx:
            # is channel without NCCL transport
            encoded, lst_tensors, _ = marshalize_object(comm.rank, obj, 1)
            assert len(lst_tensors) == 0, f"{self} can only be used to transfer python objects and non-GPU tensors!"
            with self.send_lock:
                packet = Packet(self.name, comm.rank, self.snd_nxt, encoded)
                self.snd_nxt += 1
            return rpc.rpc_async(f'node{self.dst_rank}', Communicator.rpc_enqueue_msg, (packet,))

        else:
            ctx = comm.nccl_per_node_ctx[self.dst_rank]
            ctx.enqueueTx(self, obj)

    def enque_recv(self, packet: Packet):

        # Handle in-channel reordering
        with self.recv_lock:
            if self.ooo_q.rcv_nxt == packet.seq_no:
                self.recv_q.put(packet.payload)
                self.ooo_q.rcv_nxt += 1
                while len(self.ooo_q) > 0 and self.ooo_q.rcv_nxt == self.ooo_q.head().seq_no:
                    p: Packet = self.ooo_q.get()
                    self.recv_q.put(p.payload)
                    self.ooo_q.rcv_nxt += 1
                    del p
            else:
                self.ooo_q.put(packet)

    def recv(self, timeout=None):
        return self.recv_q.get(timeout=timeout)

    def recv_nowait(self):
        return self.recv_q.get_nowait()


class MuxChannel(Channel):
    def __init__(self, *channels: Channel):
        self.channels = channels
        self.recv_q = LocalQueue()

        for channel in channels:
            channel.recv_q = self.recv_q

    def send(self, comm: 'Communicator', obj):
        raise NotImplementedError

    def enque_recv(self, packet: Packet):
        raise NotImplementedError


class Communicator:
    __instance = None

    @classmethod
    def __getInstance(cls):
        return cls.__instance

    @classmethod
    def instance(cls, *args, **kargs):
        cls.__instance = cls(*args, **kargs)
        cls.instance = cls.__getInstance
        return cls.__instance

    @property
    def rank(self):
        return self.comm_param.rank

    @property
    def world_size(self):
        return self.comm_param.world_size

    def __init__(self, device: Optional[torch.device], comm_param: CommunicatorParam):

        Communicator.__instance = self
        Communicator.instance = Communicator.__getInstance

        self.device = device
        self.running = True
        self.comm_param = comm_param
        ip, rank, world_size = comm_param.ip, comm_param.rank, comm_param.world_size

        assert world_size is not None
        assert rank is not None

        assert (rank < self.comm_param.num_partition and device is not None) or \
               (rank >= self.comm_param.num_partition and device is None)

        self.channel: Dict[str, Channel] = {}

        self.rank_process_group_map: Dict[int, dist.ProcessGroupNCCL] = {}

        self.is_ready = False
        self.is_ready_cond = threading.Condition()

        self.nccl_per_node_ctx: Optional[Dict[int, NcclNodeContext]] = None

        nccl_eligible = self.device is not None and rank < self.comm_param.num_partition
        self.lst_threads_send_nccl: List[threading.Thread] = []
        self.lst_threads_recv_nccl: List[threading.Thread] = []
        if nccl_eligible:
            Log.v(f"Running communicator with device {device}")
            torch.cuda.set_device(self.device)
            nccl_init_method = f'tcp://{ip}:{PYTORCH_DISTRIBUTED_NCCL_PORT}'
            Log.v(f"Initializing process group at {nccl_init_method}")

            if rank == 0:
                Log.v("Waiting for secondary nodes to attach...")

            dist.init_process_group(backend="nccl", init_method=nccl_init_method,
                                    rank=rank, world_size=self.comm_param.num_partition)

            self.nccl_per_node_ctx: Dict[int, NcclNodeContext] = {}
            self.lst_dst_ranks = []

            def create_nccl_threads(r: int):
                self.lst_dst_ranks.append(r)
                self.nccl_per_node_ctx[r] = NcclNodeContext(self)
                thd = threading.Thread(target=self._thread_sender_nccl, args=(r, ))
                thd.name = f'NcclSenderThread{r}'
                self.lst_threads_send_nccl.append(thd)
                thd = threading.Thread(target=self._thread_receiver_nccl, args=(r, ))
                thd.name = f'NcclReceiverThread{r}'
                self.lst_threads_recv_nccl.append(thd)

            if rank > 0:
                create_nccl_threads(rank - 1)
            if rank + 1 < self.comm_param.num_partition:
                create_nccl_threads(rank + 1)

            for thd in [*self.lst_threads_send_nccl, *self.lst_threads_recv_nccl]:
                thd.start()

            Log.v("NCCL Init Okay.")

        rpc_init_method = f"tcp://{ip}:{PYTORCH_DISTRIBUTED_RPC_PORT}"
        Log.v(f"Initializing RPC with {rpc_init_method} at node {rank}")
        rpc.init_rpc(f'node{rank}', rank=rank, world_size=world_size,
                     rpc_backend_options=rpc.TensorPipeRpcBackendOptions(init_method=rpc_init_method,
                                                                         _transports=['shm', 'uv']))

        self.unknown_channel_messages = DefaultDict(list)
        Log.v("RPC Init Okay.")

        if device is not None:
            # Send dummy tensor on specified device
            torch.rand((1,), device=device)
            Log.v("Pytorch CUDA is ready.")

    def wait_ready(self):
        for i in range(self.scheduler_process_rank):
            rpc.rpc_sync(f'node{i}', Communicator.rpc_wait_ready)
            Log.v(f"RPC Checked {i} is Ready.")

    def mark_ready(self):
        with self.is_ready_cond:
            self.is_ready = True
            self.is_ready_cond.notify_all()
        Log.v(f"RPC Marking node {self.rank} as Ready.")

    @staticmethod
    def rpc_wait_ready():
        comm: 'Communicator' = Communicator.instance()
        Log.v(f"RPC Wait ready: rank:{comm.rank}")
        with comm.is_ready_cond:
            Log.v("Checking!")
            while not comm.is_ready:
                Log.v("Wait!")
                comm.is_ready_cond.wait()
                Log.v("Wakeup!")
        Log.v(f"RPC is now Ready for rank:{comm.rank}")

    def barrier(self):
        if self.device is None:
            warnings.warn("Calling barrier for Non-NCCL is no-op.")
            return

        print("Entering Barrier")
        dist.barrier()
        print("Leaving Barrier")

    def finish(self, wait=True):
        self.running = False
        print("Finish call")
        if self.nccl_per_node_ctx is not None:
            for ctx in self.nccl_per_node_ctx.values():
                ctx.agreement_q.put(None)
                ctx.recv_meta_q.put(None)
                with ctx.send_cond:
                    ctx.send_cond.notify()

        for thread in [*self.lst_threads_recv_nccl, *self.lst_threads_send_nccl]:
            thread.join()
        if wait:
            print("Finish wait")
            if self.device is not None:
                dist.barrier()
        print("Finish OK")

        # dirty
        sleep(5)

    channel_name_set = set()

    def create_channel(self, channel_name: str, dest_rank: int) -> Channel:
        '''Create Channel/LocalQueue'''
        assert channel_name not in Communicator.channel_name_set
        assert self.rank != dest_rank
        assert dest_rank >= 0
        Communicator.channel_name_set.add(channel_name)

        self.channel[channel_name] = chn = Channel(channel_name, dest_rank)

        if channel_name in self.unknown_channel_messages:
            print("Flushing cached messages")
            for message in self.unknown_channel_messages[channel_name]:
                chn.enque_recv(message)
            del self.unknown_channel_messages[channel_name]
        return chn

    def create_channel_mux(self, channel_name: str, *channels: Channel) -> MuxChannel:
        assert channel_name not in Communicator.channel_name_set
        Communicator.channel_name_set.add(channel_name)
        self.channel[channel_name] = chn = MuxChannel(*channels)
        return chn

    def get_channel(self, channel_name: str) -> Optional[Channel]:
        if channel_name in self.channel:
            return self.channel[channel_name]

    def is_channel_valid(self, channel_name: str):
        return channel_name in Communicator.channel_name_set

    def send(self, channel_name: str, obj, async_: bool = True):
        '''Block here: API'''
        chn = self.channel[channel_name]
        return chn.send(self, obj)

    def recv(self, channel: str, timeout=None):
        '''Block here: API'''
        return self.channel[channel].recv(timeout=timeout)

    def recv_nowait(self, channel: str):
        return self.channel[channel].recv_nowait()

    def _thread_receiver_nccl(self, src_dst_rank: int):
        assert src_dst_rank != self.rank
        is_slave = src_dst_rank > self.rank
        ctx = self.nccl_per_node_ctx[src_dst_rank]

        while self.running or not ctx.recv_meta_q.empty():
            with ctx.send_cond:
                if is_slave:
                    # wait until there are metadata to receive and no pending metadata to recv
                    while ctx.recv_meta_q.empty() or not len(ctx.send_tensor_map) == 0:
                        ctx.send_cond.wait()
                    # check proper synchronization
                    assert not ctx.recv_meta_q.empty()
                    assert len(ctx.send_tensor_map) == 0
                else:
                    # wait until there are metadata to receive
                    while ctx.recv_meta_q.empty():
                        ctx.send_cond.wait()
                    # check proper synchronization
                    assert not ctx.recv_meta_q.empty()

                halt = False
                nccl_rx_tasks: List[NcclTask] = []
                while not ctx.recv_meta_q.empty():
                    ncclTask = ctx.recv_meta_q.get()
                    if ncclTask is None:
                        halt = True
                        break
                    nccl_rx_tasks.append(ncclTask)
                    assert ncclTask.packet.nccl_seq_no is not None
                    del ncclTask  # to prevent memory leak
                if halt:
                    continue

                # send agreement with nccl_rx_tasks
                # print("Sending agreement for packet ", [rx_task.packet.nccl_seq_no for rx_task in nccl_rx_tasks])
                agreement = Agreement(self.rank, [rx_task.packet.nccl_seq_no for rx_task in nccl_rx_tasks])
                rpc.rpc_async(f'node{src_dst_rank}', Communicator.rpc_enqueue_agreement, (agreement,))

                if is_slave:
                    assert len(ctx.send_tensor_map) == 0

                # reserve space for reception
                for task_rx in nccl_rx_tasks:
                    lst_tensor = []
                    lst_tph = []

                    def resolve_tensors(ph: TensorPlaceholder):
                        assert self.rank != ph.tensor_src_rank
                        tensor = torch.empty(ph.shape, dtype=ph.dtype, device=self.device)
                        lst_tensor.append(tensor)
                        lst_tph.append(ph)
                        return tensor
                    task_rx.packet.set_payload(
                        object_decode(task_rx.packet.payload, TensorPlaceholder, resolve_tensors))
                    task_rx.lst_tensor = lst_tensor
                    task_rx.lst_tph = lst_tph

                    assert len(lst_tensor) > 0

                # issue ncclRecv and synchronize
                with ctx.nccl_lock:
                    if is_slave:
                        assert len(ctx.send_tensor_map) == 0

                    with use_device(ctx.comm.device):
                        for task_rx in nccl_rx_tasks:
                            for tensor in task_rx.lst_tensor:
                                # prevent GPU memory leak
                                dist.recv(tensor, src_dst_rank)
                            del tensor  # necessary to prevent GPU memory leak
                            torch.cuda.synchronize(self.device)

            for task_rx in nccl_rx_tasks:
                ctx.routeRxPacket(task_rx)

            # necessary to prevent GPU memory leak
            del task_rx
            del nccl_rx_tasks
            del lst_tensor
        print("Stopping thread _thread_receiver_nccl")

    def _thread_sender_nccl(self, src_dst_rank: int):
        ctx = self.nccl_per_node_ctx[src_dst_rank]
        while self.running or len(ctx.send_tensor_map) > 0 or not ctx.agreement_q.empty():
            agreement: Optional[Agreement] = ctx.agreement_q.get()
            if agreement is None:
                continue
            tensor_list: List[torch.Tensor] = []
            with ctx.send_cond:
                for seq_no in agreement.lst_packet_seq:
                    assert seq_no in ctx.send_tensor_map
                    tensor_list.extend(ctx.send_tensor_map[seq_no])

                with ctx.nccl_lock:
                    with use_device(ctx.comm.device):
                        for tensor in tensor_list:
                            # prevent GPU memory leak
                            dist.send(tensor, src_dst_rank)
                        del tensor
                        del tensor_list
                        torch.cuda.synchronize(self.device)

                for seq_no in agreement.lst_packet_seq:
                    assert seq_no in ctx.send_tensor_map
                    ctx.send_tensor_map.pop(seq_no)
                ctx.send_cond.notify()

        print("Stopping thread _thread_sender_nccl")

    @staticmethod
    def rpc_enqueue_msg(message: Packet):
        comm: 'Communicator' = Communicator.instance()

        if comm.nccl_per_node_ctx is None or message.src_rank not in comm.nccl_per_node_ctx:
            # directly routing here
            if message.channel not in comm.channel:
                comm.unknown_channel_messages[message.channel].append(message)
                return
            comm.channel[message.channel].enque_recv(message)
        else:
            comm.nccl_per_node_ctx[message.src_rank].enqueueRx(NcclTask(message))

    @staticmethod
    def rpc_enqueue_agreement(message: Agreement):
        comm: 'Communicator' = Communicator.instance()
        comm.nccl_per_node_ctx[message.src_rank].agreement_q.put(message)

    @property
    def scheduler_process_rank(self):
        return self.world_size - 1


V = TypeVar('V')


class DistributedQueue(Generic[V]):

    def __init__(self, from_rank: Optional[int], to_rank: Optional[int], channel_name: str):
        self.from_rank = from_rank
        self.to_rank = to_rank
        self.comm = Communicator.instance()

        my_rank = self.comm.rank
        other_rank = to_rank if from_rank == my_rank else from_rank

        # N-1 Queue and 1-N Queues have NoneType from_rank or to_rank
        assert self.comm.is_channel_valid(channel_name), f"Channel {channel_name} is not valid."

        if other_rank is not None:
            assert (from_rank == my_rank or to_rank == my_rank) and (from_rank != to_rank)
            assert self.comm.channel[channel_name].dst_rank == other_rank, \
                   f"Channel {channel_name} points to rank {self.comm.channel[channel_name].dst_rank}, " + \
                   f"but tries to send to {other_rank}"

        self.channel_name = channel_name

    def get(self, timeout=None) -> V:
        return self.comm.recv(self.channel_name, timeout=timeout)

    def get_nowait(self) -> V:
        return self.comm.recv_nowait(self.channel_name)

    def put(self, obj) -> None:
        self.comm.send(self.channel_name, obj)
