import torch.multiprocessing as torch_mp

mp_ctx = torch_mp.get_context("spawn")

Queue = mp_ctx.Queue
Process = mp_ctx.Process
