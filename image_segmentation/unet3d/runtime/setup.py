import os

import torch
import torch.distributed as dist
import numpy as np
import dllogger as logger
from dllogger import StdOutBackend, Verbosity, JSONStreamBackend


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_world_size():
    return int(os.environ.get('WORLD_SIZE', 1))


def reduce_tensor(tensor, num_gpus):
    if num_gpus > 1:
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.reduce_op.SUM)
        if rt.is_floating_point():
            rt = rt / num_gpus
        else:
            rt = rt // num_gpus
        return rt
    return tensor


def init_distributed(world_size, rank):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    if rank == 0:
        print("Initializing DDP")

    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Initialize distributed communication
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        # rank=rank
    )
    if rank == 0:
        print("DDP Initialized. World size", world_size)


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def get_logger(params):
    backends = []
    if is_main_process():
        backends += [StdOutBackend(Verbosity.VERBOSE)]
        if params.log_dir:
            backends += [JSONStreamBackend(Verbosity.VERBOSE, params.log_dir)]
    logger.init(backends=backends)
    return logger
