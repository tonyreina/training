import os
import torch
import torch.distributed as dist


def is_distributed():
    world_size = 1
    local_rank = 0
    if 'LOCAL_RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])

    distributed_run = world_size > 1
    if distributed_run:
        init_distributed(world_size, local_rank)


def reduce_tensor(tensor, num_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    if rt.is_floating_point():
        rt = rt/num_gpus
    else:
        rt = rt//num_gpus
    return rt


def init_distributed(world_size, rank):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    if rank == 0:
        print("Initializing Distributed")

    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Initialize distributed communication
    dist.init_process_group(
        backend='nccl', init_method='env://',
        world_size=world_size, rank=rank
    )
    if rank == 0:
        print("Done initializing distributed")
