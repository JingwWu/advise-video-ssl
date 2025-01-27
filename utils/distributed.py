import os
import torch
import torch.distributed as dist

import pytorchvideo.layers.distributed as du
from pytorchvideo.layers.distributed import (
        cat_all_gather,
        get_local_process_group,
        get_local_rank,
        get_local_size,
        get_world_size,
        init_distributed_training as _init_distributed_training,
        )

import torch.multiprocessing as mp

def launch_job(cfg, func):
    func(cfg=cfg)


def distributed_init(cfg):
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    if 'SLURM_PROCID' in os.environ:
        mode = "multi-nodes"
        os.environ['RANK'] = os.environ['SLURM_PROCID']
        os.environ['LOCAL_RANK'] = os.environ['SLURM_LOCALID']
        os.environ['WORLD_SIZE'] = os.environ['SLURM_NTASKS']

        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])

        node_list = os.environ['SLURM_NODELIST']

        master_addr = os.environ["MASTER_ADDR"]
        master_port = os.environ["MASTER_PORT"]
        dist_url = f"tcp://{master_addr}:{master_port}"

        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(
                backend='nccl',
                init_method=dist_url,
                world_size=world_size,
                rank=rank,
                )
        torch.distributed.barrier()

        node_id = rank // cfg.NUM_GPUS
        _init_distributed_training(cfg.NUM_GPUS, node_id)

    else:
        torch.distributed.init_process_group('nccl', init_method='env://')
        rank = torch.distributed.get_rank()
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])

        master_addr = os.environ['MASTER_ADDR']
        master_port = os.environ['MASTER_PORT']
        torch.cuda.set_device(local_rank)

        # HACK: work with pytorchvideo.layers.distributed
        du._LOCAL_PROCESS_GROUP = torch.distributed.group.WORLD

        nnodes = world_size // cfg.NUM_GPUS
        mode = 'single-node' if nnodes <= 1 else 'multi-nodes'

    print(f"{mode}; rank = {rank} is initialized in {master_addr}:{master_port}; local_rank = {local_rank}", flush=True)

    # cfg.DEVICE = torch.device("cuda", local_rank)
    cfg.LOCAL_RANK = local_rank
    cfg.RANK = rank
    cfg.NUM_WORLDS = world_size


def get_rank():
    """
    Get the rank of the current process.
    """
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def all_reduce(tensors, average=True):
    """
    All reduce the provided tensors from all processes across machines.
    Args:
        tensors (list): tensors to perform all reduce across all processes in
        all machines.
        average (bool): scales the reduced tensor by the number of overall
        processes across all machines.
    """

    for tensor in tensors:
        dist.all_reduce(tensor, async_op=False)
    if average:
        world_size = dist.get_world_size()
        for tensor in tensors:
            tensor.mul_(1.0 / world_size)
    return tensors


def all_gather(tensors):
    """
    All gathers the provided tensors from all processes across machines.
    Args:
        tensors (list): tensors to perform all gather across all processes in
        all machines.
    """

    gather_list = []
    output_tensor = []
    world_size = dist.get_world_size()
    for tensor in tensors:
        tensor_placeholder = [
            torch.ones_like(tensor) for _ in range(world_size)
        ]
        dist.all_gather(tensor_placeholder, tensor, async_op=False)
        gather_list.append(tensor_placeholder)
    for gathered_tensor in gather_list:
        output_tensor.append(torch.cat(gathered_tensor, dim=0))
    return output_tensor


class AllGatherWithGradient(torch.autograd.Function):
    """AllGatherWithGradient"""

    @staticmethod
    def forward(ctx, input):
        world_size = dist.get_world_size()
        x_gather = [torch.ones_like(input) for _ in range(world_size)]
        torch.distributed.all_gather(x_gather, input, async_op=False)
        x_gather = torch.cat(x_gather, dim=0)
        return x_gather

    @staticmethod
    def backward(ctx, grad_output):

        reduction = torch.distributed.all_reduce(grad_output, async_op=True)
        reduction.wait()

        world_size = dist.get_world_size()
        N = grad_output.size(0)
        mini_batchsize = N // world_size
        cur_gpu = torch.distributed.get_rank()
        grad_output = grad_output[
            cur_gpu * mini_batchsize : (cur_gpu + 1) * mini_batchsize
        ]
        return grad_output
