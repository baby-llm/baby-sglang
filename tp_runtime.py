from __future__ import annotations

import os
from typing import Tuple

import torch
import torch.distributed as dist


def init_tp() -> Tuple[int, int, object]:
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    tp_group = dist.group.WORLD
    if dist.is_initialized():
        rank = dist.get_rank(group=tp_group)
        world_size = dist.get_world_size(group=tp_group)
    return rank, world_size, tp_group


def tp_all_reduce_sum(x: torch.Tensor, tp_group) -> torch.Tensor:
    if dist.get_world_size(group=tp_group) == 1:
        return x

    dist.all_reduce(x, op=dist.ReduceOp.SUM, group=tp_group)
    return x


def tp_all_gather(x: torch.Tensor, tp_group) -> torch.Tensor:
    world_size = dist.get_world_size(group=tp_group)
    if world_size == 1:
        return x

    gather_list = [torch.empty_like(x) for _ in range(world_size)]
    dist.all_gather(gather_list, x, group=tp_group)
    return torch.cat(gather_list, dim=0)


def tp_broadcast(x: torch.Tensor, src: int, tp_group) -> torch.Tensor:
    if dist.get_world_size(group=tp_group) == 1:
        return x

    dist.broadcast(x, src=src, group=tp_group)
    return x
