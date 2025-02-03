#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim

import os
import socket
from mpi4py import MPI

from functools import partial

from accelerate import Accelerator, FullyShardedDataParallelPlugin
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, BackwardPrefetch, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    CheckpointImpl,
    checkpoint_wrapper,
)

## os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def init_dist_env_on_s3df():
    # Use mpi4py to get rank and size information
    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()

    # Calculate local rank based on the available GPUs
    mpi_local_rank = mpi_rank % torch.cuda.device_count()

    # Are we using multiple ranks?
    uses_dist = mpi_size > 1

    if uses_dist:
        MAIN_RANK = 0

        # MPI environment variables detected (e.g., Summit)
        os.environ["WORLD_SIZE"] = str(mpi_size)
        os.environ["RANK"]       = str(mpi_rank)
        os.environ["LOCAL_RANK"] = str(mpi_local_rank)

        # Set the default master address and port, prioritizing definition in the job script
        os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "29500")

        master_addr = os.getenv("MASTER_ADDR", None)
        if master_addr is None:
            # Try to determine the master address and broadcast it to every rank
            master_addr = socket.gethostbyname(socket.gethostname()) if mpi_rank == MAIN_RANK else None
            master_addr = mpi_comm.bcast(master_addr, root = MAIN_RANK)
            os.environ["MASTER_ADDR"] = master_addr
        else:
            os.environ["MASTER_ADDR"] = "127.0.0.1"

        print(f"Environment setup for distributed computation: WORLD_SIZE={os.environ['WORLD_SIZE']}, RANK={os.environ['RANK']}, LOCAL_RANK={os.environ['LOCAL_RANK']}, MASTER_ADDR={os.environ['MASTER_ADDR']}, MASTER_PORT={os.environ['MASTER_PORT']}")

fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False),  # Control model state dict saving
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=False, rank0_only=False),  # Control optimizer state dict
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # Full parameter sharding
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # Prefetching for better performance
    auto_wrap_policy=transformer_auto_wrap_policy  # Auto-wrap transformer layers if present
)

# --------------------------
# NEW: Define a simple model (unchanged)
# --------------------------
class SimpleModel(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, output_dim=2):
        super(SimpleModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)

# --------------------------
# Main training function with Accelerate and FSDP Plugin
# --------------------------
def main():
    dist_backend = 'nccl'
    init_dist_env_on_s3df()
    uses_dist = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if uses_dist:
        rank       = int(os.environ["RANK"      ])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend     = dist_backend,
                                rank        = rank,
                                world_size  = world_size,
                                init_method = "env://",)
        print(f"RANK:{rank},LOCAL_RANK:{local_rank},WORLD_SIZE:{world_size}")
    else:
        rank       = 0
        local_rank = 0
        world_size = 1
        print(f"NO distributed environment is required.  RANK:{rank},LOCAL_RANK:{local_rank},WORLD_SIZE:{world_size}")

    # NEW: Initialize Accelerator with FSDP plugin and mixed precision
    accelerator = Accelerator(fsdp_plugin=fsdp_plugin, mixed_precision="bf16")

    # Standard model and optimizer creation (unchanged)
    model = SimpleModel()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # --------------------------
    # NEW: Apply activation checkpointing to Linear layers
    # --------------------------
    # --- Activation checkpointing
    non_reentrant_wrapper = partial(
        checkpoint_wrapper,
        ## offload_to_cpu  = False,
        checkpoint_impl = CheckpointImpl.NO_REENTRANT,
    )
    ac_layer = SimpleModel
    if ac_layer is not None:
        check_fn = lambda submodule: isinstance(submodule, ac_layer)
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn = non_reentrant_wrapper,
            check_fn              = check_fn
        )

    model, optimizer = accelerator.prepare(model, optimizer)

    if accelerator.is_main_process:
        # Check if the (unwrapped) model is an instance of FSDP.
        unwrapped_model = accelerator.unwrap_model(model)
        if isinstance(unwrapped_model, FSDP):
            print("Model is successfully sharded using FSDP!")
        else:
            print("Warning: Model is not wrapped in FSDP.")
        print(f"Using device: {accelerator.device}")

    # Dummy input data and target (unchanged)
    x = torch.randn(32, 128)
    target = torch.randint(0, 2, (32,))

    # NEW: Move dummy data to the appropriate device
    x = x.to(accelerator.device)
    target = target.to(accelerator.device)

    # Training loop (unchanged except for accelerator.backward)
    model.train()
    for epoch in range(10):
        optimizer.zero_grad()
        output = model(x)
        loss = nn.functional.cross_entropy(output, target)
        accelerator.backward(loss)
        optimizer.step()
        if accelerator.is_main_process:
            print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    if accelerator.is_main_process:
        ckpt_dir = "accelerate_ckpt"
        # Save the entire training state (model, optimizer, etc.) to ckpt_dir.
        accelerator.save_state(ckpt_dir)
        print(f"Accelerate checkpoint saved in directory: {ckpt_dir}")
    accelerator.wait_for_everyone()

    if uses_dist:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()
