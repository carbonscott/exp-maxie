#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for streaming dataset, adapted from train.py.
This script focuses only on testing the dataset functionality.

Usage:
   python test_streaming_dataset.py config.yaml
"""

# =============
# BASIC IMPORTS
# =============
import os
import yaml
import tqdm
import argparse
import logging
import traceback
import time
import json

from contextlib import nullcontext
from omegaconf import OmegaConf

# =====
# TORCH
# =====
import torch
import torch.distributed as dist

# =====
# Dataset imports
# =====
# Import the streaming dataset
from maxie.datasets.streaming_dataset import StreamingDataset, StreamingDataConfig
from maxie.utils.dist import dist_setup
from maxie.utils.seed import set_seed
from maxie.utils.logger import init_logger

## # Get the logger
## logger = logging.getLogger(__name__)

# ======================
# COMMAND LINE INTERFACE
# ======================
parser = argparse.ArgumentParser(description="Test streaming dataset using a YAML config file.")
parser.add_argument("yaml_file", help="Path to the YAML file")
args = parser.parse_args()

# =============
# CONFIGURATION
# =============
# Load CONFIG from YAML
fl_yaml = args.yaml_file
with open(fl_yaml, 'r') as fh:
    config = OmegaConf.create(yaml.safe_load(fh))

# ====
# DIST
# ====
dist_config = dist_setup(
    config.dist.cpu_only,
    config.dist.device_per_node,
    config.dist.backend,
)
uses_dist = dist_config.uses_dist
dist_rank = dist_config.rank
dist_local_rank = dist_config.local_rank
dist_world_size = dist_config.world_size
device = dist_config.device

# ======
# LOGGER
# ======
timestamp, logger = init_logger(
    uses_dist,
    dist_rank,
    device,
    "test_streaming",
    config.logging.directory,
    config.logging.level,
    'console',
)

# =======
# SEEDING
# =======
base_seed = 0
seed_offset = dist_rank if config.dist.uses_unique_world_seed else 0
world_seed = base_seed + seed_offset
set_seed(world_seed)

# =======
# DATASET
# =======
# Determine node ID and local rank for dataset configuration
device_per_node = config.dist.device_per_node if hasattr(config.dist, 'device_per_node') else torch.cuda.device_count()
node_id = dist_rank // device_per_node if torch.cuda.is_available() else 0
num_nodes = (dist_world_size + device_per_node - 1) // device_per_node if torch.cuda.is_available() else 1
logger.info(f"Rank {dist_rank}, Local Rank {dist_local_rank}, Node ID {node_id}")

# Create streaming dataset config
dataset_config = StreamingDataConfig(
    C=config.dataset.input.C,
    H=config.dataset.input.H,
    W=config.dataset.input.W,
    addresses=config.dataset.streaming_addresses if hasattr(config.dataset, 'streaming_addresses') else [config.dataset.streaming_address],
    address_assignment=config.dataset.address_assignment if hasattr(config.dataset, 'address_assignment') else "round-robin",
    sockets_per_node=config.dataset.sockets_per_node if hasattr(config.dataset, 'sockets_per_node') else 1,
    node_address_map=config.dataset.node_address_map if hasattr(config.dataset, 'node_address_map') else None,
    queue_size=config.dataset.queue_size if hasattr(config.dataset, 'queue_size') else 128,
    timeout_ms=config.dataset.timeout_ms if hasattr(config.dataset, 'timeout_ms') else 1000,
    max_wait_time=config.dataset.max_wait_time if hasattr(config.dataset, 'max_wait_time') else 60,
    connect_timeout=config.dataset.connect_timeout if hasattr(config.dataset, 'connect_timeout') else 10,
    dist_rank=dist_rank,
    dist_world_size=dist_world_size,
    local_rank=dist_local_rank,
    node_id=node_id,
    num_nodes=num_nodes,
    lock_dir=os.getcwd(),
)

# Create the dataset
dataset = StreamingDataset(dataset_config)
logger.info(f"[RANK {dist_rank}] Creating streaming dataset with addresses: {dataset.node_addresses}")

# Create the dataloader (adapting from train.py's wrap_with_torch_dataloader)
batch_size = config.dataset.batch_size
num_workers = config.dataset.num_workers
pin_memory = config.dataset.pin_memory if hasattr(config.dataset, 'pin_memory') else True
prefetch_factor = config.dataset.prefetch_factor if hasattr(config.dataset, 'prefetch_factor') else 2

logger.info(f"[RANK {dist_rank}] Creating DataLoader with batch_size={batch_size}, workers={num_workers}")
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=pin_memory,
    prefetch_factor=prefetch_factor,
)

# ============
# TESTING LOOP
# ============
try:
    # Process batches
    logger.info(f"[RANK {dist_rank}] Starting test loop for {config.dataset.num_batches} batches")
    start_time = time.time()
    total_samples = 0

    with tqdm.tqdm(
        total=config.dataset.num_batches,
        desc=f'[RANK {dist_rank}] Testing',
    ) as pbar:
        # Process batches
        for batch_idx, batch_data in enumerate(dataloader):
            # Move batch to device
            batch_data = batch_data.to(device, non_blocking=True)

            # Update stats
            batch_size = batch_data.size(0)
            total_samples += batch_size

            # Log progress
            if batch_idx % 10 == 0:
                elapsed = time.time() - start_time
                samples_per_sec = total_samples / elapsed if elapsed > 0 else 0
                logger.info(f"[RANK {dist_rank}] Batch {batch_idx}: "
                           f"shape={batch_data.shape}, "
                           f"samples/sec={samples_per_sec:.1f}")

            # Update progress bar
            pbar.update(1)
            pbar.set_postfix(shape=str(batch_data.shape))

            # Break after processing enough batches
            if batch_idx + 1 >= config.dataset.num_batches:
                break

            # Synchronize ranks periodically
            if uses_dist and batch_idx % 10 == 0:
                dist.barrier()

    # Final stats
    elapsed = time.time() - start_time
    samples_per_sec = total_samples / elapsed if elapsed > 0 else 0

    logger.info(f"[RANK {dist_rank}] Test completed: processed {total_samples} samples "
               f"in {elapsed:.2f}s ({samples_per_sec:.1f} samples/sec)")

    # Save checkpoint info if requested
    if config.dataset.save_checkpoint:
        # Get checkpoint info from all ranks
        checkpoint_info = []

        if uses_dist:
            # Each rank gets its own info
            local_info = dataset.get_checkpoint_info() if dist_rank == dist_rank else None

            # Gather info from all ranks
            gathered_info = [None] * dist_world_size
            dist.all_gather_object(gathered_info, local_info)
            checkpoint_info = [info for info in gathered_info if info is not None]
        else:
            # Single process mode
            checkpoint_info = [dataset.get_checkpoint_info()]

        # Save checkpoint info to file
        if dist_rank == 0:
            os.makedirs("checkpoints", exist_ok=True)
            checkpoint_file = f"checkpoints/dataset_checkpoint_{timestamp}.json"
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_info, f, indent=2)
            logger.info(f"Saved checkpoint info to {checkpoint_file}")

except KeyboardInterrupt:
    logger.info(f"[RANK {dist_rank}] Test interrupted by user")
except Exception as e:
    tb = traceback.format_exc()
    logger.error(f"[RANK {dist_rank}] Error: {e}\nTraceback: {tb}")
finally:
    # Clean up resources
    if 'dataset' in locals():
        logger.info(f"[RANK {dist_rank}] Closing dataset")
        dataset.close()

    # Cleanup distributed environment
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

logger.info(f"[RANK {dist_rank}] Test script completed")
