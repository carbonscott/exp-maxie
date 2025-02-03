import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, DataLoader
import zarr

import os
import socket

from mpi4py import MPI

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

import torch
from torch.utils.data import IterableDataset, DataLoader
import zarr

# ================================================================
# Dummy Zarr Files Setup
# ================================================================

# Global dictionary to simulate Zarr files in memory.
DUMMY_ZARR_FILES = {}

# Create 100 dummy file names: data/file0.zarr, ..., data/file99.zarr.
# Each file will have either 9 or 10 images:
#   - Even-indexed files get 9 images.
#   - Odd-indexed files get 10 images.
dummy_file_list = []
num_files = 32
for i in range(num_files):
    file_name = f"data/file{i}.zarr"
    dummy_file_list.append(file_name)
    num_images = 9 if i % 2 == 0 else 10
    # Each image is represented as a tuple (file_name, image_index)
    images = [(file_name, j) for j in range(num_images)]
    DUMMY_ZARR_FILES[file_name] = {"images": images}

# Monkey-patch zarr.open to use our dummy files.
def dummy_zarr_open(file_path, mode='r'):
    return DUMMY_ZARR_FILES[file_path]

zarr.open = dummy_zarr_open

# ================================================================
# GlobalCheckpointLazyZarrIterableDataset Definition
# ================================================================

class GlobalCheckpointLazyZarrIterableDataset(IterableDataset):
    def __init__(self, file_list, rank=0, world_size=1, start_state=None):
        """
        Args:
            file_list (List[str]): Global (deterministically sorted) list of file paths.
            rank (int): Rank of the current process.
            world_size (int): Total number of distributed processes.
            start_state (dict, optional): Global checkpoint state.
                Expected format: {"file": <absolute file path>, "image_index": <int>}.
                If None, iteration starts at the beginning.
        """
        super(GlobalCheckpointLazyZarrIterableDataset, self).__init__()
        self.file_list = file_list
        self.rank = rank
        self.world_size = world_size
        # Global checkpoint state: independent of how files are partitioned.
        self.start_state = start_state if start_state is not None else {"file": None, "image_index": 0}
        self._state = None  # Will hold the most recent state during iteration

    def __iter__(self):
        # ------------------------------
        # Worker support: if using DataLoader(num_workers > 0)
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id = 0
            num_workers = 1

        # ------------------------------
        # Process the global checkpoint state.
        if self.start_state["file"] is not None:
            checkpoint_file = self.start_state["file"]
            checkpoint_image_index = self.start_state["image_index"]
            try:
                # Find the checkpoint file's global index in the full file list.
                checkpoint_global_index = self.file_list.index(checkpoint_file)
            except ValueError:
                checkpoint_global_index = 0
                checkpoint_image_index = 0
        else:
            checkpoint_global_index = 0
            checkpoint_image_index = 0

        # ------------------------------
        # Partition the global file list among ranks.
        # Attach each file its global index.
        global_partition = [(i, f) for i, f in enumerate(self.file_list) if i % self.world_size == self.rank]
        # Further partition among DataLoader workers (if any)
        global_partition = global_partition[worker_id::num_workers]

        # ------------------------------
        # Determine where to start in this partition.
        # Find the first file whose global index is >= checkpoint's file index.
        start_partition_idx = 0
        for idx, (global_idx, file_path) in enumerate(global_partition):
            if global_idx >= checkpoint_global_index:
                start_partition_idx = idx
                break
        else:
            # If no file in this partition is beyond the checkpoint, nothing to iterate.
            return iter([])

        self._state = {"file": None, "image_index": 0}

        # ------------------------------
        # Iterate over files starting at start_partition_idx.
        for i in range(start_partition_idx, len(global_partition)):
            global_idx, file_path = global_partition[i]
            # Lazy-load the file.
            z = zarr.open(file_path, mode='r')
            images = z["images"]

            # Determine starting image index.
            # If this file is the checkpoint file, resume at the checkpoint image index.
            if global_idx == checkpoint_global_index:
                start_img_idx = checkpoint_image_index
            else:
                start_img_idx = 0

            print(f"[Rank {self.rank} Worker {worker_id}] Processing File {file_path} "
                  f"(Global index {global_idx}) starting at image index {start_img_idx} "
                  f"with {len(images)} images.")

            for img_idx in range(start_img_idx, len(images)):
                self._state = {"file": file_path, "image_index": img_idx}
                yield images[img_idx]

    def state_dict(self):
        """
        Returns the current global checkpoint state.
        """
        return self._state if self._state is not None else self.start_state

    def load_state_dict(self, state_dict):
        """
        Loads the given global checkpoint state.
        """
        self.start_state = state_dict.copy()

# ================================================================
# Simulation Functions for Testing
# ================================================================

def simulate_dataset(world_size, checkpoint_state=None, num_workers=0):
    """
    Simulates the dataset iteration for each rank.

    Args:
        world_size (int): Number of ranks to simulate.
        checkpoint_state (dict or None): Global checkpoint state.
        num_workers (int): Number of DataLoader workers to simulate (0 means run __iter__ directly).

    Returns:
        dict: Mapping from rank to list of yielded images.
    """
    rank = dist.get_rank()
    print("\n" + "="*60)
    if checkpoint_state:
        print(f"[RANK {rank}]Simulating WITH checkpoint state:", checkpoint_state)
    else:
        print(f"[RANK {rank}]Simulating WITHOUT checkpoint state.")
    print("="*60)

    results = {}
    print(f"\n--- Simulating Rank {rank} ---")
    dataset = GlobalCheckpointLazyZarrIterableDataset(
        file_list=dummy_file_list,
        rank=rank,
        world_size=world_size,
        start_state=checkpoint_state
    )
    if num_workers > 0:
        loader = DataLoader(dataset, batch_size=1, num_workers=num_workers)
        images_yielded = []
        # Note: DataLoader with IterableDataset yields batches.
        for batch in loader:
            # Each batch is a list with one image (our dummy tuple)
            images_yielded.append(batch[0])
    else:
        images_yielded = list(dataset)
    results[rank] = images_yielded
    return results

# ================================================================
# Main Simulation
# ================================================================
if __name__ == "__main__":
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

    ## # ----- Simulation 1: Without a Checkpoint, single-worker -----
    ## results_no_checkpoint = simulate_dataset(world_size, checkpoint_state=None, num_workers=0)

    dist.barrier(device_ids=[local_rank])

    # ----- Simulation 2: With a Checkpoint, single-worker -----
    # For example, suppose in a previous run we processed up to file "data/file42.zarr"
    # at image index 5.
    checkpoint_state = {"file": "data/file9.zarr", "image_index": 5}
    results_with_checkpoint = simulate_dataset(world_size, checkpoint_state=checkpoint_state, num_workers=0)

    ## # ----- Simulation 3: Without a Checkpoint, multi-worker (2 workers per rank) -----
    ## results_no_checkpoint_multi = simulate_dataset(world_size, checkpoint_state=None, num_workers=2)

    ## # ----- Simulation 4: With a Checkpoint, multi-worker (2 workers per rank) -----
    ## results_with_checkpoint_multi = simulate_dataset(world_size, checkpoint_state=checkpoint_state, num_workers=2)

    dist.destroy_process_group()
