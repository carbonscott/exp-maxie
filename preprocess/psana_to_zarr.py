"""
On summit,

# 4 nodes, 42 MPI tasks, 42 cores
OMP_NUM_THREADS=1 jsrun -n4 -a42 -c42 -g0 python psana_to_zarr.py mfxl1027522 epix10k2M --run 351 349 348
"""

import argparse
import numpy as np
import zarr
import os
import json
import logging
from mpi4py import MPI
from psana_wrapper import PsanaWrapperSmd, ImageRetrievalMode

def setup_logging(rank, log_level=logging.INFO):
    logging.basicConfig(
        level=log_level,
        format=f'%(asctime)s - Rank {rank} - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def create_zarr_store(output_dir, exp, run, rank, partition):
    filepath = os.path.join(output_dir, exp, f"r{run:d}", f"r{rank:d}.p{partition:d}.zarr")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    return zarr.open(filepath, mode='w')

def create_zarr_dataset(store, num_images, image_shape):
    return store.create_dataset(
        "data",
        shape=(num_images, *image_shape),
        chunks=(1, *image_shape),
        dtype=np.float32,
        compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.SHUFFLE)
    )

def create_checkpoint(output_dir, exp, completed_runs):
    checkpoint_file = os.path.join(output_dir, f"{exp}_checkpoint.json")
    with open(checkpoint_file, 'w') as f:
        json.dump({"completed_runs": completed_runs}, f)

def read_checkpoint(output_dir, exp):
    checkpoint_file = os.path.join(output_dir, f"{exp}_checkpoint.json")
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            return json.load(f)["completed_runs"]
    return []

def process_run(exp, run, detector, partition_size, output_dir, logger):
    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()

    logger.info(f"Starting processing for exp={exp}, run={run}, detector={detector}")

    try:
        smd_wrapper = PsanaWrapperSmd(exp=exp, run=str(run), detector_name=detector)
        logger.info(f"Successfully initialized PsanaWrapperSmd(exp={exp}, run={run}, detector={detector})")
    except Exception as e:
        logger.error(f"Error initializing PsanaWrapperSmd for exp={exp}, run={run}: {e}")
        return False

    images = []
    partition = 0
    store = None

    for idx, data in enumerate(smd_wrapper.iter_events(ImageRetrievalMode.image)):
        if data is None:
            logger.warning(f"Received None data for event {idx}")
            continue

        logger.debug(f"Processing event {idx}")

        images.append(data)

        if len(images) == partition_size:
            if store is None:
                store = create_zarr_store(output_dir, exp, run, mpi_rank, partition)

            dataset = create_zarr_dataset(store, len(images), images[0].shape)
            for i, img in enumerate(images):  # Optimize memory usage
                dataset[i] = img

            logger.info(f"Saved partition {partition} with {len(images)} images")

            # Reset for next partition
            images = []
            partition += 1
            store = None

    # Save any remaining images
    if images:
        if store is None:
            store = create_zarr_store(output_dir, exp, run, mpi_rank, partition)

        dataset = create_zarr_dataset(store, len(images), images[0].shape)
        for i, img in enumerate(images):  # Optimize memory usage
            dataset[i] = img

        logger.info(f"Saved final partition {partition} with {len(images)} images")

    return True

def main():
    parser = argparse.ArgumentParser(description="Process Psana data and save as Zarr files")
    parser.add_argument("exp", help="Experiment name")
    parser.add_argument("detector", help="Detector name")
    parser.add_argument("--run", type=int, nargs='+', required=True, help="Run numbers to process")
    parser.add_argument("--partition-size", type=int, default=512, help="Number of images per partition (default: 512)")
    parser.add_argument("--output-dir", default="./output", help="Directory to save Zarr files (default: ./output)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set the logging level")
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup MPI
    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()

    # Setup logging
    logger = setup_logging(mpi_rank, getattr(logging, args.log_level))

    # Only rank 0 should log this message and handle checkpointing
    if mpi_rank == 0:
        logger.info(f"Output directory: {os.path.abspath(args.output_dir)}")
        logger.info(f"Processing runs: {args.run}")

        completed_runs = read_checkpoint(args.output_dir, args.exp)
        runs_to_process = [run for run in args.run if run not in completed_runs]
        logger.info(f"Runs to process: {runs_to_process}")
    else:
        runs_to_process = None

    # Broadcast runs_to_process to all ranks
    runs_to_process = mpi_comm.bcast(runs_to_process, root=0)

    for run in runs_to_process:
        run_success = process_run(args.exp, run, args.detector, args.partition_size, args.output_dir, logger)

        # Synchronize all ranks after processing the run
        mpi_comm.Barrier()

        # Only rank 0 updates the checkpoint
        if mpi_rank == 0 and run_success:
            completed_runs.append(run)
            create_checkpoint(args.output_dir, args.exp, completed_runs)
            logger.info(f"Completed run {run} and updated checkpoint.")

    logger.info("All runs processed successfully.")

if __name__ == "__main__":
    main()
