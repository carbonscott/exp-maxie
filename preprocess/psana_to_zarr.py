"""
On summit,

# 4 nodes, 42 MPI tasks, 42 cores
OMP_NUM_THREADS=1 jsrun -n4 -a42 -c42 -g0 python psana_to_zarr.py mfxl1027522 epix10k2M --run 351 349 348
"""

import argparse
import numpy as np
import zarr
import os
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
    filename = f"{exp}.{run:06d}.rank{rank:06d}.part{partition:06d}.zarr"
    filepath = os.path.join(output_dir, filename)
    return zarr.open(filepath, mode='w')

def process_run(exp, run, detector, partition_size, output_dir):
    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()

    logger = setup_logging(mpi_rank)

    logger.info(f"Starting processing for exp={exp}, run={run}, detector={detector}")

    try:
        smd_wrapper = PsanaWrapperSmd(exp=exp, run=str(run), detector_name=detector)
        logger.info(f"Successfully initialized PsanaWrapperSmd(exp={exp}, run={run}, detector={detector})")
    except Exception as e:
        logger.error(f"Error initializing PsanaWrapperSmd for exp={exp}, run={run}: {e}")
        return

    images = []
    partition = 0
    store = create_zarr_store(output_dir, exp, run, mpi_rank, partition)

    for idx, data in enumerate(smd_wrapper.iter_events(ImageRetrievalMode.image)):
        if data is None:
            logger.warning(f"Received None data for event {idx}")
            continue

        logger.info(f"Processing event {idx}")

        images.append(data)

        if len(images) == partition_size:
            # Create a new zarr array and store the images
            zarr_array = store.create_dataset(
                "data",
                data=np.array(images),
                chunks=(min(64, partition_size), *data.shape),  # Adjust chunk size based on partition_size
                compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.SHUFFLE)
            )
            logger.info(f"Saved partition {partition} with {len(images)} images")

            # Reset for next partition
            images = []
            partition += 1
            store = create_zarr_store(output_dir, exp, run, mpi_rank, partition)

    # Save any remaining images
    if images:
        zarr_array = store.create_dataset(
            "data",
            data=np.array(images),
            chunks=(min(64, len(images)), *data.shape),
            compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.SHUFFLE)
        )
        logger.info(f"Saved final partition {partition} with {len(images)} images")

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

    # Setup logging
    mpi_rank = MPI.COMM_WORLD.Get_rank()
    logger = setup_logging(mpi_rank, getattr(logging, args.log_level))

    # Only rank 0 should log this message
    if mpi_rank == 0:
        logger.info(f"Output directory: {os.path.abspath(args.output_dir)}")
        logger.info(f"Processing runs: {args.run}")

    for run in args.run:
        process_run(args.exp, run, args.detector, args.partition_size, args.output_dir)

if __name__ == "__main__":
    main()
