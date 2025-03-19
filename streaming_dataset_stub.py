import torch
from torch.utils.data import IterableDataset
import torch.multiprocessing as mp
from dataclasses import dataclass
from typing import Optional, Union, List, Tuple
import io
import numpy as np
from pynng import Pull0, Timeout
import threading
import queue
import time
import ast
import logging
import signal
import os
import pathlib
import uuid
import socket

logger = logging.getLogger(__name__)

@dataclass
class StreamingDataConfig:
    C: int                     # Number of channels
    H: int                     # Height of tensor
    W: int                     # Width of tensor
    address: str               # Address to pull data from
    queue_size: int = 128      # Size of shared queue
    timeout_ms: int = 1000     # Socket timeout in milliseconds
    max_wait_time: int = 60    # Max time to wait if queue is empty (seconds)
    connect_timeout: int = 10  # Seconds to wait for initial connection
    transforms: Union[None, List, Tuple] = None  # Data transforms
    dtype: torch.dtype = None  # Optional dtype conversion
    dist_rank: int = 0         # Distributed rank (for logging)
    dist_world_size: int = 1   # World size (for stats only)
    local_rank: int = 0        # Local rank (within node)
    num_nodes: int = 1         # Number of nodes
    node_id: int = 0           # ID of this node
    lock_dir: str = None       # Directory for lock files (default: current directory)


class StreamingDataset(IterableDataset):
    """
    An optimized streaming dataset for multi-node training that:
    1. Has one socket connection per node (not per rank)
    2. Waits instead of generating random tensors if data isn't available
    3. Tracks global indices for simple checkpointing/resumption
    4. Works with distributed training across multiple nodes
    """

    def __init__(self, config):
        self.config = config
        self.dtype = config.dtype
        self.transforms = config.transforms
        self.rank = config.dist_rank
        self.world_size = config.dist_world_size
        self.local_rank = config.local_rank
        self.node_id = config.node_id

        # Initialize reporting variables BEFORE starting any threads
        self.last_report_time = time.time()
        self.report_interval = 10.0  # seconds

        # Track highest global index seen by this rank
        self.highest_index = -1

        # Initialize thread and stop flag
        self.puller_thread = None
        self.stop_flag = mp.Event()

        # Use a unique queue per rank instead of trying to share
        # This is simpler and more reliable in distributed environments
        self.queue = mp.Queue(config.queue_size)

        # Create shared statistics counters
        self.stats = {
            'total_received': mp.Value('i', 0),
            'total_yielded': mp.Value('i', 0),
            'latest_index': mp.Value('i', -1),
        }

        # Start puller thread for all ranks
        # Each rank will have its own connection
        self._start_puller_thread()

        # Register signal handlers for cleanup
        self._original_sigint_handler = signal.getsignal(signal.SIGINT)
        self._original_sigterm_handler = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _start_puller_thread(self):
        """Start the puller thread if it's not already running"""
        # Log that we're starting a thread
        logger.info(f"[RANK {self.rank}] Starting puller thread for {self.config.address}")

        # Create and start the thread
        self.puller_thread = threading.Thread(
            target=self._pull_data_thread,
            args=(
                self.config.address,
                self.config.timeout_ms,
                self.node_id,
                self.rank,
            )
        )
        self.puller_thread.daemon = True
        self.puller_thread.start()

        # Wait for some initial data
        got_data = False
        start_time = time.time()
        logger.info(f"[RANK {self.rank}] Waiting for initial data...")

        while time.time() - start_time < self.config.connect_timeout:
            try:
                if not self.queue.empty():
                    got_data = True
                    break
            except Exception as e:
                logger.warning(f"[RANK {self.rank}] Error checking queue: {e}")
            time.sleep(0.1)

        if got_data:
            logger.info(f"[RANK {self.rank}] Received initial data")
        else:
            logger.warning(f"[RANK {self.rank}] No initial data received after {self.config.connect_timeout}s")

    def _signal_handler(self, sig, frame):
        """Handle signals to clean up resources properly"""
        logger.info(f"[RANK {self.rank}] Received signal {sig}, cleaning up...")
        self.close()

        # Call the original handler
        if sig == signal.SIGINT and self._original_sigint_handler:
            self._original_sigint_handler(sig, frame)
        elif sig == signal.SIGTERM and self._original_sigterm_handler:
            self._original_sigterm_handler(sig, frame)

    def _pull_data_thread(self, address, timeout_ms, node_id, rank):
        """Background thread to continuously pull data and fill the queue"""
        # Make sure we have our own report time for the thread
        thread_report_time = time.time()

        try:
            with Pull0(dial=address) as sock:
                sock.recv_timeout = timeout_ms

                logger.info(f"[RANK {rank}] Puller initialized, dialing to {address}")

                while not self.stop_flag.is_set():
                    try:
                        # Pull data
                        data = sock.recv()

                        # Track received data
                        with self.stats["total_received"].get_lock():
                            self.stats["total_received"].value += 1

                        # Extract tensor and metadata
                        tensor, metadata = self._parse_data(data)

                        if tensor is not None:
                            # Convert dtype if specified
                            if self.dtype is not None:
                                tensor = tensor.to(self.dtype)

                            # Update latest global index if available
                            if metadata and 'index' in metadata:
                                with self.stats["latest_index"].get_lock():
                                    self.stats["latest_index"].value = max(
                                        self.stats["latest_index"].value, 
                                        metadata['index']
                                    )

                            # Put in the queue (block if full)
                            try:
                                self.queue.put((tensor, metadata), block=True, timeout=1.0)
                            except queue.Full:
                                logger.warning(f"[RANK {rank}] Queue full, dropping tensor")
                                continue

                        # Periodic reporting (thread-local time)
                        current_time = time.time()
                        if current_time - thread_report_time > self.report_interval:
                            try:
                                queue_size = self.queue.qsize()
                            except:
                                queue_size = "Unknown"  # qsize() not reliable on all platforms

                            received = self.stats["total_received"].value
                            yielded = self.stats["total_yielded"].value
                            latest_idx = self.stats["latest_index"].value

                            logger.info(f"[RANK {rank}] Stats: queue={queue_size}, "
                                      f"received={received}, yielded={yielded}, "
                                      f"latest_index={latest_idx}")
                            thread_report_time = current_time

                    except Timeout:
                        # Just continue on timeout - this is normal
                        continue
                    except Exception as e:
                        logger.error(f"[RANK {rank}] Error pulling data: {e}")
                        continue

        except Exception as e:
            logger.error(f"[RANK {rank}] Fatal error in puller thread: {e}")

        logger.info(f"[RANK {rank}] Puller thread exiting")

    def _parse_data(self, data):
        """Parse data received from the push socket"""
        # Extract metadata
        newline_index = data.find(b'\n')
        if newline_index != -1:
            metadata_bytes = data[:newline_index]
            data = data[newline_index + 1:]

            try:
                metadata = ast.literal_eval(metadata_bytes.decode('utf-8'))
            except Exception as e:
                logger.error(f"[RANK {self.rank}] Error parsing metadata: {e}")
                metadata = None
        else:
            metadata = None

        # Extract tensor
        try:
            buffer = io.BytesIO(data)
            tensor_np = np.load(buffer)
            tensor = torch.from_numpy(tensor_np)
            return tensor, metadata
        except Exception as e:
            logger.error(f"[RANK {self.rank}] Error deserializing tensor: {e}")
            return None, None

    def __iter__(self):
        """Return an iterator over the streaming data"""
        # Get worker info for this worker
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers

        logger.info(f"[RANK {self.rank}] Worker {worker_id}/{num_workers} starting iteration")

        # Iterator function
        def data_iterator():
            while not self.stop_flag.is_set():
                try:
                    # Try to get an item from the queue
                    try:
                        # Block with timeout
                        tensor, metadata = self.queue.get(block=True, timeout=0.1)
                    except queue.Empty:  # Use standard queue.Empty exception
                        # If queue is empty, check how long we've been waiting
                        if not hasattr(data_iterator, 'wait_start_time'):
                            data_iterator.wait_start_time = time.time()

                        wait_time = time.time() - data_iterator.wait_start_time

                        # If we've waited too long, raise an exception
                        if wait_time > self.config.max_wait_time:
                            logger.error(f"[RANK {self.rank}] No data received for {wait_time:.1f} seconds. Data source may be inactive.")
                            raise RuntimeError(f"Data source appears to be inactive after {wait_time:.1f} seconds")

                        # Otherwise just continue waiting
                        if wait_time > 5 and int(wait_time) % 5 == 0:  # Log every 5 seconds
                            logger.warning(f"[RANK {self.rank}] Waiting for data: {wait_time:.1f}s elapsed")

                        continue

                    # Reset wait timer since we got data
                    if hasattr(data_iterator, 'wait_start_time'):
                        delattr(data_iterator, 'wait_start_time')

                    # Apply transformations if specified
                    if self.transforms is not None:
                        for transform in self.transforms:
                            tensor = transform(tensor)

                    # Update stats
                    with self.stats["total_yielded"].get_lock():
                        self.stats["total_yielded"].value += 1

                    # Update the highest index seen by this rank
                    if metadata and 'index' in metadata:
                        self.highest_index = max(self.highest_index, metadata['index'])

                    yield tensor

                except KeyboardInterrupt:
                    logger.info(f"[RANK {self.rank}] Worker {worker_id} received KeyboardInterrupt")
                    break
                except Exception as e:
                    logger.error(f"[RANK {self.rank}] Worker {worker_id} error: {e}")
                    # Don't continue on exceptions other than queue timeouts
                    raise

        return data_iterator()

    def get_checkpoint_info(self):
        """
        Return checkpoint info that can be saved with the model checkpoint.
        This provides hints to the data pusher about where to resume.
        """
        # Return the highest global index processed by this rank
        return {
            'rank': self.rank,
            'highest_index': self.highest_index,
            'total_samples_received': self.stats["total_received"].value,
            'total_samples_processed': self.stats["total_yielded"].value,
        }

    def close(self):
        """Clean up resources"""
        logger.info(f"[RANK {self.rank}] Closing streaming dataset. "
                   f"Received {self.stats['total_received'].value} tensors, "
                   f"yielded {self.stats['total_yielded'].value} tensors, "
                   f"highest index {self.highest_index}")

        # Stop the puller thread
        if hasattr(self, 'puller_thread') and self.puller_thread:
            self.stop_flag.set()
            self.puller_thread.join(timeout=2.0)

        # Restore original signal handlers
        signal.signal(signal.SIGINT, self._original_sigint_handler)
        signal.signal(signal.SIGTERM, self._original_sigterm_handler)
