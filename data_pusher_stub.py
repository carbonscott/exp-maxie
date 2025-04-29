import torch
import time
import argparse
import io
import numpy as np
from pynng import Push0
import threading
import multiprocessing as mp

def tensor_to_bytes(tensor):
    """Convert a PyTorch tensor to bytes for transmission."""
    buffer = io.BytesIO()
    tensor_np = tensor.cpu().numpy()
    np.save(buffer, tensor_np)
    return buffer.getvalue()

def dummy_data_pusher_thread(
    address,
    C=1,
    H=1920,
    W=1920,
    total_size=10000,
    push_interval=0.001,
    continuous=False,
    stream_id=0,
    shared_counter=None,
):
    """
    Push random tensors to the specified address.

    Args:
        address (str): Address to push data to (URL format).
        C (int): Number of channels.
        H (int): Height of the tensor.
        W (int): Width of the tensor.
        total_size (int): Total number of samples to push.
        push_interval (float): Interval between pushes in seconds.
        continuous (bool): Whether to continuously push data.
        stream_id (int): Identifier for this stream.
    """
    with Push0(listen=address) as sock:
        print(f"Stream {stream_id}: Listening at {address}, pushing data with shape ({C}, {H}, {W})")

        # Counter for continuous mode
        counter = 0

        while True:
            for i in range(total_size):
                # In continuous mode, use a counter to ensure unique indices
                if continuous:
                    if shared_counter is not None:
                        # Atomically get and increment the shared counter
                        with shared_counter.get_lock():
                            global_idx = shared_counter.value
                            shared_counter.value += 1
                    else:
                        # Fallback to local counter for backward compatibility
                        global_idx = counter
                        counter += 1
                else:
                    global_idx = i

                # Generate a random tensor
                tensor = torch.randn(C, H, W)

                # Convert tensor to bytes
                data = tensor_to_bytes(tensor)

                # Prepare metadata
                metadata = {
                    'index': global_idx,
                    'shape': (C, H, W),
                    'total_size': total_size,
                    'stream_id': stream_id
                }
                metadata_bytes = str(metadata).encode('utf-8')

                # Push metadata header and data payload
                sock.send(metadata_bytes + b'\n' + data)

                ## # Wait for the specified interval
                ## time.sleep(push_interval)

                if (i + 1) % 1000 == 0:
                    print(f"Stream {stream_id}: Pushed {i + 1} samples in current cycle")

            # If not continuous, break after one cycle
            if not continuous:
                print(f"Stream {stream_id}: Done pushing {total_size} samples")
                break

            print(f"Stream {stream_id}: Completed one cycle of {total_size} samples, continuing...")

def dummy_data_pusher(
    address,
    C=1,
    H=1920,
    W=1920,
    total_size=10000,
    push_interval=0.001,
    continuous=False,
    num_streams=1
):
    """
    Push random tensors to the specified address(es).

    Args:
        address (str): Base address to push data to (URL format).
        C (int): Number of channels.
        H (int): Height of the tensor.
        W (int): Width of the tensor.
        total_size (int): Total number of samples to push.
        push_interval (float): Interval between pushes in seconds.
        continuous (bool): Whether to continuously push data.
        num_streams (int): Number of push streams to create.
    """
    threads = []

    # Create a shared counter for continuous mode
    shared_counter = mp.Value('i', 0) if continuous else None

    for i in range(num_streams):
        # Create address for this stream
        if num_streams > 1:
            # Parse the base address to create a new one with offset port
            if address.startswith('tcp://') and ':' in address.split('//')[1]:
                # Format: tcp://host:port
                host_port = address.split('//')[1]
                host, port = host_port.rsplit(':', 1)
                try:
                    new_port = int(port) + i
                    stream_address = f"tcp://{host}:{new_port}"
                except ValueError:
                    # Fallback if port can't be parsed
                    stream_address = f"{address}_{i}"
            else:
                # Non-TCP or format without port
                stream_address = f"{address}_{i}"
        else:
            stream_address = address

        # Create and start thread for this stream
        thread = threading.Thread(
            target=dummy_data_pusher_thread,
            args=(
                stream_address,
                C,
                H,
                W,
                total_size,
                push_interval,
                continuous,
                i,
                shared_counter,
            )
        )
        thread.daemon = True
        thread.start()
        threads.append(thread)

    # In continuous mode, we don't want to block the main thread waiting
    if continuous:
        try:
            while True:
                time.sleep(1)  # Keep the main thread alive
        except KeyboardInterrupt:
            print("Interrupted by user, shutting down...")
    else:
        # In non-continuous mode, wait for all threads to complete
        for thread in threads:
            thread.join()

def main():
    parser = argparse.ArgumentParser(description='Push dummy data for PyTorch training.')
    parser.add_argument('--address', default='tcp://127.0.0.1:5555', help='Address to push data to (URL format).')
    parser.add_argument('--C', type=int, default=3, help='Number of channels.')
    parser.add_argument('--H', type=int, default=224, help='Height of the tensor.')
    parser.add_argument('--W', type=int, default=224, help='Width of the tensor.')
    parser.add_argument('--total-size', type=int, default=10000, help='Total number of samples to push.')
    parser.add_argument('--push-interval', type=float, default=0.001, help='Interval between pushes in seconds.')
    parser.add_argument('--continuous', action='store_true', help='Continuously push data (loop through indices).')
    parser.add_argument('--num-streams', type=int, default=1, help='Number of push streams to create.')

    args = parser.parse_args()

    dummy_data_pusher(
        args.address,
        args.C,
        args.H,
        args.W,
        args.total_size,
        args.push_interval,
        args.continuous,
        args.num_streams
    )

if __name__ == '__main__':
    main()
