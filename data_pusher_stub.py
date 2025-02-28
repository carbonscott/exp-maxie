import torch
import time
import argparse
import io
import numpy as np
from pynng import Push0

def tensor_to_bytes(tensor):
    """Convert a PyTorch tensor to bytes for transmission."""
    buffer = io.BytesIO()
    tensor_np = tensor.cpu().numpy()
    np.save(buffer, tensor_np)
    return buffer.getvalue()

def dummy_data_pusher(
    address,
    C=1,
    H=1920,
    W=1920,
    total_size=10000,
    push_interval=0.001,
    continuous=False
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
    """
    with Push0(listen=address) as sock:
        print(f"Listening at {address}, pushing data with shape ({C}, {H}, {W})")

        # Counter for continuous mode
        counter = 0

        while True:
            for i in range(total_size):
                # In continuous mode, use a counter to ensure unique indices
                if continuous:
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
                    'total_size': total_size
                }
                metadata_bytes = str(metadata).encode('utf-8')

                # Push metadata header and data payload
                sock.send(metadata_bytes + b'\n' + data)

                ## # Wait for the specified interval
                ## time.sleep(push_interval)

                if (i + 1) % 1000 == 0:
                    print(f"Pushed {i + 1} samples in current cycle")

            # If not continuous, break after one cycle
            if not continuous:
                print(f"Done pushing {total_size} samples")
                break

            print(f"Completed one cycle of {total_size} samples, continuing...")

def main():
    parser = argparse.ArgumentParser(description='Push dummy data for PyTorch training.')
    parser.add_argument('--address', default='tcp://127.0.0.1:5555', help='Address to push data to (URL format).')
    parser.add_argument('--C', type=int, default=3, help='Number of channels.')
    parser.add_argument('--H', type=int, default=224, help='Height of the tensor.')
    parser.add_argument('--W', type=int, default=224, help='Width of the tensor.')
    parser.add_argument('--total-size', type=int, default=10000, help='Total number of samples to push.')
    parser.add_argument('--push-interval', type=float, default=0.001, help='Interval between pushes in seconds.')
    parser.add_argument('--continuous', action='store_true', help='Continuously push data (loop through indices).')

    args = parser.parse_args()

    dummy_data_pusher(
        args.address,
        args.C,
        args.H,
        args.W,
        args.total_size,
        args.push_interval,
        args.continuous
    )

if __name__ == '__main__':
    main()
