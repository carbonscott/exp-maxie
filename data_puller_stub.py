import time
import argparse
import io
import numpy as np
import torch
import ast
from pynng import Pull0, Timeout

def bytes_to_tensor(data):
    """Convert bytes to a PyTorch tensor."""
    # Skip metadata (everything before the first newline)
    newline_index = data.find(b'\n')
    if newline_index != -1:
        data = data[newline_index + 1:]

    try:
        buffer = io.BytesIO(data)
        tensor_np = np.load(buffer)
        return torch.from_numpy(tensor_np)
    except Exception as e:
        print(f"Error deserializing tensor: {e}")
        return None

def simple_puller(address, is_dial=True, test_tensors=5, report_interval=1.0, timeout_ms=1000):
    """
    Pull data from the specified address and display statistics.

    Args:
        address (str): Address to pull data from (URL format).
        is_dial (bool): Whether to dial to the address (otherwise listen).
        test_tensors (int): Number of tensors to display in detail (-1 for none).
        report_interval (float): How often to report statistics, in seconds.
        timeout_ms (int): Socket timeout in milliseconds.
    """
    sock_args = {"dial": address} if is_dial else {"listen": address}

    try:
        with Pull0(**sock_args) as sock:
            sock.recv_timeout = timeout_ms
            print(f"{'Dialing to' if is_dial else 'Listening at'} {address}")

            # Statistics
            count = 0
            total_bytes = 0

            # Two time tracking mechanisms
            start_time = time.time()
            last_report_time = start_time

            # Track active receiving time (excluding timeouts)
            active_time = 0
            last_active_time = start_time

            # Track window statistics (for recent performance)
            window_start_time = start_time
            window_count = 0
            window_bytes = 0
            window_active_time = 0

            # Interruption tracking
            interruption_count = 0
            last_interruption_time = None
            in_interruption = False
            total_interruption_time = 0
            current_interruption_start = None

            while True:
                try:
                    # Pull data
                    data = sock.recv()
                    current_time = time.time()

                    # If we were in an interruption, calculate its duration
                    if in_interruption:
                        interruption_duration = current_time - current_interruption_start
                        total_interruption_time += interruption_duration
                        print(f"\nResume receiving after {interruption_duration:.2f}s interruption")
                        in_interruption = False
                        current_interruption_start = None

                    # Update activity time
                    active_time += (current_time - last_active_time)
                    window_active_time += (current_time - last_active_time)
                    last_active_time = current_time

                    # Update statistics
                    count += 1
                    window_count += 1
                    data_size = len(data)
                    total_bytes += data_size
                    window_bytes += data_size

                    # Extract metadata
                    newline_index = data.find(b'\n')
                    if newline_index != -1:
                        metadata_str = data[:newline_index].decode('utf-8')
                        try:
                            metadata = eval(metadata_str)

                            # Display the first few tensors in detail
                            if test_tensors > 0:
                                tensor = bytes_to_tensor(data)
                                if tensor is not None:
                                    print(f"\nReceived tensor {count}:")
                                    print(f"  Metadata: {metadata}")
                                    print(f"  Shape: {tensor.shape}")
                                    print(f"  Min/Max/Mean: {tensor.min():.3f}/{tensor.max():.3f}/{tensor.mean():.3f}")
                                    test_tensors -= 1
                        except Exception as e:
                            print(f"Error parsing metadata: {e}")

                    # Report statistics periodically
                    if current_time - last_report_time >= report_interval:
                        total_elapsed = current_time - start_time
                        window_elapsed = current_time - window_start_time

                        # Calculate rates based on active time (excluding interruptions)
                        # This gives accurate throughput regardless of interruptions
                        if active_time > 0:
                            active_rate = total_bytes / active_time / (1024 * 1024)  # MB/s
                            active_rate_msgs = count / active_time  # msgs/s
                        else:
                            active_rate = 0
                            active_rate_msgs = 0

                        # Calculate window rates (recent performance)
                        if window_active_time > 0:
                            window_rate = window_bytes / window_active_time / (1024 * 1024)  # MB/s
                            window_rate_msgs = window_count / window_active_time  # msgs/s
                        else:
                            window_rate = 0
                            window_rate_msgs = 0

                        print(f"\nStatistics after {total_elapsed:.1f} seconds (active time: {active_time:.1f}s):")
                        print(f"  Received {count} messages ({active_rate_msgs:.1f} msgs/s based on active time)")
                        print(f"  Transferred {total_bytes / (1024*1024):.2f} MB ({active_rate:.2f} MB/s based on active time)")
                        print(f"  Recent performance: {window_rate:.2f} MB/s, {window_rate_msgs:.1f} msgs/s")
                        print(f"  Interruptions: {interruption_count} (total {total_interruption_time:.2f}s)")

                        # Reset window statistics
                        window_start_time = current_time
                        window_count = 0
                        window_bytes = 0
                        window_active_time = 0

                        last_report_time = current_time

                except Timeout:
                    # Track the beginning of an interruption
                    current_time = time.time()
                    if not in_interruption:
                        current_interruption_start = current_time
                        in_interruption = True
                        interruption_count += 1
                    print(".", end="", flush=True)

                    # Don't update last_active_time during timeouts
                    continue

                except KeyboardInterrupt:
                    # Exit on Ctrl+C
                    break

                except Exception as e:
                    print(f"\nError receiving data: {e}")
                    # Don't count this as active time
                    last_active_time = time.time()
                    continue

            # Final statistics
            current_time = time.time()
            total_elapsed = current_time - start_time

            # If we were in an interruption at the end, add its duration
            if in_interruption:
                interruption_duration = current_time - current_interruption_start
                total_interruption_time += interruption_duration

            # Calculate final rates based on active time
            if active_time > 0:
                active_rate = total_bytes / active_time / (1024 * 1024)  # MB/s
                active_rate_msgs = count / active_time  # msgs/s
            else:
                active_rate = 0
                active_rate_msgs = 0

            # Also calculate rates based on total time for comparison
            total_rate = total_bytes / total_elapsed / (1024 * 1024) if total_elapsed > 0 else 0  # MB/s
            total_rate_msgs = count / total_elapsed if total_elapsed > 0 else 0  # msgs/s

            print(f"\nFinal statistics after {total_elapsed:.1f} seconds:")
            print(f"  Active time: {active_time:.1f} seconds ({(active_time/total_elapsed*100):.1f}% of total)")
            print(f"  Interruptions: {interruption_count} (total {total_interruption_time:.2f}s)")
            print(f"  Received {count} messages")
            print(f"  Throughput based on active time: {active_rate:.2f} MB/s, {active_rate_msgs:.1f} msgs/s")
            print(f"  Throughput based on total time: {total_rate:.2f} MB/s, {total_rate_msgs:.1f} msgs/s")
            print(f"  Total transferred: {total_bytes / (1024*1024):.2f} MB")

    except Exception as e:
        print(f"Error setting up socket: {e}")

def main():
    parser = argparse.ArgumentParser(description='Pull data from a pynng socket and display statistics.')
    parser.add_argument('--address', default='tcp://127.0.0.1:5555', 
                        help='Address to pull data from (URL format)')
    parser.add_argument('--listen', action='store_true', 
                        help='Listen at the address instead of dialing to it')
    parser.add_argument('--test-tensors', type=int, default=5, 
                        help='Number of tensors to display in detail (-1 for none)')
    parser.add_argument('--report-interval', type=float, default=1.0, 
                        help='How often to report statistics, in seconds')
    parser.add_argument('--timeout', type=int, default=1000, 
                        help='Socket timeout in milliseconds')

    args = parser.parse_args()

    simple_puller(
        args.address,
        not args.listen,  # is_dial is True if not listening
        args.test_tensors,
        args.report_interval,
        args.timeout
    )

if __name__ == '__main__':
    main()
