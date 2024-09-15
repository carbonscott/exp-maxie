import os
import sqlite3
import argparse
import zarr
import logging
from tqdm import tqdm
from multiprocessing import Process, Queue, Value, Lock
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_database(db_name):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Create experiments table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS experiments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE
    )
    ''')

    # Create zarr_files table with corrected schema
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS zarr_files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        experiment_id INTEGER NOT NULL,
        run TEXT NOT NULL,
        file_path TEXT NOT NULL,
        shape TEXT,
        FOREIGN KEY (experiment_id) REFERENCES experiments (id),
        UNIQUE (experiment_id, run, file_path)
    )
    ''')

    # Create indices for faster lookups
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_zarr_files_experiment ON zarr_files (experiment_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_zarr_files_run ON zarr_files (run)')

    conn.commit()
    return conn

def get_zarr_shape(file_path):
    try:
        z = zarr.open(file_path, mode='r')
        if 'data' in z:
            return str(z['data'].shape)
    except Exception as e:
        logging.warning(f"Error reading Zarr file {file_path}: {e}")
    return None

def worker_process(task_queue, result_queue, root_dir, progress_counter, progress_lock):
    while True:
        task = task_queue.get()
        if task is None:  # Signal process termination
            break

        experiment, run = task
        run_path = os.path.join(root_dir, experiment, run)
        results = []

        for file in os.listdir(run_path):
            if file.endswith('.zarr'):
                file_path = os.path.join(experiment, run, file)
                full_path = os.path.join(root_dir, file_path)
                shape = get_zarr_shape(full_path)
                results.append((experiment, run, file_path, shape))

        result_queue.put(results)

        with progress_lock:
            progress_counter.value += 1

def db_writer_process(result_queue, db_name, total_tasks):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    processed_tasks = 0
    experiment_cache = {}  # Cache to store experiment_id for each experiment name

    while processed_tasks < total_tasks:
        results = result_queue.get()
        if results is None:
            continue

        for experiment, run, file_path, shape in results:
            # Check if we already have the experiment_id cached
            if experiment not in experiment_cache:
                # Insert experiment if it doesn't exist
                cursor.execute('INSERT OR IGNORE INTO experiments (name) VALUES (?)', (experiment,))

                # Get the experiment_id (whether it was just inserted or already existed)
                cursor.execute('SELECT id FROM experiments WHERE name = ?', (experiment,))
                exp_id = cursor.fetchone()[0]
                experiment_cache[experiment] = exp_id
            else:
                exp_id = experiment_cache[experiment]

            # Insert zarr file
            cursor.execute('INSERT OR IGNORE INTO zarr_files (experiment_id, run, file_path, shape) VALUES (?, ?, ?, ?)', 
                           (exp_id, run, file_path, shape))

        conn.commit()
        processed_tasks += 1

    conn.close()

def main():
    parser = argparse.ArgumentParser(description='Convert Zarr directory structure to SQLite database with shape information using multiprocessing.')
    parser.add_argument('root_dir', help='Root directory containing the Zarr file structure')
    parser.add_argument('--db_name', default='zarr_files_with_shape_mp.db', help='Name of the SQLite database file')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker processes')

    args = parser.parse_args()

    logging.info(f"Starting the Zarr to SQLite conversion process with {args.num_workers} workers")

    # Create database
    create_database(args.db_name)

    # Create queues
    task_queue = Queue()
    result_queue = Queue()

    # Create shared progress counter and lock
    progress_counter = Value('i', 0)
    progress_lock = Lock()

    # Collect tasks
    tasks = []
    for experiment in os.listdir(args.root_dir):
        exp_path = os.path.join(args.root_dir, experiment)
        if os.path.isdir(exp_path):
            for run in os.listdir(exp_path):
                run_path = os.path.join(exp_path, run)
                if os.path.isdir(run_path):
                    tasks.append((experiment, run))

    total_tasks = len(tasks)
    logging.info(f"Total number of tasks: {total_tasks}")

    # Start worker processes
    workers = []
    for _ in range(args.num_workers):
        p = Process(target=worker_process, args=(task_queue, result_queue, args.root_dir, progress_counter, progress_lock))
        workers.append(p)
        p.start()

    # Start database writer process
    db_writer = Process(target=db_writer_process, args=(result_queue, args.db_name, total_tasks))
    db_writer.start()

    # Add tasks to the queue
    for task in tasks:
        task_queue.put(task)

    # Add poison pills to stop workers
    for _ in range(args.num_workers):
        task_queue.put(None)

    # Monitor progress
    with tqdm(total=total_tasks, desc="Processing tasks") as pbar:
        last_count = 0
        while progress_counter.value < total_tasks:
            current_count = progress_counter.value
            pbar.update(current_count - last_count)
            last_count = current_count
            time.sleep(0.1)

    # Wait for all processes to finish
    for worker in workers:
        worker.join()

    db_writer.join()

    logging.info(f"Database '{args.db_name}' has been created and populated with the Zarr file structure and shape information from '{args.root_dir}'.")

if __name__ == '__main__':
    main()
