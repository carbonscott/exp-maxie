import sqlite3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import argparse
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def sqlite_to_parquet(db_path, base_dir, output_file, chunk_size=100000):
    try:
        conn = sqlite3.connect(db_path)

        query = """
        SELECT
            zarr_files.file_path AS relative_path,
            zarr_files.shape
        FROM zarr_files
        JOIN experiments ON zarr_files.experiment_id = experiments.id
        """

        schema = pa.schema([
            ('absolute_path', pa.string()),
            ('shape', pa.string())
        ])
        writer = pq.ParquetWriter(output_file, schema)

        offset = 0
        while True:
            chunk_query = f"{query} LIMIT {chunk_size} OFFSET {offset}"
            df_chunk = pd.read_sql_query(chunk_query, conn)

            if df_chunk.empty:
                break

            df_chunk['absolute_path'] = base_dir + '/' + df_chunk['relative_path']

            table_chunk = pa.Table.from_pandas(df_chunk[['absolute_path', 'shape']], schema=schema)

            writer.write_table(table_chunk)

            offset += chunk_size
            logging.info(f"Processed {offset} records")

        writer.close()
        logging.info(f"Successfully created Parquet file: {output_file}")
    except sqlite3.Error as e:
        logging.error(f"SQLite error occurred: {e}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        if conn:
            conn.close()

def main():
    parser = argparse.ArgumentParser(description='Convert SQLite Zarr file paths to Parquet format')
    parser.add_argument('db_path', help='Path to the SQLite database')
    parser.add_argument('base_dir', help='Base directory for Zarr files')
    parser.add_argument('output_file', help='Output Parquet file path')
    parser.add_argument('--chunk_size', type=int, default=100000, help='Number of records to process at a time')

    args = parser.parse_args()

    # Check if the database file exists
    if not os.path.exists(args.db_path):
        logging.error(f"Database file not found: {args.db_path}")
        return

    # Check if the base directory exists
    if not os.path.exists(args.base_dir):
        logging.error(f"Base directory not found: {args.base_dir}")
        return

    # Check if the output directory exists, create if not
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created output directory: {output_dir}")

    sqlite_to_parquet(args.db_path, args.base_dir, args.output_file, args.chunk_size)

if __name__ == '__main__':
    main()
