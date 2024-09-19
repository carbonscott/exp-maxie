import os
import random
import pyarrow.parquet as pq
import pandas as pd
import argparse

def split_dataset(data, fracA=0.8, seed=None):
    """
    Split a list or DataFrame into training and testing sets.

    :param data: List or DataFrame to split
    :param fracA: Ratio of data to use for training (default 0.8 for 80/20 split)
    :param seed: Seed for random number generator (optional)
    :return: Tuple of (train_data, test_data)
    """
    if seed is not None:
        random.seed(seed)

    data_size = len(data)
    train_size = int(data_size * fracA)

    indices = list(range(data_size))
    random.shuffle(indices)

    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_data = data.iloc[train_indices]
    test_data = data.iloc[test_indices]

    return train_data, test_data

def main():
    parser = argparse.ArgumentParser(description='Split parquet dataset')
    parser.add_argument('parquet_path', help='Parquet path')
    parser.add_argument('train_frac', help='Training set fraction')

    args = parser.parse_args()

    base_path = args.parquet_path[:args.parquet_path.rfind('.')]

    table = pq.read_table(args.parquet_path)
    df = table.to_pandas()

    df_A, df_B = split_dataset(df, float(args.train_frac), seed = 42)

    path_A = base_path + '.train.parquet'
    path_B = base_path + '.val.parquet'

    df_A.to_parquet(path_A, index=False)
    df_B.to_parquet(path_B, index=False)

if __name__ == '__main__':
    main()
