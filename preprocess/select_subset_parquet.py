import argparse
import os
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Export first N items from a Parquet file, with optional shuffling.")
    parser.add_argument("input_file", help="Input Parquet file path")
    parser.add_argument("num_items", type=int, help="Number of items to export")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle the DataFrame before selecting items")
    args = parser.parse_args()

    # Read the input Parquet file
    df = pd.read_parquet(args.input_file)

    if args.shuffle:
        # Shuffle the entire DataFrame
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Take the first N items
    df_subset = df.head(args.num_items)

    # Generate the output file name
    input_name = os.path.splitext(os.path.basename(args.input_file))[0]
    shuffle_suffix = "_shuffled" if args.shuffle else ""
    output_file = f"{input_name}.sub{args.num_items}{shuffle_suffix}.parquet"

    # Save the subset to a new Parquet file
    df_subset.to_parquet(output_file, index=False)
    print(f"Exported {args.num_items} items to {output_file}")

if __name__ == "__main__":
    main()
