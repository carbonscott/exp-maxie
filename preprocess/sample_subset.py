import os
import zarr
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
import argparse
import logging
from maxie.datasets.zarr_dataset import DistributedZarrDataset
from maxie.utils.seed import set_seed
from maxie.tensor_transforms import (
    NoTransform,
    PolarCenterCrop,
    MergeBatchPatchDims,
    Pad,
)
import torch

# Get the logger
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------- #
#  COMMAND LINE INTERFACE
# ----------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description="Load training configuration and create datasets.")
parser.add_argument("yaml_file", help="Path to the YAML file")
parser.add_argument("--output_name", default="subset", help="Base name for output files (default: overfit_this)")
parser.add_argument("--num_images", type=int, default=100, help="Number of images to process (default: 100)")
args = parser.parse_args()

# ----------------------------------------------------------------------- #
#  CONFIGURATION
# ----------------------------------------------------------------------- #
# Load CONFIG from YAML
with open(args.yaml_file, 'r') as fh:
    config = yaml.safe_load(fh)

# -- Dataset
dataset_config = config.get("dataset")
path_train = dataset_config.get("path_train")
batch_size = dataset_config.get("batch_size")
num_workers = dataset_config.get("num_workers")
seg_size = dataset_config.get("seg_size")
pin_memory = dataset_config.get("pin_memory")
prefetch_factor = dataset_config.get("prefetch_factor")
transforms_config = dataset_config.get("transforms")
set_transforms = transforms_config.get("set")

# Extract only the necessary transform parameters
Hv = transforms_config.get("Hv")
Wv = transforms_config.get("Wv")
H_pad = transforms_config.get("H_pad")
W_pad = transforms_config.get("W_pad")
sigma = transforms_config.get("sigma")
num_crop = transforms_config.get("num_crop")

uses_pad = set_transforms.get("pad")
uses_polar_center_crop = set_transforms.get("polar_center_crop")

# ----------------------------------------------------------------------- #
#  DATASET
# ----------------------------------------------------------------------- #
# -- Seeding
base_seed = 0
set_seed(base_seed)

# -- Set up transformation
merges_batch_patch_dims = uses_polar_center_crop
pre_transforms = (
    Pad(H_pad, W_pad) if uses_pad else NoTransform(),
    PolarCenterCrop(
        Hv=Hv,
        Wv=Wv,
        sigma=sigma,
        num_crop=num_crop,
    ) if uses_polar_center_crop else NoTransform(),
    MergeBatchPatchDims() if merges_batch_patch_dims else NoTransform(),
)

# -- Set up training set
dataset_train = DistributedZarrDataset(
    path_train,
    seg_size=seg_size,
    transforms=pre_transforms,
    seed=base_seed
)

dataloader = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=pin_memory,
    prefetch_factor=prefetch_factor,
)

# ----------------------------------------------------------------------- #
#  ZARR
# ----------------------------------------------------------------------- #
def create_zarr_dataset(store, num_images, image_shape):
    return store.create_dataset(
        "data",
        shape=(num_images, *image_shape),
        chunks=(1, *image_shape),
        dtype=np.float32,
        compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.SHUFFLE)
    )

filepath = f"{args.output_name}.zarr"
image_shape = (Hv, Wv)
store = zarr.open(filepath, mode='w')
ds = create_zarr_dataset(store, args.num_images, image_shape)

for idx, data in enumerate(dataloader):
    if idx >= args.num_images: break
    ds[idx] = data.numpy()[0, 0]

# ----------------------------------------------------------------------- #
#  PARQUET
# ----------------------------------------------------------------------- #
output_file = f'{args.output_name}.parquet'
schema = pa.schema([
    ('absolute_path', pa.string()),
    ('shape', pa.string())
])

zarr_absolute_path = os.path.abspath(f'{args.output_name}.zarr')
writer = pq.ParquetWriter(output_file, schema)
df = pd.DataFrame({
    'absolute_path': [zarr_absolute_path],
    'shape': [f'{image_shape}'],
})

table_chunk = pa.Table.from_pandas(df[['absolute_path', 'shape']], schema=schema)
writer.write_table(table_chunk)
writer.close()

print(f"Zarr file created at: {zarr_absolute_path}")
print(f"Parquet file created at: {os.path.abspath(output_file)}")
