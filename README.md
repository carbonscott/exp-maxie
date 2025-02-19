# Training Stub Documentation

A distributed training implementation supporting DDP, ZeRO-2, and ZeRO-3 optimization strategies.

## Quick Start

### Single Process
```bash
python train_stub.py test_config_training.yaml
```

### Distributed Training
SLURM:
```bash
MASTER_ADDR=$(hostname) srun --mpi=none --nodes 1 --ntasks-per-node 10 --gpus-per-node 10 bash -c "python train_stub.py test_config_training.yaml"
```

MPI:
```bash
mpirun -n 10 python train_stub.py test_config_training.yaml
```

## Configuration

### Key Sections

#### Distributed Training
```yaml
dist:
  cpu_only: False
  backend: nccl
  dtype: bfloat16
  sharding_stage: zero3  # Options: zero0 (DDP), zero2 (SHARD_GRAD_OP), zero3 (FULL_SHARD)
```

#### Dataset Configuration
```yaml
dataset:
  batch_size: 32
  num_workers: 1
  path_train: experiments/datasets/safetensor_dataset.train.csv
  path_eval: experiments/datasets/safetensor_dataset.validate.csv
```

Note: The dataset paths are placeholders. The stub generates random tensor data to verify training pipeline functionality.

#### Model Architecture
```yaml
model:
  hf_config:
    hidden_size: 1280
    num_hidden_layers: 32
    num_attention_heads: 16
    # ... other architecture parameters
```

#### Training Parameters
```yaml
optim:
  lr: 0.0003
  weight_decay: 0.05
lr_scheduler:
  total_iterations: 1000000
  warmup_iterations: 10
```

## Checkpoint System

Checkpoints are saved at:
```
experiments/chkpts/<prefix>.<timestamp>
experiments/chkpts/<prefix>.<timestamp>.preempt/
```
preempt means the checkpoint is saved preemptively to avoid interruption.

Each checkpoint contains:
- `model_state_dict.pt`: Model weights
- `optim_state_dict.pt`: Optimizer state
- `lr_state_dict.pt`: Learning rate scheduler state
- `iter_state_dict.pt`: Training iteration metadata including:
  - Current epoch
  - Segment information
    - Which segment
    - Start index of this segment
    - End index of this segment
  - Best loss achieved

### Checkpoint Configuration
```yaml
checkpoint:
  state_dict_type: full
  chkpt_saving_iterations: 100
  directory: experiments/chkpts
  prefix: dummy-0.1
```

## Distributed Training Details

The training supports three distributed strategies:
- `zero0`: Traditional DistributedDataParallel (DDP)
- `zero2`: Gradient & optimizer state sharding
- `zero3`: Full model, gradient, and optimizer state sharding

Select the strategy via `dist.sharding_stage` in the config file.
