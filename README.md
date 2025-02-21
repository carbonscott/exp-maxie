# Training Stub Documentation

A distributed training implementation supporting DDP, ZeRO-2, and ZeRO-3 data parallel strategies.

## Dependencies

**Install PyTorch**:
- Fresh install on S3DF
- Follow [this link](https://docs.olcf.ornl.gov/software/analytics/pytorch_frontier.html#installing-pytorch) to install it on Frontier

[**Install maxie**](https://github.com/carbonscott/maxie#pip)

```bash
pip install transformers
pip install omegaconf colorama
```

## Quick Start

> **Required if running on Frontier**
>
> Before running the code, execute these commands to set up the MIOpen cache:
> ```bash
> export MIOPEN_USER_DB_PATH="/tmp/$(openssl rand -base64 12 | head -c 16)-miopen-cache"
> export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
> rm -rf ${MIOPEN_USER_DB_PATH}
> mkdir -p ${MIOPEN_USER_DB_PATH}
> ```
>
> If you need to install AWS-OFI-RCCL Plugin, please refer to this [link](https://docs.olcf.ornl.gov/software/analytics/pytorch_frontier.html#aws-ofi-rccl-plugin).
> You have to run
> ```bash
> export LD_LIBRARY_PATH=${PATH TO THE PLUGIN}/lib/:${LD_LIBRARY_PATH}
> ```
> The installation guide will show you the exact `export LD_LIBRARY_PATH=...`
> that you can cut, paste and run.  For example, mine looks like 
> ```bash
> export LD_LIBRARY_PATH=/lustre/orion/mph121/proj-shared/cwang31/packages/aws-ofi-rccl/lib:$LD_LIBRARY_PATH
> ```

### Single Process
```bash
# S3DF/ada
python train_stub.py test_config_training.yaml

# Frontier
NCCL_NET_GDR_LEVEL=3 NCCL_ALGO="TREE or RING" NCCL_CROSS_NIC=1 OMP_NUM_THREADS=1 NCCL_SOCKET_IFNAME=hsn0 MASTER_PORT=3442 TRANSFORMERS_CACHE=.cache/huggingface/hub/ MASTER_ADDR=$(hostname -i) python train_stub.py test_config_training.yaml
```

### Distributed Training
SLURM:
```bash
# 1 node, 10 GPUs per node (S3DF/ada)
MASTER_ADDR=$(hostname) srun --mpi=none --nodes 1 --ntasks-per-node 10 --gpus-per-node 10 bash -c "python train_stub.py test_config_training.yaml"

# 2 nodes, 8 GPUs per node (Frontier)
NCCL_NET_GDR_LEVEL=3 NCCL_ALGO="TREE or RING" NCCL_CROSS_NIC=1 OMP_NUM_THREADS=1 NCCL_SOCKET_IFNAME=hsn0 MASTER_PORT=3442 TRANSFORMERS_CACHE=.cache/huggingface/hub/ MASTER_ADDR=$(hostname -i) srun --gres=gpu:8  --gpu-bind=closest --nodes 2 --ntasks-per-node 8 --gpus-per-node 8 bash -c "python train_stub.py test_config_training.yaml"
```

MPI:
```bash
# S3DF/ada
mpirun -n 10 python train_stub.py test_config_training.yaml
```

![figures/multi_node.png]

## Configuration

### Key Sections

#### Distributed Training
```yaml
dist:
  cpu_only: False
  backend: nccl
  dtype: bfloat16
  sharding_stage: zero3  # Options: zero0 (DDP), zero2 (SHARD_GRAD_OP), zero3 (FULL_SHARD)
                         #                       zero2_hybrid,          zero3_hybrid
                         # Hybrid means DDP across nodes and FSDP within a node
```

#### Dataset Configuration
```yaml
dataset:
  batch_size: 256
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
