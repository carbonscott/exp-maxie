#!/bin/bash
#SBATCH --output=slurm/{{ job }}.%j.log
#SBATCH --error=slurm/{{ job }}.%j.err
#SBATCH --account lcls_g
#SBATCH --time {{ walltime }}          # Regular only allows a max of 12 hours.  See https://docs.nersc.gov/jobs/policy/
#SBATCH --exclusive
#SBATCH --job-name={{ job }}
#SBATCH --qos {{ qos }}
#SBATCH --constraint gpu
#SBATCH --nodes={{ num_nodes }}
#SBATCH --ntasks-per-node={{ num_gpus_per_node }}
#SBATCH --cpus-per-task={{ num_cpus_per_task }}
#SBATCH --gpus-per-task=1
#SBATCH --gpus-per-node={{ num_gpus_per_node }}

export SLURM_CPU_BIND="cores"
export NCCL_DEBUG=INFO

# Reversing order of GPUs to match default CPU affinities from Slurm
export CUDA_VISIBLE_DEVICES=3,2,1,0

## module load python
## module load pytorch/2.3.1
module load conda
conda activate /global/u2/c/cwang31/data_root/conda/envs/p310

# Set up the Huggingface's cache directory
export TRANSFORMERS_CACHE={{ transformers_cache }}

export OMP_NUM_THREADS={{ OMP_NUM_THREADS }}
export NCCL_DEBUG=INFO
export TORCH_NCCL_BLOCKING_WAIT=1

# Set up a meta checkpoint file
export PREEMPT_ROOT="preempt"
mkdir -p $PREEMPT_ROOT
export PREEMPT_METADATA_PATH="$PREEMPT_ROOT/{{ job }}"

# Check if a checkpoint exists and resume from it
if [ -f $PREEMPT_METADATA_PATH ]; then
    echo "Resuming from preemptive checkpoint..."
    python -c "import yaml, os; data = yaml.safe_load(open('{{ yaml_config }}')); data['checkpoint']['path_chkpt_prev'] = open(os.getenv('PREEMPT_METADATA_PATH')).read().strip(); yaml.safe_dump(data, open('{{ yaml_config }}', 'w'))"
fi

## nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
## head_node=${nodes[0]}
## head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
## head_node_ip=$(echo "$head_node_ip" | awk '{print $1}')

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500  # Ensure this port is free or choose another

## export RANK=$SLURM_PROCID
## export LOCAL_RANK=$SLURM_LOCALID
## export WORLD_SIZE=$SLURM_NTASKS
## echo "Master node: $MASTER_ADDR"
## echo "Master port: $MASTER_PORT"

set -x

# Use torchrun to launch the distributed training
srun bash -c "python {{trainer}} {{yaml_config}}"
