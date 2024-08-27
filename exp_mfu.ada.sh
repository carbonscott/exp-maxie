#!/bin/bash

RUNS_NSYS=0
NUM_MPI_TASKS=10

JOB=dummy-0.0
INPUT_H=1440
INPUT_W=1440
H_PAD=1440
W_PAD=1440
Hv=256
Wv=256
NUM_CROP=1
MODEL_IMAGE_SIZE=1440
BATCH_SIZE=2
NUM_WORKERS=2
USES_PAD=true
USES_POLAR_CENTER_CROP=false
USES_BATCH_SAMPLER=false
USES_RANDOM_PATCH=false
USES_RANDOM_ROTATE=false
USES_RANDOM_SHIFT=false
NUM_HIDDEN_LAYERS_ENCODER=28
SHARDING_STAGE=zero0

SEG_SIZE=$((BATCH_SIZE * 60))
TOTAL_SIZE=$((BATCH_SIZE * 1000))

python launch_job.slurm.exp_mfu.py \
job=$JOB \
auto_submit=false \
sbatch_config.trainer=train.fsdp.dummy_dataset.py \
exp_mfu.checkpoint.prefix=$JOB \
exp_mfu.checkpoint.state_dict_type=full \
exp_mfu.checkpoint.preempt_chkpt_saving_iterations=null \
exp_mfu.checkpoint.chkpt_saving_iterations=null \
exp_mfu.dataset.num_workers=$NUM_WORKERS \
exp_mfu.dataset.prefetch_factor=10 \
exp_mfu.dataset.pin_memory=true \
exp_mfu.dataset.seg_size=$SEG_SIZE \
exp_mfu.loss.grad_accum_steps=10 \
exp_mfu.dataset.batch_size=$BATCH_SIZE \
exp_mfu.dataset.input.H=$INPUT_H \
exp_mfu.dataset.input.W=$INPUT_W \
exp_mfu.dataset.input.total_size=$TOTAL_SIZE \
exp_mfu.dataset.transforms.set.pad=$USES_PAD \
exp_mfu.dataset.transforms.set.polar_center_crop=$USES_POLAR_CENTER_CROP \
exp_mfu.dataset.transforms.set.batch_sampler=$USES_BATCH_SAMPLER \
exp_mfu.dataset.transforms.set.random_patch=$USES_RANDOM_PATCH \
exp_mfu.dataset.transforms.set.random_rotate=$USES_RANDOM_ROTATE \
exp_mfu.dataset.transforms.set.random_shift=$USES_RANDOM_SHIFT \
exp_mfu.dataset.transforms.H_pad=$H_PAD \
exp_mfu.dataset.transforms.W_pad=$W_PAD \
exp_mfu.dataset.transforms.Hv=$Hv \
exp_mfu.dataset.transforms.Wv=$Wv \
exp_mfu.dataset.transforms.num_crop=$NUM_CROP \
exp_mfu.model.hf_config.image_size=$MODEL_IMAGE_SIZE \
"exp_mfu.model.hf_config.num_hidden_layers=$NUM_HIDDEN_LAYERS_ENCODER" \
exp_mfu.optim.lr=0.0003 \
exp_mfu.optim.fused=false \
exp_mfu.misc.monitors_dynamics=false \
exp_mfu.misc.compiles_model=false \
exp_mfu.misc.max_eval_iter=10 \
exp_mfu.misc.sharding_stage=$SHARDING_STAGE \
exp_mfu.lr_scheduler.warmup_iterations=10 \
exp_mfu.lr_scheduler.total_iterations=1000000 \
exp_mfu.logging.prefix=$JOB \
exp_mfu.dist.dtype=bfloat16 \

base_command="mpirun -n $NUM_MPI_TASKS python train.fsdp.dummy_dataset.py experiments/yaml/$JOB.yaml"
final_command="OMP_NUM_THREADS=1 "

if [ $RUNS_NSYS -eq 1 ]; then
    final_command+="nsys profile -w true -t cuda,mpi --mpi-impl=openmpi --wait primary -o /sdf/data/lcls/ds/prj/prjcwang31/results/proj-peaknet/nsight_reports/$JOB.profile -f true --cudabacktrace=true -x true "
fi
final_command+="$base_command"

eval $final_command
