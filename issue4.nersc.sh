#!/bin/bash

RUNS_NSYS=0
## NUM_MPI_TASKS=4

# [SLURM]
JOB=nersc-issue4-huge
NUM_NODES=4
NUM_TASKS=16
NUM_GPUS_PER_NODE=4
NUM_CPUS_PER_TASK=2
TRAINER='train.fsdp.nersc.py'
PARTITION=""
WALLTIME="48:00:00"
QOS="regular"
EXCLUDE_NODES=""

# [TRAIN]
H_PAD=980
W_PAD=980
Hv=980
Wv=980
NUM_CROP=1
BATCH_SIZE=1
NUM_WORKERS=2
USES_PAD=true
USES_POLAR_CENTER_CROP=true
USES_INSTANCE_NORM=true
USES_BATCH_SAMPLER=false
USES_RANDOM_PATCH=false
USES_RANDOM_ROTATE=false
USES_RANDOM_SHIFT=false
PATH_CHKPT_PREV=null
PREEEMPT_CHKPT_SAVING_ITERATIONS=10
CHKPT_SAVING_ITERATIONS=20
SHARDING_STAGE=zero3
WARMUP=20
OFFLOAD_TO_CPU=false

# [MODEL]
MODEL_HIDDEN_SIZE=1280
MODEL_NUM_HIDDEN_LAYERS=32
MODEL_NUM_ATTENTION_HEADS=16
MODEL_INTERMEDIATE_SIZE=5120
MODEL_HIDDEN_ACT="gelu"
MODEL_HIDDEN_DROPOUT_PROB=0.0
MODEL_ATTENTION_PROBS_DROPOUT_PROB=0.0
MODEL_INITIALIZER_RANGE=0.02
MODEL_IMAGE_SIZE=980
MODEL_PATCH_SIZE=14
MODEL_NUM_CHANNELS=1
MODEL_QKV_BIAS=true
MODEL_DECODER_NUM_ATTENTION_HEADS=16
MODEL_DECODER_HIDDEN_SIZE=512
MODEL_DECODER_NUM_HIDDEN_LAYERS=8
MODEL_DECODER_INTERMEDIATE_SIZE=2048
MODEL_MASK_RATIO=0.75
MODEL_NORM_PIX_LOSS=true

# [DATASET]
PATH_TRAIN='preprocess/maxie_peaknet20k.train.parquet'
PATH_VAL='preprocess/maxie_peaknet20k.val.parquet'

PREEMPT_ROOT="preempt"
mkdir -p $PREEMPT_ROOT
PREEMPT_METADATA_PATH="$PREEMPT_ROOT/$JOB"

SEG_SIZE=$((BATCH_SIZE * 60))

python launch_job.nersc.py \
job=$JOB \
auto_submit=true \
path.file_sbatch_template=template.nersc.sbatch \
sbatch_config.trainer=$TRAINER \
sbatch_config.num_nodes=$NUM_NODES \
sbatch_config.num_tasks=$NUM_TASKS \
sbatch_config.partition=$PARTITION \
sbatch_config.walltime=$WALLTIME \
sbatch_config.num_gpus_per_node=$NUM_GPUS_PER_NODE \
sbatch_config.num_cpus_per_task=$NUM_CPUS_PER_TASK \
sbatch_config.exclude_nodes=$EXCLUDE_NODES \
sbatch_config.qos=$QOS \
train_config.checkpoint.prefix=$JOB \
train_config.checkpoint.path_chkpt_prev=$PATH_CHKPT_PREV \
train_config.checkpoint.state_dict_type=full \
train_config.checkpoint.preempt_metadata_path=$PREEMPT_METADATA_PATH \
train_config.checkpoint.preempt_chkpt_saving_iterations=$PREEEMPT_CHKPT_SAVING_ITERATIONS \
train_config.checkpoint.chkpt_saving_iterations=$CHKPT_SAVING_ITERATIONS \
train_config.checkpoint.offload_to_cpu=$OFFLOAD_TO_CPU \
train_config.dataset.path_train=$PATH_TRAIN \
train_config.dataset.path_val=$PATH_VAL \
train_config.dataset.num_workers=$NUM_WORKERS \
train_config.dataset.prefetch_factor=10 \
train_config.dataset.pin_memory=true \
train_config.dataset.seg_size=$SEG_SIZE \
train_config.loss.grad_accum_steps=10 \
train_config.dataset.batch_size=$BATCH_SIZE \
train_config.dataset.transforms.set.pad=$USES_PAD \
train_config.dataset.transforms.set.polar_center_crop=$USES_POLAR_CENTER_CROP \
train_config.dataset.transforms.set.instance_norm=$USES_INSTANCE_NORM \
train_config.dataset.transforms.set.batch_sampler=$USES_BATCH_SAMPLER \
train_config.dataset.transforms.set.random_patch=$USES_RANDOM_PATCH \
train_config.dataset.transforms.set.random_rotate=$USES_RANDOM_ROTATE \
train_config.dataset.transforms.set.random_shift=$USES_RANDOM_SHIFT \
train_config.dataset.transforms.H_pad=$H_PAD \
train_config.dataset.transforms.W_pad=$W_PAD \
train_config.dataset.transforms.Hv=$Hv \
train_config.dataset.transforms.Wv=$Wv \
train_config.dataset.transforms.num_crop=$NUM_CROP \
"train_config.model.hf_config.hidden_size=$MODEL_HIDDEN_SIZE" \
"train_config.model.hf_config.num_hidden_layers=$MODEL_NUM_HIDDEN_LAYERS" \
"train_config.model.hf_config.num_attention_heads=$MODEL_NUM_ATTENTION_HEADS" \
"train_config.model.hf_config.intermediate_size=$MODEL_INTERMEDIATE_SIZE" \
"train_config.model.hf_config.hidden_act=$MODEL_HIDDEN_ACT" \
"train_config.model.hf_config.hidden_dropout_prob=$MODEL_HIDDEN_DROPOUT_PROB" \
"train_config.model.hf_config.attention_probs_dropout_prob=$MODEL_ATTENTION_PROBS_DROPOUT_PROB" \
"train_config.model.hf_config.initializer_range=$MODEL_INITIALIZER_RANGE" \
"train_config.model.hf_config.image_size=$MODEL_IMAGE_SIZE" \
"train_config.model.hf_config.patch_size=$MODEL_PATCH_SIZE" \
"train_config.model.hf_config.num_channels=$MODEL_NUM_CHANNELS" \
"train_config.model.hf_config.qkv_bias=$MODEL_QKV_BIAS" \
"train_config.model.hf_config.decoder_num_attention_heads=$MODEL_DECODER_NUM_ATTENTION_HEADS" \
"train_config.model.hf_config.decoder_hidden_size=$MODEL_DECODER_HIDDEN_SIZE" \
"train_config.model.hf_config.decoder_num_hidden_layers=$MODEL_DECODER_NUM_HIDDEN_LAYERS" \
"train_config.model.hf_config.decoder_intermediate_size=$MODEL_DECODER_INTERMEDIATE_SIZE" \
"train_config.model.hf_config.mask_ratio=$MODEL_MASK_RATIO" \
"train_config.model.hf_config.norm_pix_loss=$MODEL_NORM_PIX_LOSS" \
train_config.optim.lr=0.0003 \
train_config.optim.fused=false \
train_config.misc.monitors_dynamics=false \
train_config.misc.compiles_model=false \
train_config.misc.max_epochs=200 \
train_config.misc.max_eval_iter=40 \
train_config.misc.sharding_stage=$SHARDING_STAGE \
train_config.misc.peak_flops_per_sec=312000000000000 \
train_config.lr_scheduler.warmup_iterations=$WARMUP \
train_config.lr_scheduler.total_iterations=3200 \
train_config.logging.prefix=$JOB \
train_config.dist.dtype=bfloat16

## base_command="mpirun -n $NUM_MPI_TASKS python train.fsdp.dummy_dataset.py experiments/yaml/$JOB.yaml"
## final_command="OMP_NUM_THREADS=1 "
## 
## if [ $RUNS_NSYS -eq 1 ]; then
##     final_command+="nsys profile -w true -t cuda,mpi --mpi-impl=openmpi --wait primary -o /sdf/data/lcls/ds/prj/prjcwang31/results/proj-peaknet/nsight_reports/$JOB.profile -f true --cudabacktrace=true -x true "
## fi
## final_command+="$base_command"
## 
## eval $final_command
