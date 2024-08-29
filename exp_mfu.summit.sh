#!/bin/bash

RUNS_NSYS=0
## NUM_MPI_TASKS=4
NNODES=10
QOS=debug
WALLTIME="00:30"

JOB=exp-0.0
H_PAD=1120
W_PAD=1120
Hv=1120
Wv=1120
NUM_CROP=1
BATCH_SIZE=1
NUM_WORKERS=2
USES_PAD=true
USES_POLAR_CENTER_CROP=true
USES_BATCH_SAMPLER=false
USES_RANDOM_PATCH=false
USES_RANDOM_ROTATE=false
USES_RANDOM_SHIFT=false
SHARDING_STAGE=zero3

MODEL_HIDDEN_SIZE=1280
MODEL_NUM_HIDDEN_LAYERS=32
MODEL_NUM_ATTENTION_HEADS=16
MODEL_INTERMEDIATE_SIZE=5120
MODEL_HIDDEN_ACT="gelu"
MODEL_HIDDEN_DROPOUT_PROB=0.0
MODEL_ATTENTION_PROBS_DROPOUT_PROB=0.0
MODEL_INITIALIZER_RANGE=0.02
MODEL_IMAGE_SIZE=1120
MODEL_PATCH_SIZE=14
MODEL_NUM_CHANNELS=1
MODEL_QKV_BIAS=true
MODEL_DECODER_NUM_ATTENTION_HEADS=16
MODEL_DECODER_HIDDEN_SIZE=512
MODEL_DECODER_NUM_HIDDEN_LAYERS=8
MODEL_DECODER_INTERMEDIATE_SIZE=2048
MODEL_MASK_RATIO=0.75
MODEL_NORM_PIX_LOSS=false

SEG_SIZE=$((BATCH_SIZE * 60))
TOTAL_SIZE=$((BATCH_SIZE * 1000000))

python launch_job.py \
job=$JOB \
auto_submit=true \
bsub_config.trainer=train.fsdp.py \
bsub_config.num_gpus_for_client=6 \
bsub_config.num_nodes=$NNODES \
bsub_config.walltime=$WALLTIME \
bsub_config.qos=$QOS \
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
"exp_mfu.model.hf_config.hidden_size=$MODEL_HIDDEN_SIZE" \
"exp_mfu.model.hf_config.num_hidden_layers=$MODEL_NUM_HIDDEN_LAYERS" \
"exp_mfu.model.hf_config.num_attention_heads=$MODEL_NUM_ATTENTION_HEADS" \
"exp_mfu.model.hf_config.intermediate_size=$MODEL_INTERMEDIATE_SIZE" \
"exp_mfu.model.hf_config.hidden_act=$MODEL_HIDDEN_ACT" \
"exp_mfu.model.hf_config.hidden_dropout_prob=$MODEL_HIDDEN_DROPOUT_PROB" \
"exp_mfu.model.hf_config.attention_probs_dropout_prob=$MODEL_ATTENTION_PROBS_DROPOUT_PROB" \
"exp_mfu.model.hf_config.initializer_range=$MODEL_INITIALIZER_RANGE" \
"exp_mfu.model.hf_config.image_size=$MODEL_IMAGE_SIZE" \
"exp_mfu.model.hf_config.patch_size=$MODEL_PATCH_SIZE" \
"exp_mfu.model.hf_config.num_channels=$MODEL_NUM_CHANNELS" \
"exp_mfu.model.hf_config.qkv_bias=$MODEL_QKV_BIAS" \
"exp_mfu.model.hf_config.decoder_num_attention_heads=$MODEL_DECODER_NUM_ATTENTION_HEADS" \
"exp_mfu.model.hf_config.decoder_hidden_size=$MODEL_DECODER_HIDDEN_SIZE" \
"exp_mfu.model.hf_config.decoder_num_hidden_layers=$MODEL_DECODER_NUM_HIDDEN_LAYERS" \
"exp_mfu.model.hf_config.decoder_intermediate_size=$MODEL_DECODER_INTERMEDIATE_SIZE" \
"exp_mfu.model.hf_config.mask_ratio=$MODEL_MASK_RATIO" \
"exp_mfu.model.hf_config.norm_pix_loss=$MODEL_NORM_PIX_LOSS" \
exp_mfu.optim.lr=0.0003 \
exp_mfu.optim.fused=false \
exp_mfu.misc.monitors_dynamics=false \
exp_mfu.misc.compiles_model=false \
exp_mfu.misc.max_eval_iter=10 \
exp_mfu.misc.sharding_stage=$SHARDING_STAGE \
exp_mfu.misc.peak_flops_per_sec=312000000000000 \
exp_mfu.lr_scheduler.warmup_iterations=10 \
exp_mfu.lr_scheduler.total_iterations=1000000 \
exp_mfu.logging.prefix=$JOB \
exp_mfu.dist.dtype=float16

## base_command="mpirun -n $NUM_MPI_TASKS python train.fsdp.dummy_dataset.py experiments/yaml/$JOB.yaml"
## final_command="OMP_NUM_THREADS=1 "
## 
## if [ $RUNS_NSYS -eq 1 ]; then
##     final_command+="nsys profile -w true -t cuda,mpi --mpi-impl=openmpi --wait primary -o /sdf/data/lcls/ds/prj/prjcwang31/results/proj-peaknet/nsight_reports/$JOB.profile -f true --cudabacktrace=true -x true "
## fi
## final_command+="$base_command"
## 
## eval $final_command
