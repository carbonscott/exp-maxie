#!/bin/bash

python launch_job.py \
    job=mfxx49820_test \
    auto_submit=true \
    skip_preempt=true \
    train_config.dataset.path_train=experiments/datasets/mfxx49820_r0024.smallset.train.json \
    train_config.dataset.path_eval=experiments/datasets/mfxx49820_r0024.smallset.eval.json \
    train_config.dataset.batch_size=1 \
    train_config.dataset.num_workers=2 \
    train_config.dataset.seg_size=4 \
    train_config.dataset.entry_per_cycle=1 \
    train_config.dataset.debug=true \
    train_config.loss.grad_accum_steps=2 \
    train_config.model.from_scratch=true \
    train_config.misc.max_eval_iter=2 \
    train_config.misc.num_gpus=6 \
    train_config.misc.data_dump_on=false \
    train_config.misc.max_epochs=100000 \
    train_config.misc.monitors_dynamics=true \
    train_config.lr_scheduler.warmup_iterations=1 \
    train_config.lr_scheduler.total_iterations=11370 \
    train_config.lr_scheduler.scheduler_update_iterations=8 \
    train_config.logging.prefix=mfxx49820_test \
    train_config.checkpoint.prefix=mfxx49820_r0024.smallset \
    train_config.checkpoint.chkpt_saving_iterations=null \
    train_config.checkpoint.path_chkpt_prev=null \
    train_config.checkpoint.preempt_chkpt_saving_iterations=null \
    bsub_config.ipc_workers=2 \
    bsub_config.qos=debug \
    bsub_config.walltime=2:00 \
    bsub_config.num_nodes=2 \
    bsub_config.trainer=train.fsdp.py \
    bsub_config.num_cpus_for_client=4
