#!/usr/bin/env python
# -*- coding: utf-8 -*-

# =============
# BASIC IMPORTS
# =============
import os
import yaml
import tqdm
import argparse
import inspect
import logging
import traceback
import time

from contextlib import nullcontext
from omegaconf  import OmegaConf
from packaging import version

# ===========
# HUGGINGFACE
# ===========
from transformers.models.vit_mae.configuration_vit_mae import ViTMAEConfig
from transformers.models.vit_mae.modeling_vit_mae import (
    ViTMAELayer,
    ViTMAEForPreTraining,
    ViTMAEPreTrainedModel,  # Monkey patch the _init_weights
    ViTMAEModel,  # Monkey patch the _init_weights
)
# Imports for monitoring training dynamics
from transformers.activations import ACT2CLS

# =====
# TORCH
# =====
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

# Fully Sharded Data Parallel (FSDP)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

# =====
# MAXIE
# =====
from maxie.datasets.dummy_dataset import (
    DistributedSegmentedDummyImageDataConfig,
    DistributedSegmentedDummyImageData,
)
from maxie.utils.seed import set_seed
from maxie.utils.misc import is_action_due
from maxie.utils.flops import estimate_transformer_flops
from maxie.utils.signal import register_handlers
from maxie.utils.dist import dist_setup
from maxie.utils.checkpoint import init_checkpointer
from maxie.utils.logger import init_logger
from maxie.utils.model import (
    _init_weights_in_encoder,
    _init_weights_in_decoder,
    logging_model_init,
    unfreeze_model,
)
from maxie.utils.eval import estimate_loss
from maxie.utils.data import wrap_with_torch_dataloader
from maxie.lr_scheduler import CosineLRScheduler
from maxie.tensor_transforms import (
    Pad,
    NoTransform,
)
from maxie.utils.fsdp import (
    MemoryMaximizer,
    set_sharding_strategy,
    fsdp_wrapped_layers,
    backward_prefetch,
    act_chkpt,
)
from maxie.utils.monitor import (
    ActivationMonitor,
    monitor_param_update_metrics,
)
from maxie.patches.build_metadata import patch_build_metadata

# =====
# Debug
# =====
# [WARNING] Making it True may throw errors when using float16.
# Invalid gradients are expected to occur during mixed-precision training in
# float16 and anomaly detection will thus report false errors.
# Refer to https://discuss.pytorch.org/t/convolutionbackward0-returned-nan-values-in-its-0th-output/175571/4
torch.autograd.set_detect_anomaly(False)

# Get the logger
logger = logging.getLogger(__name__)
register_handlers()

# ======================
# COMMAND LINE INTERFACE
# ======================
parser = argparse.ArgumentParser(description = "Load training configuration from a YAML file to a dictionary.")
parser.add_argument("yaml_file", help="Path to the YAML file")
args = parser.parse_args()

# =============
# CONFIGURATION
# =============
# Load CONFIG from YAML
fl_yaml = args.yaml_file
with open(fl_yaml, 'r') as fh:
    config = OmegaConf.create(yaml.safe_load(fh))

# ====
# DIST
# ====
dist_config = dist_setup(
    config.dist.cpu_only,
    config.dist.backend,
)
uses_dist = dist_config.uses_dist
dist_rank = dist_config.rank
dist_local_rank = dist_config.local_rank
dist_world_size = dist_config.world_size
device = dist_config.device

# Set up performance utility
memmax = MemoryMaximizer() if dist_local_rank == 0 else None

# Monkey patch one method (Not always required)
torch_version = torch.__version__
torch_version = torch_version[:torch_version.find("+") if "+" in torch_version else None]
if version.parse(torch_version) <= version.parse("2.0.1"):
    if config.checkpoint.state_dict_type == "sharded":
        patch_build_metadata()

# ==========
# FSDP SETUP
# ==========
sharding_strategy = set_sharding_strategy(config.dist.sharding_stage)
auto_wrap_policy = fsdp_wrapped_layers({ViTMAELayer})
backward_prefetch = backward_prefetch()

# ======
# LOGGER
# ======
timestamp = init_logger(
    uses_dist,
    dist_rank,
    device,
    config.logging.prefix,
    config.logging.directory,
    config.logging.level,
    'console',
)

# =======
# SEEDING
# =======
base_seed  = 0
seed_offset = dist_rank if config.dist.uses_unique_world_seed else 0
world_seed = base_seed + seed_offset
set_seed(world_seed)

# =======
# DATASET
# =======
H_pad = config.dataset.transforms.H_pad
W_pad = config.dataset.transforms.W_pad
pre_transforms = (
    Pad(H_pad, W_pad) if config.dataset.transforms.set.pad else NoTransform(),
)
C = config.dataset.input.C
H = config.dataset.input.H
W = config.dataset.input.W
seg_size = config.dataset.seg_size
total_size = config.dataset.input.total_size
dataset_train_config = DistributedSegmentedDummyImageDataConfig(
    C, H, W, seg_size, total_size, dist_rank, dist_world_size, pre_transforms, None,
)
dataset_train = DistributedSegmentedDummyImageData(dataset_train_config)
dataset_eval_train = DistributedSegmentedDummyImageData(dataset_train_config)
dataset_eval_val_config = DistributedSegmentedDummyImageDataConfig(
    C, H, W, seg_size, total_size, dist_rank, dist_world_size, pre_transforms, None,
)
dataset_eval_val = DistributedSegmentedDummyImageData(dataset_eval_val_config)
custom_collate = None
transforms = None

# ===================
# CHECKPOINT PRE FSDP
# ===================
checkpointer = init_checkpointer(
    config.checkpoint.state_dict_type,
    uses_dist,
)
from_resume = config.checkpoint.path_chkpt_prev is not None

# =====
# MODEL
# =====
logger.debug(f'[RANK {dist_rank}] Configuring model...')
# Define model
ViTMAEModel._init_weights = _init_weights_in_encoder
ViTMAEPreTrainedModel._init_weights = _init_weights_in_decoder
hf_model_config = config.model.hf_config
model_config = ViTMAEConfig(**hf_model_config)
model = ViTMAEForPreTraining(model_config)
if not uses_dist: model.to(device)
logging_model_init(dist_config, model)
unfreeze_model(model)
if dist_rank == 0:
    logger.debug(f"{sum(p.numel() for p in model.parameters())/1e6} M pamameters.")

# Model resumption
if from_resume:
    checkpointer.pre_fsdp_load(dist_rank, model, config.checkpoint.path_chkpt_prev)

# Mixed precision
mixed_precision_dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config.dist.dtype]
mixed_precision = MixedPrecision(
    param_dtype  = mixed_precision_dtype,
    reduce_dtype = mixed_precision_dtype,
    buffer_dtype = mixed_precision_dtype,
)

# Autocast
device_type = 'cuda' if 'cuda' in device else 'cpu'
autocast_context = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type = device_type, dtype = mixed_precision_dtype)

# GradScaler
# If enabled = False scaler is a no-op
scaler_func = ShardedGradScaler if uses_dist else torch.cuda.amp.GradScaler
scaler = scaler_func(enabled=(config.dist.dtype == 'float16'))

# Compile the model
if config.misc.compiles_model:
    logger.debug("Compiling the model...")
    model = torch.compile(model) # requires PyTorch 2.0

# Wrapping the model in FSDP
if uses_dist:
    # Convert BatchNorm to SyncBatchNorm
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Wrap it up using FSDP
    model = FSDP(
        model,
        auto_wrap_policy  = auto_wrap_policy,
        mixed_precision   = mixed_precision,
        backward_prefetch = backward_prefetch,
        forward_prefetch  = True,
        sharding_strategy = sharding_strategy,
        limit_all_gathers = True,
        use_orig_params   = False,
        device_id         = device,
    )

    sharded_param_count = sum(p.numel() for p in model.parameters())  # .module will return the raw model view when use_orig_params = True
                                                                      # making it effectively reporting the sharded param count.  Removing
                                                                      # .module makes it more consistent regardles of use_orig_params.
    logger.debug(f"RANK {dist_rank} - sharded parameter count: {sharded_param_count*1e-6} M.")

    dist.barrier()

# -- Optional grad sync off (to allow grad accumulation)
grad_sync_context = lambda enables_sync: nullcontext() if enables_sync or not uses_dist else model.no_sync()

# Activation checkpointing
act_chkpt(model, ViTMAELayer)

# ================
# CRITERION (LOSS)
# ================
logger.debug(f'[RANK {dist_rank}] Configuring criterion (Skip, it is configured in the model)...')

# =======================
# OPTIMIZER AND SCHEDULER
# =======================
logger.debug(f'[RANK {dist_rank}] Configuring optimizer...')
param_iter = model.parameters()
optim_arg_dict = dict(
    lr           = config.optim.lr,
    weight_decay = config.optim.weight_decay,
    betas        = (config.optim.beta1, config.optim.beta2),
)
if 'fused' in inspect.signature(optim.AdamW).parameters:
    optim_arg_dict['fused'] = config.optim.fused
optimizer = optim.AdamW(param_iter, **optim_arg_dict)
scheduler = CosineLRScheduler(optimizer         = optimizer,
                              warmup_iterations = config.lr_scheduler.warmup_iterations,
                              total_iterations  = config.lr_scheduler.total_iterations,
                              min_lr            = config.lr_scheduler.min_lr)

# ====================
# CHECKPOINT POST FSDP
# ====================
logger.debug(f'[RANK {dist_rank}] Confguring model, optim, scheduler, training state checkpoint...')
# Set init training state dict
loss_min = float('inf')
iter_state = dict(
    epoch     = 0,
    seg       = 0,
    start_idx = dataset_train.start_idx,
    end_idx   = dataset_train.end_idx,
    loss_min  = loss_min,
)

# Optional resumption
last_epoch = 0
last_seg   = -1
if from_resume:
    # Optimizer, scheduler are loaded
    checkpointer.post_fsdp_load(dist_rank, model, optimizer, scheduler, iter_state, config.checkpoint.path_chkpt_prev)

    # Training state
    last_epoch = iter_state.get("epoch")
    last_seg   = iter_state.get("seg")
    loss_min   = iter_state.get("loss_min")

    logger.info(f"Loading from checkpoint -- {config.checkpoint.path_chkpt_prev}.")
    logger.info(f"PREV - last_epoch {last_epoch}, last_seg {iter_state.get('start_idx')}-{iter_state.get('end_idx')}, loss_min = {loss_min}")

# ============================
# Monitoring training dynamics
# ============================
monitors_dynamics = config.misc.monitors_dynamics
if monitors_dynamics:
    modules_to_monitor = (ACT2CLS[model.config.hidden_act], )
    act_monitor = ActivationMonitor(model, modules_to_monitor)
    act_monitor.add_hooks()

def is_last_batch(batch_idx, num_batches):
    return batch_idx + 1 == num_batches

# =============
# TRAINING LOOP
# =============
batch_input_shape = None
state_dict_type = config.checkpoint.state_dict_type
logger.debug(f'[RANK {dist_rank}] Ready for training loop...')
iteration_counter = 0  # One iteration is one param update after one or a few forward/backward pass
try:
    # ================
    # Loop over epochs
    # ================
    # Only increment starting epoch if current epoch was fully completed
    for epoch in tqdm.tqdm(range(config.misc.max_epochs), desc = f'[RANK {dist_rank}] Epoch'):
        # Skip epochs up to, but not including the last_epoch
        if epoch < last_epoch: continue

        # Reset dataset in a new epoch???
        if not from_resume:
            dataset_train.reset()
        # Otherwise, update the dataset index according to the training state
        else:
            # Update the dataset status
            dataset_train.start_idx = iter_state.get("start_idx")
            dataset_train.end_idx   = iter_state.get("end_idx")

        # ==========================
        # Loop over dataset segments
        # ==========================
        for seg in tqdm.tqdm(range(dataset_train.num_seg), desc = f'[RANK {dist_rank}] Segment'):
            # Switch to training state
            model.train()

            # ========================
            # Prepare for next segment
            # ========================
            # Skip previous segments up to and including the last_seg
            if epoch == last_epoch and seg <= last_seg:
                continue

            # Prepare training on one segment (iteration)
            # Set next segment or break the loop when having no next segment
            requires_reset = dataset_train.set_start_idx(dataset_train.end_idx)
            if requires_reset:
                break
            if dist_rank == 0:
                logger.info(f"Working on segment: {dataset_train.start_idx}:{dataset_train.end_idx}; Total size: {dataset_train.total_size}")
            dataloader, sampler = wrap_with_torch_dataloader(
                dataset_train,
                base_seed,
                config.dataset.drop_last_in_sampler,
                config.dataset.drop_last_in_loader,
                uses_dist,
                config.dataset.batch_size,
                config.dataset.num_workers,
                custom_collate,
                config.dataset.pin_memory,
                config.dataset.prefetch_factor,
                epoch,
                is_eval=True,
            )
            # Shuffle the training example
            if uses_dist:
                sampler.set_epoch(epoch)

            # ====================================
            # Go through Mini batches in a segment
            # ====================================
            logger.debug(f"[RANK {dist_rank}] Start processing {len(dataloader)} batches at epoch {epoch}, seg {seg}.")
            # Start memmax
            if dist_local_rank == 0:
                memmax.start()

            # Set up helper variables for gradient accum and reporting
            # Set up gradient accumulation helper variables
            grad_nosync_counter         = 0
            num_batches                 = len(dataloader)
            num_remainder_batches       = num_batches % config.loss.grad_accum_steps
            start_idx_remainder_batches = num_batches - num_remainder_batches  # e.g. total=102, steps=5, idx = 102 - 102%5 = 100

            # Aggregate the loss and number of processed tokens and batches during each gradient accumulation
            total_loss       = torch.tensor(0.0, device = device)
            total_num_tokens = torch.tensor(0.0, device = device)
            total_num_batch  = torch.tensor(0.0, device = device)

            # Set a timer flag
            starts_timer = True
            for batch_idx, batch_data in tqdm.tqdm(
                enumerate(dataloader),
                total = num_batches,
                desc  = f'[RANK {dist_rank}] Mini batch',
            ):
                # Start timer???
                if starts_timer:
                    t_start = time.monotonic()
                    starts_timer = False

                # =====================
                # Forward/Backward pass
                # =====================
                # Prepare batch input
                batch_input = batch_data  # (B, C, H, W)
                batch_input = batch_input.to(device, non_blocking = True, dtype = mixed_precision_dtype)
                if transforms is not None:
                    for enum_idx, trans in enumerate(transforms):
                        batch_input = trans(batch_input)

                # Specify the effective grad accum steps
                real_grad_accum_steps = config.loss.grad_accum_steps if batch_idx < start_idx_remainder_batches else num_remainder_batches

                # Conditionally turn off grad sync for grad accumulation to simulate a larger batch unless the sync is due or the last batch
                # Refer to https://github.com/pytorch/pytorch/blob/6c4f43f82675b5fcfe8cf3e5983d0c0f326408aa/test/distributed/fsdp/test_fsdp_grad_acc.py#L180
                is_grad_sync_required = is_last_batch(batch_idx, len(dataloader)) or is_action_due(grad_nosync_counter, config.loss.grad_accum_steps)
                with grad_sync_context(is_grad_sync_required):
                    # Forward
                    with autocast_context:
                        batch_output = model(batch_input)
                        loss = batch_output.loss  # Refer to https://github.com/huggingface/transformers/blob/e34da3ee3c9d2d628fdbeb60cee45c4f8f32945a/src/transformers/models/vit_mae/modeling_vit_mae.py#L1001
                        loss = loss / real_grad_accum_steps  # scale the loss to account for gradient accumulation

                    # Accumulate loss
                    total_loss += loss.detach()

                    # Accumulate number of tokens processed
                    total_numel = batch_data.numel()  # Get number of numeric elements
                    token_size  = model.config.patch_size**2
                    num_tokens  = total_numel / token_size
                    total_num_tokens += num_tokens

                    # Accumulate number of batches
                    num_batch = batch_data.size(0)
                    total_num_batch += num_batch

                    # Backward
                    scaler.scale(loss).backward()

                # Increment the grad nosync counter
                grad_nosync_counter += 1

                # ========================================================
                # Conditional parameter updates when grad sync is required
                # ========================================================
                if is_grad_sync_required:
                    # Grad clipping
                    if config.optim.grad_clip != 0.0:
                        scaler.unscale_(optimizer)
                        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config.optim.grad_clip) \
                                    if (not uses_dist) or sharding_strategy == ShardingStrategy.NO_SHARD \
                                    else \
                                    model.clip_grad_norm_(config.optim.grad_clip)

                    # Update parameters
                    scaler.step(optimizer)
                    scaler.update()

                    # Increment the iteration counter after param update
                    iteration_counter += 1

                    # Obtain the mean total loss
                    if uses_dist:
                        dist.all_reduce(total_loss, op = dist.ReduceOp.AVG)  # Avg across ranks

                    # ==========
                    # Report MFU
                    # ==========
                    # Obtain the total number of tokens processed
                    if uses_dist:
                        dist.all_reduce(total_num_tokens, op = dist.ReduceOp.SUM)  # Sum across ranks
                        dist.all_reduce(total_num_batch , op = dist.ReduceOp.SUM)  # Sum across ranks

                    # Wait for all gpus to complete work
                    if device_type == "cuda":
                        torch.cuda.synchronize()

                    # Stop timer
                    t_end = time.monotonic()

                    # Calculate tokens per second
                    t_delta = t_end - t_start
                    tokens_per_sec = total_num_tokens / t_delta

                    # Log the training loop loss after a forward/backward/update
                    if dist_rank == 0:
                        # MFU...
                        # ...Encoder
                        model_hidden_size = model_config.hidden_size
                        num_heads         = model_config.num_attention_heads
                        num_layers        = model_config.num_hidden_layers
                        image_size        = model_config.image_size
                        patch_size        = model_config.patch_size
                        context_length    = (image_size/patch_size)**2
                        mask_ratio        = model_config.mask_ratio
                        encoder_flops     = estimate_transformer_flops(model_hidden_size, num_heads, num_layers, context_length*(1-mask_ratio))

                        # ...Decoder
                        model_hidden_size = model_config.decoder_hidden_size
                        num_heads         = model_config.decoder_num_attention_heads
                        num_layers        = model_config.decoder_num_hidden_layers
                        decoder_flops     = estimate_transformer_flops(model_hidden_size, num_heads, num_layers, context_length)

                        model_flops_per_sec = (encoder_flops+decoder_flops) * total_num_batch / t_delta
                        mfu = model_flops_per_sec / config.misc.peak_flops_per_sec

                        # Misc
                        current_lrs   = scheduler.get_lr()
                        seg_start_idx = dataset_train.start_idx
                        seg_end_idx   = dataset_train.end_idx

                        # Log
                        log_data = {
                            "rank"               : dist_rank,
                            "logevent"           : "LOSS:TRAIN",
                            "iteration"          : iteration_counter,
                            "segment"            : f"{seg_start_idx}-{seg_end_idx}",
                            "learning_rate"      : ",".join(f"{lr}" for lr in current_lrs),
                            "grad_norm"          : f"{grad_norm:.6f}",
                            "mean_train_loss"    : f"{total_loss:.6f}",
                            "tokens_per_sec"     : f"{tokens_per_sec:.1e}",
                            "mfu"                : f"{mfu:.3f}",
                            "grad_nosync_counter": grad_nosync_counter,
                        }
                        log_msg = " | ".join([f"{k}={v}" for k, v in log_data.items()])
                        logger.info(log_msg)

                    # =========================
                    # Monitor training dynamics
                    # =========================
                    # Do it before zero-ing gradients
                    if monitors_dynamics:
                        # Monitor preactivation and activation of the nonlinearity
                        for name, act in act_monitor.activations.items():
                            mean_preact, std_preact = act.get('pre')
                            mean_act, std_act       = act.get('pos')
                            log_data = {
                                "rank"        : dist_rank,
                                "iteration"   : iteration_counter,
                                "logevent"    : "DYNAMICS:ACT",
                                "name"        : name,
                                "preact.mean" : mean_preact,
                                "preact.std"  : std_preact,
                                "act.mean"    : mean_act,
                                "act.std"     : std_act,
                            }
                            log_msg = " | ".join([f"{k}={v}" for k, v in log_data.items()])
                            logger.info(log_msg)

                        # Monitor param update
                        current_lr = scheduler.get_lr()[0]  # It's a list
                        encoder_param_monitor = monitor_param_update_metrics(model.vit.encoder, current_lr)  # Motifs like transformer blocks
                        decoder_param_monitor = monitor_param_update_metrics(model.decoder.decoder_layers, current_lr)

                        for k, v in encoder_param_monitor.get('percent_param_update').items():
                            log_data = {
                                "rank"      : dist_rank,
                                "iteration" : iteration_counter,
                                "logevent"  : "DYNAMICS:PARAMS",
                                "type"      : "encoder",
                                "name"      : k,
                                "update"    : v,
                            }
                            log_msg = " | ".join([f"{k}={v}" for k, v in log_data.items()])
                            logger.info(log_msg)

                        for k, v in decoder_param_monitor.get('percent_param_update').items():
                            log_data = {
                                "rank"      : dist_rank,
                                "iteration" : iteration_counter,
                                "logevent"  : "DYNAMICS:PARAMS",
                                "type"      : "decoder",
                                "name"      : k,
                                "update"    : v,
                            }
                            log_msg = " | ".join([f"{k}={v}" for k, v in log_data.items()])
                            logger.info(log_msg)

                    # ============================
                    # Reset for the next iteration
                    # ============================
                    # Flush the gradients
                    optimizer.zero_grad(set_to_none = True)

                    # Reset grad accum counter
                    grad_nosync_counter = 0

                    # Reset the loss accumulator
                    total_loss *= 0.0

                    # Reset the token and batch accumulator
                    total_num_tokens *= 0
                    total_num_batch  *= 0

                    # Reset timer flag
                    starts_timer = True

                    # Update lr every few seg (X segs = one step/iteration)
                    if is_action_due(iteration_counter, config.lr_scheduler.scheduler_update_iterations):
                        scheduler.step()
                        if dist_rank == 0:
                            current_lrs = scheduler.get_lr()
                            current_lrs_msg = ",".join(f"{lr}" for lr in current_lrs)
                            logger.info(f"lr is updated to {current_lrs_msg}.")

                    # ======================
                    # Eval and checkpointing
                    # ======================
                    if is_action_due(iteration_counter, config.checkpoint.chkpt_saving_iterations):
                        # !!!!!!!!!!!!!!!
                        # !! Data dump !!
                        # !!!!!!!!!!!!!!!
                        data_dump_timestamp = {
                            "uses_dist"       : uses_dist,
                            "dist_rank"       : dist_rank,
                            "dist_world_size" : dist_world_size,
                        }
                        if config.misc.data_dump_on:
                            data_dump_timestamp.update({
                                "fl_log_prefix"   : config.logging.prefix,
                                "epoch"           : epoch,
                                "seg"             : seg,
                            })

                        if dist_rank == 0:
                            logger.debug(f'[RANK {dist_rank}] Start evaluation...')

                        # ====
                        # Eval
                        # ====
                        # 1. Training loss
                        # Get a random subset of the training set
                        train_loss = torch.tensor(float('nan'))
                        num_eval_retry = 0
                        while torch.isnan(train_loss) and (num_eval_retry < config.misc.max_eval_retry):
                            dataset_eval_train.reset()
                            high_seg_idx = max(dataset_eval_train.total_size - seg_size * dist_world_size, 1)
                            rand_start_idx = torch.randint(low = 0, high = high_seg_idx, size = (1,)).item()
                            dataset_eval_train.set_start_idx(rand_start_idx)
                            dataloader_eval, sampler_eval = wrap_with_torch_dataloader(
                                dataset_eval_train,
                                base_seed,
                                config.dataset.drop_last_in_sampler,
                                config.dataset.drop_last_in_loader,
                                dist_config.uses_dist,
                                config.dataset.batch_size,
                                config.dataset.num_workers,
                                custom_collate,
                                config.dataset.pin_memory,
                                config.dataset.prefetch_factor,
                                epoch,
                                is_eval=True,
                            )
                            if dist_config.uses_dist:
                                sampler_eval.set_epoch(rand_start_idx)  # Any integer is fine
                            train_loss = estimate_loss(
                                dataloader_eval,
                                model,
                                autocast_context,
                                max_iter              = config.misc.max_eval_iter,
                                desc                  = '(training set)',
                                device                = device,
                                dummy_input_shape     = batch_input_shape,
                                mixed_precision_dtype = mixed_precision_dtype,
                                transforms            = transforms,
                                **data_dump_timestamp,
                            )
                            num_eval_retry += 1

                        # Log the train loss
                        if dist_rank == 0:
                            seg_start_idx = dataset_eval_train.start_idx
                            seg_end_idx   = dataset_eval_train.end_idx
                            logger.info(f"[RANK {dist_rank}] LOSS:EVAL - epoch {epoch}, seg {seg_start_idx}-{seg_end_idx}, mean train loss = {train_loss:.8f}")

                        # 2. Validation loss
                        # Get a random subset of the validation set
                        validate_loss = torch.tensor(float('nan'))
                        num_eval_retry = 0
                        while torch.isnan(validate_loss) and (num_eval_retry < config.misc.max_eval_retry):
                            dataset_eval_val.reset()
                            high_seg_idx = max(dataset_eval_val.total_size - seg_size * dist_world_size, 1)
                            rand_start_idx = torch.randint(low = 0, high = high_seg_idx, size = (1,)).item()
                            dataset_eval_val.set_start_idx(rand_start_idx)
                            dataloader_eval_val, sampler_eval = wrap_with_torch_dataloader(
                                dataset_eval_val,
                                base_seed,
                                config.dataset.drop_last_in_sampler,
                                config.dataset.drop_last_in_loader,
                                dist_config.uses_dist,
                                config.dataset.batch_size,
                                config.dataset.num_workers,
                                custom_collate,
                                config.dataset.pin_memory,
                                config.dataset.prefetch_factor,
                                epoch,
                                is_eval=True,
                            )
                            if uses_dist:
                                sampler_eval.set_epoch(rand_start_idx)  # Any integer is fine
                            validate_loss = estimate_loss(
                                dataloader_eval,
                                model,
                                autocast_context,
                                max_iter              = config.misc.max_eval_iter,
                                desc                  = '(validation set)',
                                device                = device,
                                dummy_input_shape     = batch_input_shape,
                                mixed_precision_dtype = mixed_precision_dtype,
                                transforms            = transforms,
                                **data_dump_timestamp,
                            )
                            num_eval_retry += 1

                        # Log the validation loss
                        if dist_rank == 0:
                            seg_start_idx = dataset_eval_val.start_idx
                            seg_end_idx   = dataset_eval_val.end_idx
                            logger.info(f"[RANK {dist_rank}] LOSS:EVAL - epoch {epoch}, seg {seg_start_idx}-{seg_end_idx}, mean validation loss = {validate_loss:.8f}")

                        # =============
                        # Checkpointing
                        # =============
                        if validate_loss < loss_min:
                            loss_min = validate_loss

                            # Collect training state
                            iter_state["epoch"]     = epoch
                            iter_state["seg"]       = seg
                            iter_state["start_idx"] = dataset_train.start_idx
                            iter_state["end_idx"]   = dataset_train.end_idx
                            iter_state["loss_min"]  = loss_min

                            dir_chkpt = f"{timestamp}.epoch_{epoch}.end_idx_{dataset_train.end_idx}"
                            if config.checkpoint.prefix is not None: dir_chkpt = f"{config.checkpoint.prefix}.{dir_chkpt}"
                            path_chkpt = os.path.join(config.checkpoint.directory, dir_chkpt)
                            checkpointer.save(dist_rank, model, optimizer, scheduler, iter_state, path_chkpt)
                            logger.info(f"Saving checkpoint at {path_chkpt}.")

                        # All ranks wait until the end of evaluation by rank 0
                        # [WARNING] Expecting NCCL TIMEOUT ERROR if the evaluation takes too long
                        if uses_dist:
                            dist.barrier()
                        logger.debug(f'[RANK {dist_rank}] Done evaluation...')

                    # ========================
                    # Preemptive checkpointing
                    # ========================
                    if config.checkpoint.preempt_metadata_path is not None and is_action_due(iteration_counter, config.checkpoint.preempt_chkpt_saving_iterations):
                        # Collect training state
                        iter_state["epoch"]     = epoch
                        iter_state["seg"]       = seg
                        iter_state["start_idx"] = dataset_train.start_idx
                        iter_state["end_idx"]   = dataset_train.end_idx
                        iter_state["loss_min"]  = loss_min

                        dir_chkpt = f"{timestamp}.preempt"
                        if config.checkpoint.prefix is not None: dir_chkpt = f"{config.checkpoint.prefix}.{dir_chkpt}"
                        path_chkpt = os.path.join(config.checkpoint.directory, dir_chkpt)
                        checkpointer.save(dist_rank, model, optimizer, scheduler, iter_state, path_chkpt)
                        logger.info(f"[RANK {dist_rank}] Saving preemptive checkpoint (epoch {epoch}, end_idx {dataset_train.end_idx}) at {path_chkpt}.")

                        if dist_rank == 0:
                            with open(config.checkpoint.preempt_metadata_path, "w") as f:
                                f.write(path_chkpt)
                            logger.info(f"[RANK {dist_rank}] Saving preemptive metadata (epoch {epoch}, end_idx {dataset_train.end_idx}) at {config.checkpoint.preempt_metadata_path}.")

            # End performance
            if dist_local_rank == 0:
                memmax.update()
            if dist_local_rank == 0:
                memmax.stop()

        # Reset flags
        last_seg = -1
        from_resume = False

except KeyboardInterrupt:
    logger.error(f"[RANK {dist_rank}] Training was interrupted!")
except Exception as e:
    tb = traceback.format_exc()
    logger.error(f"[RANK {dist_rank}] Error occurred: {e}\nTraceback: {tb}")
finally:
    # Clean up hooks
    if monitors_dynamics:
        act_monitor.remove_hooks()

    # Ensure that the process group is always destroyed
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
