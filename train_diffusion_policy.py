import copy
import logging
import os
import shutil
import sys
from argparse import ArgumentParser
from datetime import datetime

import diffusers
import numpy as np
import plotly.express as px
import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.optimization import (
    get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup)
from diffusers.schedulers import DDIMScheduler
from omegaconf import OmegaConf
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModel

import wandb
from model.policy import DiffusionPolicyUNet
from model.wrapper import ResNetEncoderWrapper
from train_cage import print_model_info
from utils.constants import ACTION_DIM
from utils.dataset import RealWorldDataset
from utils.preprocessings import get_actual_actions
from utils.transforms import rot6d_to_rotation_matrix

logger = get_logger(__name__)

def get_parser():
    parser = ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        default='diffusion_policy',
        help='config file used for training'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='seed for deterministic training'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='resume from a checkpoint',
    )
    parser.add_argument(
        '--weights_only',
        default=False,
        action='store_true',
        help='whether to load training states as well when resuming'
    )
    parser.add_argument(
        '--new_run',
        default=False,
        action='store_true',
        help='whether to start a new run on wandb when resuming'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs',
        help="output directory for logs and checkpoints",
    )
    parser.add_argument(
        '--checkpoint_total_limit',
        type=int,
        default=2,
        help="maximum number of checkpoint saved",
    )

    parser.add_argument(
        '--mixed_precision',
        default=False,
        action='store_true',
        help='whether or not to use mixed precision for training'
    )
    
    parser.add_argument(
        '--test',
        default=False,
        action='store_true',
        help='whether this is a test session (not generating logs)'
    )
    
    return parser

def initialize_model(conf):
    # Image encoder
    model_path = os.path.join('weights', conf.model.image_encoder.name)
    # Load weights from pretrained model
    # resnet model
    if conf.model.image_encoder.pretrain:
        i_encoder = AutoModel.from_pretrained(model_path)
    else:
        i_encoder = AutoModel.from_config(AutoConfig.from_pretrained(model_path))
    i_encoder = ResNetEncoderWrapper(i_encoder, pooled=True, out_dim=conf.model.obs_dim)

    if conf.model.use_in_hand and conf.model.use_fixed:
        i_encoder_2 = copy.deepcopy(i_encoder)
    
        obs_encoders = nn.ModuleDict({
            'fixed_obs': i_encoder,
            'in_hand_obs': i_encoder_2,
        })
    elif conf.dataset.meta_data.use_in_hand:
        obs_encoders = nn.ModuleDict({
            'in_hand_obs': i_encoder,
        })
    else:
        obs_encoders = nn.ModuleDict({
            'fixed_obs': i_encoder,
        })

    # Policy model
    if conf.model.name == 'unet':
        model = DiffusionPolicyUNet(
            obs_encoders,
            act_dim     = ACTION_DIM,
            obs_num     = conf.dataset.meta_data.fixed_views + conf.dataset.meta_data.in_hand_views,
            obs_horizon = conf.dataset.obs_horizon,
            obs_dim     = conf.model.obs_dim,
        )
    elif conf.model.name == 'transformer':
        # TODO: transformer backbone not implemented yet
        raise NotImplementedError
    else:
        raise NotImplementedError(f"Diffusion model type: {conf.model.name} not implemented!")
    
    return model

def calc_loss(batch, model, noise_scheduler, use_proprio=True, weight_dtype=torch.float32):
    # fixed_obs: B, Nf, To, C, H, W
    # in_hand_obs: B, Ni, To, C, H, W
    obs_dict = {k: v.to(dtype=weight_dtype, non_blocking=True) for k, v in batch.items() if '_obs' in k}
    actions = batch['actions'].float()  # B, Ta, D=10 (rot6d)

    bs = actions.shape[0]
    # Sample gaussian noises
    noise = torch.randn_like(actions)
    # Sample a random timestep for each sample
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=actions.device, dtype=torch.long)
    noisy_actions = noise_scheduler.add_noise(actions, noise, timesteps)

    offset = 0
    if use_proprio:
        proprio = batch['proprio'].float()  # B, To, D=10
        offset = proprio.shape[1]
        # Use prev_actions as proprioception
        noisy_actions = torch.cat([proprio, noisy_actions], dim=1)
    noisy_actions = noisy_actions.to(dtype=weight_dtype)
    
    # Predict the noise residual
    pred = model(noisy_actions, timesteps, obs_dict=obs_dict)[:, offset:, :]

    # Get the target for loss depending on the prediction type
    if noise_scheduler.config.prediction_type == "epsilon":
        target = noise
    elif noise_scheduler.config.prediction_type == "v_prediction":
        target = noise_scheduler.get_velocity(actions, noise, timesteps)
    else:
        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

    # Compute loss and mean over the non-batch dimensions
    loss = F.mse_loss(pred.float(), target.float(), reduction="none").mean([1, 2])
    return loss

def sample_actions(batch, generator, model, noise_scheduler, use_proprio=True, weight_dtype=torch.float32, num_inference_steps=50):
    obs_dict = {k: v.to(dtype=weight_dtype, non_blocking=True) for k, v in batch.items() if '_obs' in k}
    real_actions = batch['actions']
    proprio = batch['proprio'].to(dtype=weight_dtype, non_blocking=True)
    To = proprio.shape[1]

    # Timesteps
    noise_scheduler.set_timesteps(num_inference_steps, device=real_actions.device)

    # Initial noise
    pred_actions = torch.randn(real_actions.shape, generator=generator, device=real_actions.device, dtype=weight_dtype)
    pred_actions = pred_actions * noise_scheduler.init_noise_sigma

    obs_emb = model.preprocess_obs(obs_dict)
    for t in noise_scheduler.timesteps:
        if use_proprio:
            pred_noise = model(torch.cat([proprio, pred_actions], dim=1), t, obs_emb=obs_emb)[:, To:, :]
        else:
            pred_noise = model(pred_actions, t, obs_emb=obs_emb)
        # compute the previous noisy sample x_t -> x_t-1
        pred_actions = noise_scheduler.step(pred_noise, t, pred_actions).prev_sample

    return pred_actions.float()

def main():
    sys.path.append(os.getcwd())

    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    if opt.resume is not None and not opt.new_run:
        print('Resuming from a previous run, loading its config file as default...')
        project_path = os.path.join(opt.output_dir, os.path.dirname(opt.resume))
        base_conf = OmegaConf.load(os.path.join(project_path, 'config.yaml'))
    else:
        project_path = os.path.join(opt.output_dir, opt.config + f'-{datetime.now()}'.split('.')[0][:-3].replace(':', '-'))
        base_conf = OmegaConf.load(os.path.join('configs', opt.config+'.yaml'))

    cli = OmegaConf.from_dotlist(unknown)
    conf = OmegaConf.merge(base_conf, cli)
    
    proj_conf = ProjectConfiguration(project_dir=project_path, logging_dir=os.path.join(project_path, 'logs'))
    deepspeed_plugin = DeepSpeedPlugin(
        zero_stage=2,
        gradient_accumulation_steps=conf.gradient_accumulation_steps,
        gradient_clipping=conf.gradient_clipping,
    )
    accelerator = Accelerator(
        project_config=proj_conf,
        deepspeed_plugin=deepspeed_plugin,
        gradient_accumulation_steps=conf.gradient_accumulation_steps,
        mixed_precision='bf16' if opt.mixed_precision else None,
        log_with=['wandb', 'tensorboard'] if not opt.test else None,
    )
    checkpoint_path = os.path.join(project_path, 'checkpoints')

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
        level=logging.INFO,
    )

    eff_bs = conf.batch_size * accelerator.num_processes * conf.gradient_accumulation_steps

    if not conf.model.use_in_hand:
        conf.dataset.meta_data.in_hand_views = 0
    if not conf.model.use_fixed:
        conf.dataset.meta_data.fixed_views = 0

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # hyperparameters
    hps = {
        # Dataset
        'dataset': str(conf.dataset.meta_data.path),
        'dataset_split': f'{conf.dataset.meta_data.split}/{conf.dataset.val_meta_data.split}',
        'val_meta_data': str(conf.dataset.val_meta_data),

        'sample_interval': conf.dataset.sample_interval,
        'observation_horizon': conf.dataset.obs_horizon,
        'action_horizon': conf.dataset.act_horizon,
        'action_representation': conf.dataset.act_repr,
        'timestamp_unit': conf.dataset.timestamp_unit,
        'geometry_augmentation': conf.dataset.geo_aug,

        'configs': str(conf.dataset.meta_data.configs) if conf.dataset.meta_data.configs is not None else 'all',
        'tasks': str(conf.dataset.meta_data.tasks) if conf.dataset.meta_data.tasks is not None else 'all',

        'main_camera': conf.dataset.meta_data.camera_ids.main,
        'in_hand_cameras': str(conf.dataset.meta_data.camera_ids.in_hand) if conf.dataset.meta_data.camera_ids.in_hand is not None else 'all',
        'fixed_cameras': str(conf.dataset.meta_data.camera_ids.fixed) if conf.dataset.meta_data.camera_ids.fixed is not None else 'any',

        # Policy
        'model': conf.model.name,
        'obs_dim': conf.model.obs_dim,
        'model_weight_decay': conf.model.weight_decay,

        'use_proprio': conf.model.use_proprio,
        'timesteps': conf.model.timesteps,

        'in_hand_views': conf.dataset.meta_data.in_hand_views,
        'fixed_views': conf.dataset.meta_data.fixed_views,
        'image_encoder': conf.model.image_encoder.name,
        'i_pretrain': conf.model.image_encoder.pretrain,
        'i_lr_ratio': conf.model.image_encoder.lr_ratio,
        'i_weight_decay': conf.model.image_encoder.weight_decay,

        # Training config
        'mixed_precision': opt.mixed_precision,
        'learning_rate': conf.learning_rate,
        'effective_batch_size': eff_bs,
        'lr_scheduler': conf.lr_scheduler,
        'lr_cycles': conf.lr_cycles if conf.lr_scheduler != 'constant' else 'None',
        'warm_up_steps': conf.warm_up_steps,
        'gradient_clipping': conf.gradient_clipping or 'None',
    }

    if opt.test:
        logger.info('======== Test mode ========')
        logger.info(hps)
    else:
        if opt.resume is not None and not opt.new_run:
            wandb_params = {"id": conf.run_id, "resume": "must"}
        else:
            run_id = wandb.util.generate_id()
            wandb_params = {"id": run_id}
            OmegaConf.update(conf, "run_id", run_id)
        
        # save all code for reproducibility
        wandb_params["save_code"] = True
        wandb_params["settings"] = wandb.Settings(code_dir=".")

        # TODO: change this
        accelerator.init_trackers("diffusion_policy", config=hps, init_kwargs={"wandb": wandb_params})

        if accelerator.is_main_process:
            os.makedirs(project_path, exist_ok=True)

            # dump current training config
            OmegaConf.save(conf, f=os.path.join(project_path, 'config.yaml'))

    set_seed(opt.seed)
    # enable tf32 calculation
    torch.backends.cuda.matmul.allow_tf32 = True
    # For mixed precision training we cast frozen model to bf16 to save GPU memory
    weight_dtype = torch.bfloat16 if opt.mixed_precision else torch.float32

    # Dataset
    # note: use cache to speed up data loading
    train_set = RealWorldDataset(
        conf.dataset.obs_horizon,
        conf.dataset.act_horizon,
        conf.dataset.act_repr,
        conf.dataset.meta_data,
        conf.dataset.sample_interval,
        conf.dataset.timestamp_unit,
        augment=conf.dataset.geo_aug,
        augment_color=True,
        save_cache=accelerator.is_main_process,
    )
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=conf.batch_size,
        num_workers=conf.dataset.num_workers,
        persistent_workers=True,
    )
    
    val_set = RealWorldDataset(
        conf.dataset.obs_horizon,
        conf.dataset.act_horizon,
        conf.dataset.act_repr,
        OmegaConf.merge(conf.dataset.meta_data, conf.dataset.val_meta_data),
        conf.dataset.sample_interval,
        conf.dataset.timestamp_unit,
        save_cache=accelerator.is_main_process,
    )
    val_loader = DataLoader(
        val_set,
        shuffle=True,
        batch_size=conf.batch_size,
        num_workers=conf.dataset.num_workers//2,
        persistent_workers=True,
    )

    model = initialize_model(conf)

    print_model_info(model.obs_encoders)
    print_model_info(model.model)   # unet/transformer
    
    print_model_info(model)

    # optimize for torch >= 2.0
    # model = torch.compile(model, mode='reduce-overhead')

    # if resuming, determine which checkpoint to load from
    if opt.resume is not None:
        loading_dir = os.path.join(opt.output_dir, os.path.dirname(opt.resume), "checkpoints")
        checkpoint = os.path.basename(opt.resume)
        if checkpoint == "latest":
            # Get the most recent checkpoint
            dirs = os.listdir(loading_dir)
            dirs = [d for d in dirs if d.startswith('checkpoint')]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            checkpoint = dirs[-1] if len(dirs) > 0 else None
        if checkpoint is None:
            raise ValueError(f'Checkpoint {opt.resume} does not exist!')
        
        logger.info(f"Resuming from {checkpoint}")
        
        # if ONLY load model weights, do it before prepare.
        if opt.weights_only:
            logger.info(f"Loading model weights only...")
            model.load_state_dict(torch.load(os.path.join(loading_dir, checkpoint, 'pytorch_model.bin'), map_location=accelerator.device))

    # optimizer
    # parameters: model.obs_encoders + model.model (unet/transformer)
    params = model.get_optim_groups(
        weight_decay = conf.model.weight_decay,
        i_lr = conf.model.image_encoder.lr_ratio * conf.learning_rate,
        i_weight_decay = conf.model.image_encoder.weight_decay,
    )
    optimizer = AdamW(params, lr=conf.learning_rate)

    # denoise scheduler
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=conf.model.timesteps,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
    )

    # learning rate scheduler
    total_steps = int(np.ceil(len(train_loader)/conf.gradient_accumulation_steps))
    if conf.lr_scheduler == 'constant':
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps = conf.warm_up_steps
        )
    elif conf.lr_scheduler == 'cosine':
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps   = conf.warm_up_steps,
            num_training_steps = total_steps * conf.epoch,
            num_cycles         = conf.lr_cycles
        )
    elif conf.lr_scheduler == 'cosine_with_restarts':
        lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer,
            num_warmup_steps   = conf.warm_up_steps,
            num_training_steps = total_steps * conf.epoch,
            num_cycles         = conf.lr_cycles
        )
    else:
        raise NotImplementedError(f"Unknown LR scheduler {conf.lr_scheduler}")

    # prepare everything for distributed training if needed
    model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, lr_scheduler
    )

    # use the first validation batch to compute metrics during training (different on each GPU)
    first_val_batch = next(iter(val_loader))

    logger.info("***** Start training *****")
    logger.info(f"  Num examples = {len(train_set)}")
    logger.info(f"  Num Epochs = {conf.epoch}")
    logger.info(f"  Num GPUs = {accelerator.num_processes}")
    logger.info(f"  Instantaneous batch size per device = {conf.batch_size}")
    logger.info(f"  Gradient accumulation steps = {conf.gradient_accumulation_steps}")
    logger.info(f"  Effective batch size = {eff_bs}")

    global_step = 0
    initial_epoch = 0
    initial_step = 0
    steps_per_epoch = int(np.ceil(len(train_loader)/conf.gradient_accumulation_steps))

    # Potentially load in the weights and states from a previous save
    if opt.resume is not None:
        global_step = int(checkpoint.split("-")[1])
        initial_epoch = global_step // steps_per_epoch

        if not opt.weights_only:
            logger.info(f"Loading whole training state...")
            accelerator.load_state(os.path.join(loading_dir, checkpoint))

            initial_step = global_step % steps_per_epoch

            train_loader.set_epoch(initial_epoch)
            skipped_loader = accelerator.skip_first_batches(train_loader, conf.gradient_accumulation_steps * initial_step)

    progress_bar = tqdm(
        initial=initial_step,
        total=steps_per_epoch,
        desc="Steps",
        disable=not accelerator.is_local_main_process
    )

    best_val_loss = np.inf
    if os.path.exists(os.path.join(project_path, 'best-epoch')):
        for f in os.listdir(os.path.join(project_path, 'best-epoch')):
            if f.startswith('epoch-'):
                break
        best_val_loss = float(f.split('-')[-1])
    for epoch in range(initial_epoch, conf.epoch):
        logger.info(f"***** Epoch {epoch} *****")

        model.train()

        train_loss = 0.
        train_epoch_loss = 0.

        if opt.resume is not None and not opt.weights_only and epoch == initial_epoch:
            loader = skipped_loader
            base_step = initial_step * conf.gradient_accumulation_steps
        else:
            loader = train_loader
            base_step = 0
        accumulated_steps = 0
        base_global_step = global_step

        for step, batch in enumerate(loader):
            with accelerator.accumulate(model):
                accumulated_steps += 1
                # Compute loss
                loss = calc_loss(batch, model, noise_scheduler, conf.model.use_proprio, weight_dtype)
                avg_loss = loss.mean()

                # Backpropagate
                accelerator.backward(avg_loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            # Gather the losses across all processes for logging
            all_loss = accelerator.gather_for_metrics(loss)
            # Incremental average
            train_loss += (all_loss.mean().item() - train_loss) / accumulated_steps
            
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                accumulated_steps = 0
                global_step += 1

                train_epoch_loss += (train_loss - train_epoch_loss) / (global_step - base_global_step)

                logs = {"step": base_step+step, "lr": lr_scheduler.get_last_lr()[0], "train_loss": train_loss}
                train_loss = 0.

                if global_step % conf.validation_step_interval == 0:
                    model.eval()

                    generator = torch.Generator(device=accelerator.device).manual_seed(opt.seed)
                    with torch.no_grad():
                        pred_actions = sample_actions(
                            first_val_batch, generator, 
                            model, noise_scheduler,
                            conf.model.use_proprio,
                            weight_dtype)
                    real_actions = first_val_batch['actions'].float()
                    base_pose = first_val_batch['obs_pose'].float()

                    pred_actions, real_actions, base_pose = accelerator.gather_for_metrics((pred_actions, real_actions, base_pose))
                    pred_actions = pred_actions.cpu().numpy()
                    real_actions = real_actions.cpu().numpy()
                    base_pose = base_pose.cpu().numpy()

                    pred_actions = get_actual_actions(pred_actions, base_pose, conf.dataset.act_repr)
                    pred_xyz = pred_actions[..., :3]
                    pred_rot = rot6d_to_rotation_matrix(pred_actions[..., 3:9])
                    pred_width = pred_actions[..., 9]

                    real_actions = get_actual_actions(real_actions, base_pose, conf.dataset.act_repr)
                    real_xyz = real_actions[..., :3]
                    real_rot = rot6d_to_rotation_matrix(real_actions[..., 3:9])
                    real_width = real_actions[..., 9]

                    eps = 1e-4

                    width_err = np.abs(pred_width - real_width)
                    width_err_pct = width_err / np.maximum(np.abs(real_width), eps)

                    trans_err = np.sqrt(((pred_xyz - real_xyz)**2).sum(axis=-1))
                    real_trans = np.sqrt((real_xyz**2).sum(axis=-1))
                    trans_err_pct = trans_err / np.maximum(real_trans, eps)

                    diff = pred_rot @ real_rot.swapaxes(2, 3)
                    trace = diff[..., 0,0] + diff[..., 1,1] + diff[..., 2,2]
                    cos_alpha = (trace-1) / 2
                    rot_err = np.arccos(cos_alpha.clip(-1, 1))
                    trace = real_rot[..., 0,0] + real_rot[..., 1,1] + real_rot[..., 2,2]
                    cos_alpha = (trace-1) / 2
                    real_angle = np.arccos(cos_alpha.clip(-1, 1))
                    rot_err_pct = rot_err / np.maximum(real_angle, eps)

                    # shape: Ta
                    width_err = width_err.mean(axis=0)
                    # take the median instead of mean to avoid extreme values
                    width_err_pct = np.quantile(width_err_pct, 0.5, axis=0)

                    trans_err = trans_err.mean(axis=0)
                    trans_err_pct = np.quantile(trans_err_pct, 0.5, axis=0)

                    rot_err = rot_err.mean(axis=0) / np.pi * 180
                    rot_err_pct = np.quantile(rot_err_pct, 0.5, axis=0)

                    steps = np.arange(conf.dataset.act_horizon)
                    width_fig = px.line(x=steps, y=width_err).update_layout(
                        xaxis_title='Steps', yaxis_title='Width Error',
                        yaxis_range=[0, 0.02],
                    )
                    width_pct_fig = px.line(x=steps, y=width_err_pct).update_layout(
                        xaxis_title='Steps', yaxis_title='% Width Error',
                        yaxis_range=[0, 2],
                    )

                    trans_fig = px.line(x=steps, y=trans_err).update_layout(
                        xaxis_title='Steps', yaxis_title='Translation Error',
                        yaxis_range=[0, 0.05],
                    )
                    trans_pct_fig = px.line(x=steps, y=trans_err_pct).update_layout(
                        xaxis_title='Steps', yaxis_title='% Translation Error',
                        yaxis_range=[0, 2],
                    )

                    rot_fig = px.line(x=steps, y=rot_err).update_layout(
                        xaxis_title='Steps', yaxis_title='Rotation Error',
                        yaxis_range=[0, 5],
                    )
                    rot_pct_fig = px.line(x=steps, y=rot_err_pct).update_layout(
                        xaxis_title='Steps', yaxis_title='% Rotation Error',
                        yaxis_range=[0, 2],
                    )

                    if opt.test:
                        accelerator.print(f'Avg width error/pct: {width_err.mean():.4f}/{width_err_pct.mean():.4f}, Avg translation error/pct: {trans_err.mean():.4f}/{trans_err_pct.mean():.4f}, Avg rotation error/pct: {rot_err.mean():.4f}/{rot_err_pct.mean():.4f}')
                    else:
                        accelerator.log({
                            'width_err': width_err.mean(),
                            'width_err_min': width_err.min(),
                            'width_err_max': width_err.max(),
                            'width_pct_err': width_err_pct.mean(),
                            'width_pct_err_min': width_err_pct.min(),
                            'width_pct_err_max': width_err_pct.max(),

                            'trans_err': trans_err.mean(),
                            'trans_err_min': trans_err.min(),
                            'trans_err_max': trans_err.max(),
                            'trans_pct_err': trans_err_pct.mean(),
                            'trans_pct_err_min': trans_err_pct.min(),
                            'trans_pct_err_max': trans_err_pct.max(),

                            'rot_err': rot_err.mean(),
                            'rot_err_min': rot_err.min(),
                            'rot_err_max': rot_err.max(),
                            'rot_pct_err': rot_err_pct.mean(),
                            'rot_pct_err_min': rot_err_pct.min(),
                            'rot_pct_err_max': rot_err_pct.max(),

                            'width_err_per_step': width_fig,
                            'width_pct_err_per_step': width_pct_fig,
                            'trans_err_per_step': trans_fig,
                            'trans_pct_err_per_step': trans_pct_fig,
                            'rot_err_per_step': rot_fig,
                            'rot_pct_err_per_step': rot_pct_fig,
                        }, step=global_step)
                    
                    model.train()
                postfix = dict(logs)
                progress_bar.set_postfix(**postfix)
                progress_bar.update(1)

                if opt.test:
                    continue

                accelerator.log(logs, step=global_step)

                # save checkpoint each checkpoint_steps
                if conf.checkpoint_steps is not None and global_step % conf.checkpoint_steps == 0:
                    save_checkpoint(checkpoint_path, accelerator, global_step, opt.checkpoint_total_limit)
        if not opt.test:
            accelerator.log({"train_epoch_loss": train_epoch_loss}, step=global_step)

        model.eval()

        val_loss = 0.0
        valid_steps = len(val_loader)
        val_progress_bar = tqdm(val_loader, desc="Validation steps", disable=not accelerator.is_local_main_process)
        for batch in val_progress_bar:
            # calculate val loss
            with torch.no_grad():
                loss = calc_loss(batch, model, noise_scheduler, conf.model.use_proprio, weight_dtype)
            all_loss = accelerator.gather_for_metrics(loss)
            cur_loss = all_loss.mean().item()

            val_progress_bar.set_postfix({"loss": cur_loss})
            val_loss += cur_loss / valid_steps
        if not opt.test:
            accelerator.log({"val_epoch_loss": val_loss}, step=global_step)

            # extra checkpoint at the end of epoch
            save_checkpoint(checkpoint_path, accelerator, global_step, opt.checkpoint_total_limit, end_of_epoch=True)

            if accelerator.is_main_process and val_loss < best_val_loss:
                best_val_loss = val_loss
                logger.info(f'new best ckpt at epoch {epoch} with val loss {val_loss:.4f}')

                save_path = os.path.join(project_path, 'best-epoch')
                
                # delete old best-epoch
                if os.path.exists(save_path):
                    shutil.rmtree(save_path)
                
                shutil.copytree(os.path.join(checkpoint_path, f"checkpoint-{global_step}-epoch"), save_path)
                # dump an empty file to indicate epoch & val_loss
                with open(os.path.join(save_path, f'epoch-{epoch}-val_loss-{val_loss:.4f}'), 'w') as _:
                    pass

        progress_bar.reset()

    accelerator.wait_for_everyone()

    # save the final model
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        # save the final model
        torch.save(model.state_dict(), os.path.join(project_path, 'final_model.bin'))
    
    accelerator.end_training()

def save_checkpoint(checkpoint_path, accelerator, global_step, total_limit=0, end_of_epoch=False):
    if accelerator.is_main_process and total_limit != 0 and os.path.exists(checkpoint_path):
        checkpoints = os.listdir(checkpoint_path)
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

        # Delete excess checkpoints
        if len(checkpoints) > total_limit:
            num_to_remove = len(checkpoints) - total_limit
            removing_checkpoints = checkpoints[0:num_to_remove]

            logger.info(
                f"{len(checkpoints)-1} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                )
            logger.info(f"removing earliest checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(checkpoint_path, removing_checkpoint)
                shutil.rmtree(removing_checkpoint)
    save_path = os.path.join(checkpoint_path, f"checkpoint-{global_step}")
    if end_of_epoch:
        save_path += "-epoch"
    # deepspeed requires all processes to call `save_state`
    accelerator.save_state(save_path)

if __name__ == '__main__':
    main()
