import time
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import torch
from accelerate.utils import set_seed
from diffusers.schedulers import DDIMScheduler

from agent.cage import CAGEAgent
from train_diffusion_policy import initialize_model
from utils.constants import *
from utils.preprocessings import get_actual_actions, get_normalized_actions


class DiffusionPolicyAgent(CAGEAgent):
    """
    Agent for Diffusion Policy
        takes as input images from Nc global and wrist cameras.
    """
    @staticmethod
    def _policy_runner(configs, shared_status, obs_lock, act_lock, new_obs, pred_done):
        set_seed(configs.seed)
        torch.backends.cuda.matmul.allow_tf32 = True

        max_timestep = configs.max_timestep

        act_repr = configs.dataset.act_repr
        act_horizon = configs.dataset.act_horizon
        # action horizon for prediction
        act_horizon_eval = configs.act_horizon_eval

        obs_horizon = configs.dataset.obs_horizon

        device = torch.device(f"cuda:{configs.device}" if torch.cuda.is_available() else "cpu")

        # load policy
        policy = initialize_model(configs)
        policy.load_state_dict(torch.load(configs.ckpt))
        policy.to(device)
        policy.eval()
        print(f'[Agent] Policy loaded')
        # signal main process that policy is ready
        shared_status[0] = 1

        noise_scheduler = DDIMScheduler(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon",
        )

        # link shared memory with main process
        obs_global_shm = None
        if configs.model.use_fixed:
            obs_global_shm = SharedMemory(name='obs_global')
            obs_global = np.ndarray((1, configs.dataset.meta_data.fixed_views, obs_horizon, 3, 224, 224), dtype=np.float32, buffer=obs_global_shm.buf)
        obs_wrist_shm = None
        if configs.model.use_in_hand:
            obs_wrist_shm = SharedMemory(name='obs_wrist')
            obs_wrist = np.ndarray((1, configs.dataset.meta_data.in_hand_views, obs_horizon, 3, 224, 224), dtype=np.float32, buffer=obs_wrist_shm.buf)

        tcp_horizon = 1
        if configs.model.use_proprio:
            tcp_horizon = obs_horizon
        obs_tcp_shm = SharedMemory(name='obs_tcp')
        obs_tcp = np.ndarray((tcp_horizon, 3+9+1), dtype=np.float32, buffer=obs_tcp_shm.buf)

        act_shm = SharedMemory(name='action_buffer')
        action_buffer = np.ndarray((max_timestep, max_timestep+act_horizon_eval, ACTION_DIM), dtype=np.float32, buffer=act_shm.buf)

        try:
            while True:
                # everything is ready, wait for observation
                new_obs.wait()

                with obs_lock:
                    pred_step = shared_status[1]

                    obs_dict = {}
                    if configs.model.use_fixed:
                        obs_dict['fixed_obs'] = torch.from_numpy(obs_global).to(device=device)
                    if configs.model.use_in_hand:
                        obs_dict['in_hand_obs'] = torch.from_numpy(obs_wrist).to(device=device)
                    tcp = obs_tcp.copy()
                    new_obs.clear()     # reset new observation flag

                base_pose = np.zeros((4, 4))
                base_pose[:3, 3] = (tcp[-1, :3] - TRANS_MIN) / (TRANS_MAX - TRANS_MIN)
                base_pose[:3, :3] = tcp[-1, 3:-1].reshape(3, 3)
                base_pose[3, 3] = 1

                if configs.model.use_proprio:
                    proprio = get_normalized_actions(tcp, base_pose, act_repr)
                    proprio = torch.from_numpy(proprio).unsqueeze(0).to(device)

                # Timesteps
                noise_scheduler.set_timesteps(configs.denoising_steps, device=device)

                # Initial noise
                pred_actions = torch.randn((1, act_horizon, ACTION_DIM), device=device)
                pred_actions = pred_actions * noise_scheduler.init_noise_sigma

                tic = time.time()
                with torch.inference_mode():
                    obs_emb = policy.preprocess_obs(obs_dict)
                    for t in noise_scheduler.timesteps:
                        if configs.model.use_proprio:
                            pred_noise = policy(torch.cat([proprio, pred_actions], dim=1), t, obs_emb=obs_emb)[:, obs_horizon:, :]
                        else:
                            pred_noise = policy(pred_actions, t, obs_emb=obs_emb)
                        # compute the previous noisy sample x_t -> x_t-1
                        pred_actions = noise_scheduler.step(
                            pred_noise, t, pred_actions).prev_sample
                toc = time.time()
                print(f'======== Prediction used {toc-tic:.4f} seconds ========')

                pred_actions = get_actual_actions(pred_actions[0].cpu().numpy(), base_pose, act_repr)
                # update the action buffer
                with act_lock:
                    step = shared_status[2]
                    end_step = pred_step + act_horizon_eval
                    action_buffer[step:end_step, step:end_step] = pred_actions[None, step - pred_step:act_horizon_eval]

                # inform that the prediction is done
                pred_done.set()
        finally:
            # clean up
            if obs_global_shm is not None:
                obs_global_shm.close()
            if obs_wrist_shm is not None:
                obs_wrist_shm.close()
            obs_tcp_shm.close()

            act_shm.close()
