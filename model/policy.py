import torch
from diffusers.models.embeddings import get_timestep_embedding
from torch import nn

from model.diffusion.conditional_unet1d import ConditionalUnet1D


# baseline cnn-based diffusion policy
class DiffusionPolicyUNet(nn.Module):
    def __init__(self, obs_encoders, act_dim, obs_num=1, obs_horizon=1, obs_dim=512):
        super().__init__()

        self.obs_encoders = obs_encoders
        self.model = ConditionalUnet1D(
            act_dim,
            global_cond_dim = obs_num * obs_horizon * obs_dim,
            diffusion_step_embed_dim = 128,
            down_dims = [256, 512, 1024],
            conv_kernel_size = 5,
            cond_predict_scale = True,
        )

    def preprocess_obs(self, obs_dict):
        all_obs = []
        # sort different types of obs in a fixed order
        obs_keys = sorted(obs_dict.keys())
        for obs_type in obs_keys:
            obs = obs_dict[obs_type]    # B, N, T, C, H, W
            B = obs.shape[0]

            obs_emb = self.obs_encoders[obs_type](obs.flatten(0, 2))    # B*N*T, 1, D
            all_obs.append(obs_emb.reshape(B, -1))  # B, N*T*D
        all_obs = torch.cat(all_obs, dim=-1)

        return all_obs
    
    def forward(self, noisy_actions, timesteps, obs_emb=None, obs_dict=None):
        if obs_emb is None:
            obs_emb = self.preprocess_obs(obs_dict)

        return self.model(noisy_actions, timesteps, cond=obs_emb)
    
    def get_optim_groups(self, weight_decay=1e-2, i_lr=None, i_weight_decay=None):
        params = []

        model_params = {
            "params": self.model.parameters(),
            "weight_decay": weight_decay,
        }
        obs_params = {
            'params': [p for p in self.obs_encoders.parameters() if p.requires_grad],
            'weight_decay': i_weight_decay or weight_decay,
        }
        if i_lr is not None:
            obs_params['lr'] = i_lr
        
        params.append(model_params)
        params.append(obs_params)

        return params


class CAGE(nn.Module):
    def __init__(self, obs_encoders, perceiver, backbone,
                 obs_dim=512, obs_horizon=1, obs_num=1):
        super().__init__()

        self.obs_time_emb = nn.Parameter(get_timestep_embedding(torch.arange(obs_horizon), obs_dim).reshape(1, 1, obs_horizon, 1, obs_dim))
        self.obs_type_emb = nn.Parameter(torch.zeros(1, obs_num, 1, 1, obs_dim))
        
        self.obs_encoders = obs_encoders
        self.perceiver = perceiver
        self.backbone = backbone

    def preprocess_obs(self, obs_dict):
        all_obs = []
        # sort different types of obs in a fixed order
        obs_keys = sorted(obs_dict.keys())
        for obs_type in obs_keys:
            obs = obs_dict[obs_type]
            B, N, T = obs.shape[:3]
            o = []
            for i in range(N):
                oo = []
                for j in range(T):
                    oo.append(self.obs_encoders[obs_type](obs[:, i, j])) # B, 1 or L, D
                o.append(torch.stack(oo, dim=1))    # B, T, 1 or L, D
            all_obs.append(torch.stack(o, dim=1))    # B, N, T, 1 or L, D
        all_obs = torch.cat(all_obs, dim=1)        # B, obs_num, T, 1 or L, D
        
        obs_emb = all_obs + self.obs_time_emb + self.obs_type_emb
        if self.perceiver is not None:
            obs_emb = self.perceiver(obs_emb)   # B, T, D

        return obs_emb

    def forward(self, noisy_actions, timesteps, obs_emb=None, obs_dict=None):
        if obs_emb is None:
            obs_emb = self.preprocess_obs(obs_dict)

        return self.backbone(noisy_actions, timesteps, cond=obs_emb)
    
    def get_optim_groups(self, weight_decay=1e-2, i_lr=None, i_weight_decay=None):
        params = []

        model_params = {
            "params": [p for n, p in self.named_parameters() if not n.startswith('obs_encoders')],
            "weight_decay": weight_decay,
        }
        obs_params = {
            'params': [p for p in self.obs_encoders.parameters() if p.requires_grad],
            'weight_decay': i_weight_decay or weight_decay,
        }
        if i_lr is not None:
            obs_params['lr'] = i_lr
        
        params.append(model_params)
        params.append(obs_params)

        return params
