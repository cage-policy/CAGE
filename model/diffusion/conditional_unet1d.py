import numpy as np
import torch
import torch.nn as nn
from diffusers.models.embeddings import Timesteps

from model.diffusion.blocks import *


# copied from Diffusion Policy, removed local_cond
class ConditionalUnet1D(nn.Module):
    def __init__(self, 
        input_dim,
        global_cond_dim=None,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        conv_kernel_size=3,
        n_groups=8,
        cond_predict_scale=False
        ):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = Timesteps(dsed, flip_sin_to_cos=False, downscale_freq_shift=1)
        diffusion_step_proj = nn.Sequential(
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed
        if global_cond_dim is not None:
            cond_dim += global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=conv_kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=conv_kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=conv_kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim, 
                    kernel_size=conv_kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=conv_kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=conv_kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))
        
        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=conv_kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.diffusion_step_proj = diffusion_step_proj
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

    def forward(self, sample, timestep, cond=None):
        """
        Parameters:
            sample: (B, Ha, Da)
            timestep: (B,) or int, diffusion step
            cond: (B, ...)

        Returns:
            output: (B, Ha, Da)
        """
        sample = sample.permute(0, 2, 1)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.diffusion_step_encoder(timesteps)
        t_emb = t_emb.to(dtype=sample.dtype)
        
        global_feature = self.diffusion_step_proj(t_emb)

        if cond is not None:
            cond = cond.flatten(1, -1)  # B, D
            global_feature = torch.cat([
                global_feature, cond
            ], dim=-1)
        
        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        return x.permute(0, 2, 1)


# added attn after each residual block
class ConditionalAttnUnet1D(nn.Module):
    def __init__(self, 
                 input_dim,
                 obs_dim,
                 down_dims=[256,512,1024],
                 layers_per_block=1,
                 conv_kernel_size=3,
                 num_norm_groups=8,
                 num_attn_heads=8,
                 ):
        super().__init__()

        self.timestep_emb = Timesteps(obs_dim, flip_sin_to_cos=False, downscale_freq_shift=1)
        self.timestep_proj = nn.Sequential(
            nn.Linear(obs_dim, obs_dim * 4),
            nn.SiLU(),
            nn.Linear(obs_dim * 4, obs_dim),
        )

        self.cond_norm = nn.LayerNorm(obs_dim)

        self.conv_in = nn.Conv1d(input_dim, down_dims[0], kernel_size=conv_kernel_size, padding=conv_kernel_size // 2)

        all_dims = [down_dims[0]] + down_dims
        in_out_dims = list(zip(all_dims[:-1], all_dims[1:]))

        down_modules = []
        for i, (dim_in, dim_out) in enumerate(in_out_dims):
            if i == len(in_out_dims) - 1:
                down_modules.append(DownBlock1D(
                    dim_in, dim_out, timestep_dim=obs_dim,
                    num_layers=layers_per_block, add_downsample=False,
                    kernel_size=conv_kernel_size, num_groups=num_norm_groups,
                ))
            else:
                down_modules.append(AttnDownBlock1D(
                    dim_in, dim_out, cond_dim=obs_dim,
                    num_layers=layers_per_block, add_downsample=True,
                    kernel_size=conv_kernel_size, num_groups=num_norm_groups,
                    num_heads=num_attn_heads,
                ))
        self.down_modules = nn.ModuleList(down_modules)

        self.mid_module = AttnMidBlock1D(
            all_dims[-1], cond_dim=obs_dim,
            num_layers=1,
            kernel_size=conv_kernel_size, num_groups=num_norm_groups,
            num_heads=num_attn_heads,
        )

        up_modules = []
        for i, (dim_out, dim_in) in enumerate(reversed(in_out_dims)):
            is_last = i == len(in_out_dims) - 1
            if i == 0:
                up_modules.append(UpBlock1D(
                    dim_in*2, dim_out, timestep_dim=obs_dim,
                    num_layers=layers_per_block, add_upsample=not is_last,
                    kernel_size=conv_kernel_size, num_groups=num_norm_groups,
                ))
            else:
                up_modules.append(AttnUpBlock1D(
                    dim_in*2, dim_out, cond_dim=obs_dim,
                    num_layers=layers_per_block, add_upsample=not is_last,
                    kernel_size=conv_kernel_size, num_groups=num_norm_groups,
                    num_heads=num_attn_heads,
                ))
        self.up_modules = nn.ModuleList(up_modules)
        
        self.conv_out = nn.Sequential(
            nn.GroupNorm(num_norm_groups, down_dims[0]),
            nn.Mish(),
            nn.Conv1d(down_dims[0], input_dim, kernel_size=conv_kernel_size, padding=conv_kernel_size // 2),
        )

        # initialize weights
        self.apply(self._init_weights)
        self._rescale_blocks()

    def forward(self, sample, timesteps, cond):
        """
        Parameters:
            sample: (B, Ta, Da)
            timestep: (B,) or int, diffusion step
            cond: (B, ..., D)

        Returns:
            output: (B, Ta, Da) predicted noise
        """
        sample = sample.permute(0, 2, 1)

        if not torch.is_tensor(timesteps):
            # this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.timestep_emb(timesteps)
        t_emb = t_emb.to(dtype=sample.dtype)
        t_emb = self.timestep_proj(t_emb)

        cond = cond.flatten(1, -2)  # B, L, D
        cond = self.cond_norm(cond)
        
        x = self.conv_in(sample)
        
        h = []
        for block in self.down_modules:
            x = block(x, t_emb, cond)
            h.append(x[1])
            x = x[0]

        x = self.mid_module(x, t_emb, cond)

        for block in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1)
            x = block(x, t_emb, cond)[0]

        x = self.conv_out(x)
        return x.permute(0, 2, 1)    # B, Ta, Da

    def _init_weights(self, module):
        init_std = 0.02
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.trunc_normal_(module.weight, std=init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def _rescale_blocks(self):
        def rescale(module, layer):
            for m in module.modules():
                if isinstance(m, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
                    m.weight.data.div_(np.sqrt(2.0 * layer))

        for i, module in enumerate(self.down_modules):
            rescale(module, i+1)

        rescale(self.mid_module, len(self.down_modules)+1)

        for i, module in enumerate(reversed(self.up_modules)):
            rescale(module, i+1)
