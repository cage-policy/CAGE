import torch
import torch.nn as nn

from model.diffusion.blocks import TransformerBlock


class CausalObservationPerceiver(nn.Module):
    def __init__(self, dim, obs_dim, obs_horizon, layers=1, dropout=0.):
        super().__init__()

        self.latents = nn.Parameter(torch.randn(1, obs_horizon, dim))

        self.obs_norm = nn.LayerNorm(obs_dim)

        self.x_attn = TransformerBlock(dim, cond_dim=obs_dim, dropout=dropout, cross_attn_only=True)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, dropout=dropout) for _ in range(layers)
        ])

    def forward(self, obs_emb):
        B, N, T, L, D = obs_emb.shape

        obs_emb = obs_emb.flatten(1, -2)
        obs_emb = self.obs_norm(obs_emb)

        mask = torch.ones(T, T, dtype=torch.bool).tril()
        mask = mask.unsqueeze(0).expand(B, T, T).to(device=obs_emb.device)
        cond_mask = mask.reshape(B, T, T, 1).repeat(1, 1, N, L).reshape(B, T, N*T*L)

        latents = self.latents.expand(B, T, D)
        latents = self.x_attn(latents, obs_emb, cond_mask=cond_mask)
        
        for block in self.blocks:
            latents = block(latents, mask=mask)

        return latents
