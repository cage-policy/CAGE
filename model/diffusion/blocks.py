import torch
import torch.nn as nn
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from einops.layers.torch import Rearrange


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """
        Conv1d --> GroupNorm --> Mish
    """
    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


# legacy resnet block (norm last)
class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, 
            in_channels, 
            out_channels, 
            cond_dim,
            kernel_size=3,
            n_groups=8,
            cond_predict_scale=False):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange('batch t -> batch t 1'),
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        """
        Parameters:
            x: (B, C, L)
            cond: (B, D)

        Returns:
            out: (B, C, L)
        """
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embed = embed.reshape(
                embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:,0,...]
            bias = embed[:,1,...]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


# Updated ResNetBlock with norm first, to be used with TransformerBlock
class ResNetBlock1D(nn.Module):
    def __init__(self, 
                 dim_in, dim_out, 
                 cond_dim=None,
                 dropout=0.,
                 kernel_size=3,
                 num_groups=8,
                 scale_shift=True):
        super().__init__()

        self.act = nn.Mish()
        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.GroupNorm(num_groups, dim_in)
        self.conv1 = nn.Conv1d(dim_in, dim_out, kernel_size, padding=kernel_size // 2)

        self.norm2 = nn.GroupNorm(num_groups, dim_out)
        self.conv2 = nn.Conv1d(dim_out, dim_out, kernel_size, padding=kernel_size // 2)

        self.cond_proj = None
        if cond_dim is not None:
            self.cond_type = 'scale_shift' if scale_shift else 'add'
            self.cond_proj = nn.Linear(cond_dim, dim_out * 2 if scale_shift else dim_out)

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()
        
    def forward(self, x, cond=None):
        """
        Parameters:
            x: (B, C, L)
            cond: (B, D)
        
        Returns:
            out: (B, C, L)
        """
        out = self.norm1(x)
        out = self.act(out)
        out = self.conv1(out)

        if self.cond_proj is not None:
            cond_emb = self.act(cond)
            cond_emb = self.cond_proj(cond_emb).unsqueeze(-1)

            if self.cond_type == 'scale_shift':
                out = self.norm2(out)

                scale, shift = cond_emb.chunk(2, dim=1)
                out = out * (1+scale) + shift
            else:
                out = out + cond_emb
                out = self.norm2(out)
        else:
            out = self.norm2(out)

        out = self.act(out)
        out = self.dropout(out)
        out = self.conv2(out)

        out = out + self.residual_conv(x)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, cond_dim=None, num_heads=8, dropout=0., cross_attn_only=False):
        super().__init__()

        self.cross_attn_only = cross_attn_only
        assert not cross_attn_only or cond_dim is not None, 'If only do cross attention, cond_dim must NOT be None!'

        # self attn
        if not cross_attn_only:
            self.norm1 = nn.LayerNorm(dim)
            self.attn1 = Attention(
                dim,
                heads=num_heads,
                dim_head=dim // num_heads,
                dropout=dropout,
            )

        # cross attn
        self.attn2 = None
        if cond_dim is not None:
            self.norm2 = nn.LayerNorm(dim)
            self.attn2 = Attention(
                dim,
                cross_attention_dim=cond_dim,
                heads=num_heads,
                dim_head=dim // num_heads,
                dropout=dropout,
            )

        # feedforward
        self.norm3 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dropout=dropout)

    def forward(self, x, cond=None, mask=None, cond_mask=None):
        """
        Parameters:
            x: (B, L, D)
            cond: (B, L, D)
            mask: (B, L, L)
            cond_mask: (B, L, L)

        Returns:
            out: (B, L, D)
        """
        if not self.cross_attn_only:
            norm_x = self.norm1(x)
            x = self.attn1(norm_x, attention_mask=mask) + x

        if self.attn2 is not None:
            norm_x = self.norm2(x)
            x = self.attn2(norm_x, cond, attention_mask=cond_mask) + x

        norm_x = self.norm3(x)
        x = self.ff(norm_x) + x

        return x


class TransformerBlock1D(TransformerBlock):
    def forward(self, x, cond=None, mask=None, cond_mask=None):
        x = x.permute(0, 2, 1)              # channel last (B, C, L) -> (B, L, D)
        x = super().forward(x, cond, mask, cond_mask)
        x = x.permute(0, 2, 1).contiguous() # swap back

        return x


# Below are the 1D versions of the blocks used in the UNet of SD
class AttnDownBlock1D(nn.Module):
    def __init__(self, input_dim, output_dim, cond_dim,
                 num_layers=1, add_downsample=True,
                 kernel_size=3, num_groups=8,
                 num_heads=8):
        super().__init__()

        resnets = []
        attentions = []
        for i in range(num_layers):
            dim_in = input_dim if i == 0 else output_dim
            resnets.append(
                ResNetBlock1D(
                    dim_in, output_dim,
                    cond_dim=cond_dim, 
                    kernel_size=kernel_size,
                    num_groups=num_groups,
                ),
            )
            attentions.append(
                TransformerBlock1D(
                    dim=output_dim,
                    cond_dim=cond_dim,
                    num_heads=num_heads,
                )
            )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        self.downsampler = Downsample1d(output_dim) if add_downsample else nn.Identity()

    def forward(self, x, t_emb, cond):
        for resnet, attn in zip(self.resnets, self.attentions):
            x = resnet(x, t_emb)
            x = attn(x, cond)
        
        return self.downsampler(x), x


class DownBlock1D(nn.Module):
    def __init__(self, input_dim, output_dim, timestep_dim,
                 num_layers=1, add_downsample=True,
                 kernel_size=3, num_groups=8):
        super().__init__()

        resnets = []
        for i in range(num_layers):
            dim_in = input_dim if i == 0 else output_dim
            resnets.append(
                ResNetBlock1D(
                    dim_in, output_dim,
                    cond_dim=timestep_dim, 
                    kernel_size=kernel_size,
                    num_groups=num_groups,
                ),
            )
        self.resnets = nn.ModuleList(resnets)

        self.downsampler = Downsample1d(output_dim) if add_downsample else nn.Identity()

    def forward(self, x, t_emb, cond=None):
        for resnet in self.resnets:
            x = resnet(x, t_emb)
        
        return self.downsampler(x), x


class AttnMidBlock1D(nn.Module):
    def __init__(self, dim, cond_dim,
                 num_layers=1,
                 kernel_size=3, num_groups=8,
                 num_heads=8):
        super().__init__()

        # there is always at least one resnet
        resnets = [
            ResNetBlock1D(
                dim, dim,
                cond_dim=cond_dim, 
                kernel_size=kernel_size,
                num_groups=num_groups,
            )
        ]
        attentions = []
        for _ in range(num_layers):
            resnets.append(
                ResNetBlock1D(
                    dim, dim,
                    cond_dim=cond_dim, 
                    kernel_size=kernel_size,
                    num_groups=num_groups,
                ),
            )
            attentions.append(
                TransformerBlock1D(
                    dim=dim,
                    cond_dim=cond_dim,
                    num_heads=num_heads,
                )
            )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, x, t_emb, cond):
        x = self.resnets[0](x, t_emb)
        for resnet, attn in zip(self.resnets[1:], self.attentions):
            x = attn(x, cond)
            x = resnet(x, t_emb)
        
        return x


class AttnUpBlock1D(nn.Module):
    def __init__(self, input_dim, output_dim, cond_dim,
                 num_layers=1, add_upsample=True,
                 kernel_size=3, num_groups=8,
                 num_heads=8):
        super().__init__()

        resnets = []
        attentions = []
        for i in range(num_layers):
            dim_in = input_dim if i == 0 else output_dim
            resnets.append(
                ResNetBlock1D(
                    dim_in, output_dim,
                    cond_dim=cond_dim, 
                    kernel_size=kernel_size,
                    num_groups=num_groups,
                ),
            )
            attentions.append(
                TransformerBlock1D(
                    dim=output_dim,
                    cond_dim=cond_dim,
                    num_heads=num_heads,
                )
            )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        self.upsampler = Upsample1d(output_dim) if add_upsample else nn.Identity()

    def forward(self, x, t_emb, cond):
        for resnet, attn in zip(self.resnets, self.attentions):
            x = resnet(x, t_emb)
            x = attn(x, cond)
        
        return self.upsampler(x), x


class UpBlock1D(nn.Module):
    def __init__(self, input_dim, output_dim, timestep_dim,
                 num_layers=1, add_upsample=True,
                 kernel_size=3, num_groups=8):
        super().__init__()

        resnets = []
        for i in range(num_layers):
            dim_in = input_dim if i == 0 else output_dim
            resnets.append(
                ResNetBlock1D(
                    dim_in, output_dim,
                    cond_dim=timestep_dim, 
                    kernel_size=kernel_size,
                    num_groups=num_groups,
                ),
            )
        self.resnets = nn.ModuleList(resnets)

        self.upsampler = Upsample1d(output_dim) if add_upsample else nn.Identity()

    def forward(self, x, t_emb, cond=None):
        for resnet in self.resnets:
            x = resnet(x, t_emb)
        
        return self.upsampler(x), x
