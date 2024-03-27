import torch
from torch import nn
from torch.nn import functional as F
from .decoder import VAE_AttentionBlock, VAE_ResidualBlock


class VAE_Encoder(nn.Sequential):

    def __init__(self):
        super().__init__(
            # (batch_num, num_channels, height, width) -> (batch_num, 128, height, width)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            # convolutions and normalization layers
            # batch_num 128, height, width -> batch_num, 128, height, width
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),
            # batch_num, 128, height, width -> batch_num, 128, height/2, width/2
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            # batch_num, 128, height/2, width/2 -> batch_num, 256, height/2, width/2
            VAE_ResidualBlock(128, 256),
            # batch_num, 256, height/2, width/2 -> batch_num, 256, height/2, width/2
            VAE_ResidualBlock(256, 256),
            # batch_num, 256, height/2, width/2 -> batch_num, 256, height/4, width/4
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            # batch_num, 256, height/4, width/4 -> batch_num, 512, height/4, width/4
            VAE_ResidualBlock(256, 512),
            # batch_num, 512, height/4, width/4 -> batch_num, 512, height/4, width/4
            VAE_ResidualBlock(512, 512),
            # batch_num, 512, height/4, width/4 -> batch_num, 512, height/8, width/8
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            # batch_num, 512, height/8, width/8 -> batch_num, 512, height/8, width/8
            VAE_AttentionBlock(512),
            # batch_num, 512, height/8, width/8 -> batch_num, 512, height/8, width/8
            VAE_ResidualBlock(512, 512),
            # batch_num, 512, height/8, width/8 -> batch_num, 512, height/8, width/8
            nn.GroupNorm(32, 512),
            nn.SiLU(),
            # batch_num, 512, height/8, width/8 -> batch_num, 8, height/8, width/8
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            # batch_num, 8, height/8, width/8
            nn.Conv2d(8, 8, kernel_size=1, padding=0),
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x: batch_num, channel, height, width
        # noise: batch_num, out_channels, height/8, width/8

        for module in self:
            if getattr(module, "stride", None) == (2, 2):
                # apply asymmetrical padding
                # padding_left, padding_right, padding_top, padding_bottom
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)

        # batch_size, 8, height/8, width/8 -> two tensors of shape
        # bacth_size, 4, height/8, width/8
        mean, log_variance = torch.chunk(x, 2, dim=1)

        log_variance = torch.clamp(log_variance, -30, 20)

        variance = log_variance.exp()

        stddev = variance.sqrt()

        # Z~(0, 1) -> X~N(mean, variance)?
        # X = mean + stddev * Z

        x = mean + stddev * noise

        # Scale the output by a constant, why this constant idk
        x *= 0.18125

        return x
