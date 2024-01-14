import math
import torch
from torch import nn
from RRDB import RRDBNet

class UNETv11(nn.Module):
    def __init__(
            self, in_channels=2, out_channels=1, features=[64, 128, 256, 512], emb_dim=32
        ):
        super(UNETv11, self).__init__()
        # Encodes noisy Bmode to feature map of first layer of UNET
        self.initial_block_F = nn.Conv2d(in_channels=1, out_channels=features[0], kernel_size=3, padding=1)

        # Encodes IQ to feature map of first layer of UNET
        self.initial_block_G = RRDBNet(in_nc=in_channels, nb = 3, out_nc=features[0])


        self.time_mlp = nn.Sequential(
            PositionalEncoding(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
        )

        # Down part of UNET
        self.downBlocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        for feature in features:
            self.downBlocks.append(ResnetBlock(feature, feature * 2, emb_dim))

        # Up part of UNET
        self.upBlocks = nn.ModuleList()
        self.upConvs = nn.ModuleList()
        for feature in reversed(features):
            self.upConvs.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.upBlocks.append(ResnetBlock(feature * 2, feature, emb_dim))

        self.final_block = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x, y, t):
        # x: IQ image
        # y: Noisy Bmode
        time_emb = self.time_mlp(t)
        y = self.initial_block_F(y)
        x = self.initial_block_G(x)

        x += y
        # Convolutional layers and max-pooling
        skip_connections = []
        for block in self.downBlocks:
            skip_connections.append(x)
            x = self.pool(x)
            x = block(x, time_emb)

        # Convolutional layers and up-sampling
        skip_connections = skip_connections[::-1]  # Reversing list
        for idx in range(len(self.upBlocks)):
            x = self.upConvs[idx](x)  # UpConvolution
            concat_skip = torch.cat((skip_connections[idx], x), dim=1)
            x = self.upBlocks[idx](concat_skip, time_emb)  # Double convs

        return self.final_block(x)



# PositionalEncoding Source： https://github.com/lmnt-com/wavegrad/blob/master/src/wavegrad/model.py
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype, device=noise_level.device) / count
        encoding = noise_level.unsqueeze(1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureWiseAffine, self).__init__()
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels*2)
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        gamma, beta = self.noise_func(noise_embed).view(batch, -1, 1, 1).chunk(2, dim=1)
        x = (1 + gamma) * x + beta
        return x

class Block(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3,1,1, bias=False),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=32, residual = False):
        super().__init__()
        self.noise_func = FeatureWiseAffine(noise_level_emb_dim, dim_out)

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        if residual:
            self.res_conv = nn.Conv2d(dim, dim_out, 1) 
        else:
            self.res_conv = None

    def forward(self, x, time_emb):
        h = self.block1(x)
        h = self.noise_func(h, time_emb)
        h = self.block2(h)
        if self.res_conv:
            h += self.res_conv(x)
        return h