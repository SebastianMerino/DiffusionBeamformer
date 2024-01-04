import torch
import torch.nn as nn
import math


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(1000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class DoubleConv(nn.Module):
    """ Includes Convolutional layer, batch normalization and ReLU twice """
    def __init__(self, in_channels, out_channels, emb_dim=32, residual=False):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(emb_dim, out_channels),
            nn.ReLU()
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.residual = residual

    def forward(self, x0, time_emb):
        x = self.conv1(x0)
        time_emb = self.time_mlp(time_emb)
        time_emb = time_emb[(..., ) + (None, ) * 2]
        x = x + time_emb
        x = self.conv2(x)
        if self.residual:
            # Apply a 1x1 convolutional layer to match dimensions before adding residual connection
            x = self.shortcut(x0) + x
            x /= math.sqrt(2)
        return x    

class UNETv8(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512], emb_dim=32
        ):
        super(UNETv8, self).__init__()
        self.initial_block = DoubleConv(in_channels, features[0])
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU()
        )

        # Down part of UNET
        self.downBlocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        for feature in features:
            self.downBlocks.append(DoubleConv(feature, feature * 2))

        # Up part of UNET
        self.upBlocks = nn.ModuleList()
        self.upConvs = nn.ModuleList()
        for feature in reversed(features):
            self.upConvs.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.upBlocks.append(DoubleConv(feature * 2, feature))

        self.final_block = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x, t):
        time_emb = self.time_mlp(t)
        x = self.initial_block(x, time_emb)

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
    

class UNETv9(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512], emb_dim=32
        ):
        super(UNETv9, self).__init__()
        self.initial_block = DoubleConv(in_channels, features[0], residual=True)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU()
        )

        # Down part of UNET
        self.downBlocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        for feature in features:
            self.downBlocks.append(DoubleConv(feature, feature * 2, residual=True))

        # Up part of UNET
        self.upBlocks = nn.ModuleList()
        self.upConvs = nn.ModuleList()
        for feature in reversed(features):
            self.upConvs.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.upBlocks.append(DoubleConv(feature * 2, feature, residual=True))

        self.final_block = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x, t):
        time_emb = self.time_mlp(t)
        x = self.initial_block(x, time_emb)

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