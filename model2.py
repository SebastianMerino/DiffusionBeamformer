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


class EmbedTimeLayer(nn.Module):
    """
    This class defines a generic one layer feed-forward neural network for embedding time
    to an embedding space of dimensionality emb_dim.
    """
    def __init__(self, emb_dim, out_dim):
        super(EmbedTimeLayer, self).__init__()
        self.emb_dim = emb_dim
        self.out_dim = out_dim

        # define the layers for the network
        layers = [
            SinusoidalPositionEmbeddings(emb_dim),
            nn.Linear(emb_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
        ]

        # create a PyTorch sequential model consisting of the defined layers
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # flatten the input tensor
        x = x.view(-1, 1)
        # apply the model layers to the flattened tensor
        x = self.model(x)
        return x.view(-1, self.out_dim, 1, 1)


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
        self.time_mlp = EmbedTimeLayer(emb_dim, out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.residual = residual

    def forward(self, x0, t):
        x = self.conv1(x0)
        x = x + self.time_mlp(t)
        x = self.conv2(x)
        if self.residual:
            # Apply a 1x1 convolutional layer to match dimensions before adding residual connection
            x = self.shortcut(x0) + x
            x /= math.sqrt(2)
        return x    

class UNETv6(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(UNETv6, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upConvs = nn.ModuleList()

        self.initial_block = DoubleConv(in_channels, features[0])

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(feature, feature * 2))

        # Up part of UNET
        for feature in reversed(features):
            self.upConvs.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))

        self.final_block = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x, t):

        x = self.initial_block(x, t)

        # Convolutional layers and max-pooling
        skip_connections = []
        for down in self.downs:
            skip_connections.append(x)
            x = self.pool(x)
            x = down(x, t)

        # Convolutional layers and up-sampling
        skip_connections = skip_connections[::-1]  # Reversing list
        for idx in range(len(self.ups)):
            x = self.upConvs[idx](x)  # UpConvolution
            concat_skip = torch.cat((skip_connections[idx], x), dim=1)
            x = self.ups[idx](concat_skip, t)  # Double convs

        return self.final_block(x)
    

class UNETv7(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(UNETv7, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upConvs = nn.ModuleList()

        self.initial_block = DoubleConv(in_channels, features[0], residual=True)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(feature, feature * 2, residual=True))

        # Up part of UNET
        for feature in reversed(features):
            self.upConvs.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature, residual=True))

        self.final_block = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x, t):

        x = self.initial_block(x, t)

        # Convolutional layers and max-pooling
        skip_connections = []
        for down in self.downs:
            skip_connections.append(x)
            x = self.pool(x)
            x = down(x, t)

        # Convolutional layers and up-sampling
        skip_connections = skip_connections[::-1]  # Reversing list
        for idx in range(len(self.ups)):
            x = self.upConvs[idx](x)  # UpConvolution
            concat_skip = torch.cat((skip_connections[idx], x), dim=1)
            x = self.ups[idx](concat_skip, t)  # Double convs

        return self.final_block(x)