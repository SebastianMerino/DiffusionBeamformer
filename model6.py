import math
import torch
from nn import *
from RRDB import RRDBNet

class UNETv12(nn.Module):
    def __init__(
            self, 
            in_channels=2, 
            out_channels=1, 
            features=[64, 128, 256, 512], 
            emb_dim = 256, 
            rrdb_blocks = 3,
        ):
        super(UNETv12, self).__init__()
        # Encodes noisy Bmode to feature map of first layer of UNET
        self.initial_block_F = nn.Conv2d(in_channels=1, out_channels=features[0], kernel_size=3, padding=1)

        # Encodes IQ to feature map of first layer of UNET
        self.initial_block_G = RRDBNet(in_nc=in_channels, nb = rrdb_blocks, out_nc=features[0])

        self.time_mlp = nn.Sequential(
            PositionalEncoding(features[0]),
            linear(features[0], emb_dim),
            SiLU(),
            linear(emb_dim, emb_dim),
        )

        # Down part of UNET
        self.downBlocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        for feature in features:
            self.downBlocks.append(
                ResBlock(
                    channels = feature,
                    emb_channels = emb_dim,
                    dropout = 0,
                    out_channels = feature*2,
                    use_conv = False,
                    use_scale_shift_norm = True,
                    dims = 2,
                    use_checkpoint = False,
                ))

        # Up part of UNET
        self.upBlocks = nn.ModuleList()
        self.upConvs = nn.ModuleList()
        for feature in reversed(features):
            self.upConvs.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.upBlocks.append(
                ResBlock(
                    channels = feature * 2,
                    emb_channels = emb_dim,
                    dropout = 0,
                    out_channels = feature,
                    use_conv = False,
                    use_scale_shift_norm = True,
                    dims = 2,
                    use_checkpoint = False,
                ))

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


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h
