import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        # Permute (N, C, H, W) -> (N, H, W, C), apply LayerNorm, then permute back
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=2):
        super(ChannelAttention, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = self.pool(x)
        attn = self.conv1(attn)
        attn = self.relu(attn)
        attn = self.conv2(attn)
        attn = self.sigmoid(attn)
        return x * attn

class BaselineBlock(nn.Module):
    def __init__(self, in_channels):
        super(BaselineBlock, self).__init__()
        self.norm1 = LayerNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.dconv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.gelu = nn.GELU()
        self.ca = ChannelAttention(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.norm2 = LayerNorm2d(in_channels)
        self.conv3 = nn.Conv2d(in_channels, 2 * in_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(2 * in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        residual = x

        out = self.norm1(x)
        out = self.conv1(out)
        out = self.dconv(out)
        out = self.gelu(out)
        out = self.ca(out)
        out = self.conv2(out)
        out += residual

        residual = out

        out = self.norm2(out)
        out = self.conv3(out)
        out = self.gelu(out)
        out = self.conv4(out)
        out += residual

        return out

class BaselineModel(nn.Module):
    def __init__(self, n_channels=3, width=32):
        super(BaselineModel, self).__init__()
        self.init_conv = nn.Conv2d(n_channels, width, kernel_size=3, padding=1)

        # Encoder (Downsampling path)
        self.enc1 = self._make_stage(width, 4)
        self.down1 = nn.Conv2d(width, width, kernel_size=3, stride=2, padding=1)

        self.enc2 = self._make_stage(width, 4)
        self.down2 = nn.Conv2d(width, width, kernel_size=3, stride=2, padding=1)

        self.enc3 = self._make_stage(width, 4)
        self.down3 = nn.Conv2d(width, width, kernel_size=3, stride=2, padding=1)

        self.enc4 = self._make_stage(width, 4)
        self.down4 = nn.Conv2d(width, width, kernel_size=3, stride=2, padding=1)

        # Bottleneck
        self.bottleneck = self._make_stage(width, 4)

        # Decoder (Upsampling path)
        self.up4 = self._upsample_layer(width)
        self.dec4 = self._make_stage(width, 4)

        self.up3 = self._upsample_layer(width)
        self.dec3 = self._make_stage(width, 4)

        self.up2 = self._upsample_layer(width)
        self.dec2 = self._make_stage(width, 4)

        self.up1 = self._upsample_layer(width)
        self.dec1 = self._make_stage(width, 4)

        # Final output layer
        self.final_conv = nn.Conv2d(width, n_channels, kernel_size=3, padding=1)

    def _make_stage(self, channels, num_blocks):
        """Helper function to create multiple BaselineBlocks."""
        return nn.Sequential(*[BaselineBlock(channels) for _ in range(num_blocks)])

    def _upsample_layer(self, channels):
        """Upsample using pointwise convolution followed by pixel shuffle."""
        return nn.Sequential(
            nn.Conv2d(channels, channels * 4, kernel_size=1),
            nn.PixelShuffle(2)
        )

    def forward(self, x):

        input = x
        x = self.init_conv(x)

        # Encoder
        e1 = self.enc1(x)
        x = self.down1(e1)

        e2 = self.enc2(x)
        x = self.down2(e2)

        e3 = self.enc3(x)
        x = self.down3(e3)

        e4 = self.enc4(x)
        x = self.down4(e4)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.up4(x) + e4
        x = self.dec4(x)

        x = self.up3(x) + e3
        x = self.dec3(x)

        x = self.up2(x) + e2
        x = self.dec2(x)

        x = self.up1(x) + e1
        x = self.dec1(x)

        # Final output
        x = self.final_conv(x)

        # Residual Learning Approach
        x = x + input
        
        return x
