import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        # Convert to channels_last format for efficiency
        x = x.to(memory_format=torch.channels_last)
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class SimpleGate(nn.Module):
    def __init__(self):
        super(SimpleGate, self).__init__()

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimplifiedChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(SimplifiedChannelAttention, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)  # No bias needed

    def forward(self, x):
        attn = self.pool(x)
        attn = self.conv(attn)
        return x * attn


class NAFNetBlock(nn.Module):
    def __init__(self, in_channels):
        super(NAFNetBlock, self).__init__()
        self.norm1 = LayerNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 2 * in_channels, kernel_size=1, bias=False)
        self.dconv = nn.Conv2d(2 * in_channels, 2 * in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.simplegate = SimpleGate()
        self.sca = SimplifiedChannelAttention(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)

        self.norm2 = LayerNorm2d(in_channels)
        self.conv3 = nn.Conv2d(in_channels, 2 * in_channels, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        residual = x

        out = self.norm1(x)
        out = self.conv1(out)
        out = self.dconv(out)
        out = self.simplegate(out)
        out = self.sca(out)
        out = self.conv2(out)
        out += residual

        residual = out

        out = self.norm2(out)
        out = self.conv3(out)
        out = self.simplegate(out)
        out = self.conv4(out)
        out += residual

        return out

class NAFNetModel(nn.Module):
    def __init__(self, n_channels=3, width=32):
        super(NAFNetModel, self).__init__()
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
        return nn.Sequential(*[NAFNetBlock(channels) for _ in range(num_blocks)])

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