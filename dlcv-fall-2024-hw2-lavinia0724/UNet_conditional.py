# Reference: https://www.kaggle.com/code/mojtabanafez1400/deffusion-conditional-deffusion-fash-mnist-ipynb/code
import torch
import torch.nn as nn
import torch.nn.functional as F

def one_param(m):
    "get model first parameter"
    return next(iter(m.parameters()))


class Attention(nn.Module):
    def __init__(self, channels):
        super(Attention, self).__init__()
        self.channels = channels        
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        size = x.shape[-1]
        x = x.view(-1, self.channels, size * size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, size, size)


class ConvResNet(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvResNet(in_channels, in_channels, residual=True),
            ConvResNet(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            ConvResNet(in_channels, in_channels, residual=True),
            ConvResNet(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    # modify by GPT4o
    def forward(self, x, skip_x, t):
        # Upsample the tensor
        x = self.up(x)
        
        # Check and pad if the shapes do not match
        if x.shape[-2:] != skip_x.shape[-2:]:
            diffY = skip_x.size(2) - x.size(2)
            diffX = skip_x.size(3) - x.size(3)
            x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        
        # Concatenate skip connection and upsampled tensor
        x = torch.cat([skip_x, x], dim=1)
        
        # Apply convolution and embedding
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet(nn.Module):
    def __init__(self, c_in=1, c_out=1, time_dim=256, remove_deep_conv=False):
        super().__init__()
        self.time_dim = time_dim
        self.remove_deep_conv = remove_deep_conv
        self.inc = ConvResNet(c_in, 16)
        self.down1 = Down(16, 32)
        self.sa1 = Attention(32)
        self.down2 = Down(32, 64)
        self.sa2 = Attention(64)
        self.down3 = Down(64, 64)
        self.sa3 = Attention(64)


        if remove_deep_conv:
            self.bot1 = ConvResNet(64, 64)
            self.bot3 = ConvResNet(64, 64)
        else:
            self.bot1 = ConvResNet(64, 128)
            self.bot2 = ConvResNet(128, 128)
            self.bot3 = ConvResNet(128, 64)

        self.up1 = Up(128, 32)
        self.sa4 = Attention(32)
        self.up2 = Up(64, 16)
        self.sa5 = Attention(16)
        self.up3 = Up(32, 16)
        self.sa6 = Attention(16)
        self.outc = nn.Conv2d(16, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=one_param(self).device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def unet_forwad(self, x, t):
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        if not self.remove_deep_conv:
            x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output
    
    def forward(self, x, t):
        t = t.unsqueeze(-1)
        t = self.pos_encoding(t, self.time_dim)
        return self.unet_forwad(x, t)
    
# modify for CLASSIFIER-FREE DIFFUSION GUIDANCE
class UNet_conditional(UNet):
    def __init__(self, c_in=1, c_out=1, time_dim=256, num_classes=None, **kwargs):
        super().__init__(c_in, c_out, time_dim, **kwargs)
        self.num_classes = num_classes
        
        # 調整嵌入層的輸出與 time_dim 一致
        self.label_emb = nn.Embedding(num_classes, time_dim)  # 確保與 time_dim 一致
        self.dataset_label_emb = nn.Embedding(2, time_dim)  # 假設有兩個數據集

    def forward(self, x, t, labels=None, dataset_labels=None):
        # Positional encoding for time step
        t = t.unsqueeze(-1)
        t = self.pos_encoding(t, self.time_dim)  # shape: (batch_size, time_dim)

        # Check the size of labels and dataset_labels
        if labels is not None:
            # Ensure label_emb has the same size as t
            label_emb = self.label_emb(labels)  # shape: (batch_size, time_dim)
            t = t + label_emb  # Add label embedding to t
        if dataset_labels is not None:
            # Ensure dataset_label_emb has the same size as t
            dataset_label_emb = self.dataset_label_emb(dataset_labels)  # shape: (batch_size, time_dim)
            t = t + dataset_label_emb  # Add dataset label embedding to t

        # Pass through the U-Net
        return self.unet_forwad(x, t)
