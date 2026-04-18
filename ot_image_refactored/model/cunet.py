import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class CondINorm(nn.Module):
    def __init__(self, in_channels, z_channels, eps=1e-5):
        super(CondINorm, self).__init__()
        self.eps = eps
        self.shift_conv = nn.Sequential(
            nn.Conv2d(z_channels, in_channels, kernel_size=1, padding=0, bias=True),
            nn.ReLU(True)
        )
        self.scale_conv = nn.Sequential(
            nn.Conv2d(z_channels, in_channels, kernel_size=1, padding=0, bias=True),
            nn.ReLU(True)
        )

    def forward(self, x, z):
        shift = self.shift_conv.forward(z)
        scale = self.scale_conv.forward(z)
        size = x.size()
        x_reshaped = x.view(size[0], size[1], size[2]*size[3])
        mean = x_reshaped.mean(2, keepdim=True)
        var = x_reshaped.var(2, keepdim=True)
        std =  torch.rsqrt(var + self.eps)
        norm_features = ((x_reshaped - mean) * std).view(*size)
        output = norm_features * scale + shift
        return output
    
class CondDoubleConv(nn.Module):
    """(convolution => [CIN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, z_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.norm1 = CondINorm(mid_channels, z_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = CondINorm(out_channels, z_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, z):
        x = self.conv1(x)
        x = self.norm1(x, z)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.norm2(x, z)
        x = self.relu2(x)
        return x

class CondUp(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, z_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = CondDoubleConv(in_channels, out_channels, z_channels, in_channels // 2)

    def forward(self, x1, x2, z):
        x1 = self.up(x1)
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x, z)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class CUNet(nn.Module):
    def __init__(self, n_channels, n_classes, z_channels, base_factor=32):
        super(CUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.z_channels = z_channels
        self.base_factor = base_factor

        self.inc = DoubleConv(n_channels, base_factor)
        self.down1 = Down(base_factor, 2 * base_factor)
        self.down2 = Down(2 * base_factor, 4 * base_factor)
        self.down3 = Down(4 * base_factor, 8 * base_factor)
        factor = 2
        self.down4 = Down(8 * base_factor, 16 * base_factor // factor)
        
        self.adain1 = CondINorm(16 * base_factor // factor, z_channels)
        self.up1 = Up(16 * base_factor, 8 * base_factor // factor)
        self.adain2 = CondINorm(8 * base_factor // factor, z_channels)
        self.up2 = Up(8 * base_factor, 4 * base_factor // factor)
        self.adain3 = CondINorm(4 * base_factor // factor, z_channels)
        self.up3 = Up(4 * base_factor, 2 * base_factor // factor)
        self.adain4 = CondINorm(2 * base_factor // factor, z_channels)
        self.up4 = Up(2 * base_factor, base_factor)
        
        self.outc = OutConv(base_factor, n_classes)

    def forward(self, x, z):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.adain1(x5, z)
        x = self.up1(x, x4)
        x = self.adain2(x, z)
        x = self.up2(x, x3)
        x = self.adain3(x, z)
        x = self.up3(x, x2)
        x = self.adain4(x, z)
        x = self.up4(x, x1)
        
        logits = self.outc(x)
        return logits


class TextEarlyInjectionBlock(nn.Module):
    def __init__(self, in_channels, text_dim, hidden_dim=128, num_heads=4):
        super().__init__()
        # 1. 升维映射：丰富特征容量，方便与文本匹配
        self.img_proj_in = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(hidden_dim)
        
        # 2. 交叉注意力机制 (Image=Query, Text=Key&Value)
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # 3. 降维还原：退回原始输入通道数
        self.img_proj_out = nn.Conv2d(hidden_dim, in_channels, kernel_size=3, padding=1)

    def forward(self, x, text_tokens):
        # x 形状: [Batch, Channels(4), Height(32), Width(32)]
        B, C, H, W = x.shape
        
        # --- 图像侧 (Query) ---
        h = self.img_proj_in(x) 
        h_flat = h.view(B, h.size(1), -1).transpose(1, 2) # [B, H*W, hidden_dim]
        h_flat = self.norm(h_flat)
        
        # --- 文本侧 (Key/Value) ---
        # text_tokens: [B, SeqLen, text_dim] -> [B, SeqLen, hidden_dim]
        k_v = self.text_proj(text_tokens) 
        
        # --- 注意力交互 ---
        attn_out, _ = self.cross_attn(query=h_flat, key=k_v, value=k_v)
        
        # --- 残差连接与空间恢复 ---
        h_fused = h_flat + attn_out
        h_fused = h_fused.transpose(1, 2).view(B, -1, H, W) # [B, hidden_dim, H, W]
        
        # 加回原始输入，保持完美的通道连续性
        out = x + self.img_proj_out(h_fused)
        return out


class TextGuidedCUNet(nn.Module):
    def __init__(self, in_channels, n_classes, z_channels, text_dim,
                 base_factor=32, attn_hidden_dim=256, num_heads=4,
                 latent_size=32):          # ← 新增参数
        super().__init__()

        # 输入端：cross attention
        self.early_attn = TextEarlyInjectionBlock(
            in_channels=in_channels,
            text_dim=text_dim,
            hidden_dim=attn_hidden_dim,
            num_heads=num_heads
        )

        # bottleneck：文本全局向量叠加到时间嵌入
        self.text_bottleneck_proj = nn.Sequential(
            nn.Linear(text_dim, z_channels),
            nn.SiLU(),
            nn.Linear(z_channels, z_channels)
        )

        # 方向投影头：隐空间位移向量 → 文本语义空间
        # 输入维度 = C * H * W，例如 4*32*32 = 4096
        latent_dim = in_channels * latent_size * latent_size
        self.dir_proj = nn.Sequential(
            nn.Linear(latent_dim, z_channels),
            nn.SiLU(),
            nn.Linear(z_channels, text_dim)   # 对齐到 768
        )

        # 原版核心引擎
        self.unet = CUNet(
            n_channels=in_channels,
            n_classes=n_classes,
            z_channels=z_channels,
            base_factor=base_factor
        )

    def forward(self, x, t, context=None):
        if context is not None:
            # 输入端注入
            x = self.early_attn(x, context)
            # bottleneck 注入：文本全局向量叠加到时间嵌入
            text_global = self.text_bottleneck_proj(
                context.mean(dim=1)        # [B, 768] → [B, z_channels]
            )[:, :, None, None]            # → [B, z_channels, 1, 1]
            t = t + text_global            # t 是 [B, z_channels, 1, 1]
        return self.unet(x, t)