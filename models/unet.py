# models/unet.py

import torch
import torch.nn as nn
from .layers.conditional_layers import ConditionalAdaIN

class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16, spatial_kernel=7):
        super(CBAM, self).__init__()
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=spatial_kernel, padding=spatial_kernel//2),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 通道注意力
        channel_att = self.channel_attention(x)
        x = x * channel_att
        
        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))
        x = x * spatial_att
        
        return x

class DoubleConv(nn.Module):
    """(卷积 => [BN] => ReLU) * 2"""
    # (保持不变)
    def __init__(self, in_channels, out_channels, mid_channels=None, use_cbam=True):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.use_cbam = use_cbam
        if use_cbam:
            self.cbam=CBAM(out_channels)

    def forward(self, x):
        x = self.double_conv(x)
        if self.use_cbam:
            x = self.cbam(x)
        return x


class ConditionalDoubleConv(nn.Module):
    """(卷积 => [BN] => ReLU) * 2 => AdaIN"""

    def __init__(self, in_channels, out_channels, condition_dim, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.adain = ConditionalAdaIN(condition_dim, out_channels)

    def forward(self, x, c):
        x = self.double_conv(x)
        x = self.adain(x, c)
        return x


class Down(nn.Module):
    """下采样模块：最大池化 + (条件)双卷积"""

    def __init__(self, in_channels, out_channels, condition_dim=None, use_cbam=True):
        super().__init__()
        use_conditional = condition_dim is not None
        
        if use_conditional:
            conv_block = ConditionalDoubleConv(in_channels, out_channels, condition_dim)
        else:
            conv_block = DoubleConv(in_channels, out_channels, use_cbam=use_cbam)

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            conv_block
        )

    def forward(self, x, c=None):
        if isinstance(self.maxpool_conv[1], ConditionalDoubleConv):
            return self.maxpool_conv[0](x), self.maxpool_conv[1](self.maxpool_conv[0](x), c)
        else:
            # 在非条件模式下，我们返回池化后的结果和卷积后的结果，以保持输出格式一致
            pooled = self.maxpool_conv[0](x)
            return pooled, self.maxpool_conv[1](pooled)


class Up(nn.Module):
    """上采样模块：转置卷积 + 跳跃连接 + (条件)双卷积"""

    def __init__(self, in_channels, out_channels, condition_dim=None, bilinear=True, use_cbam=True):
        super().__init__()
        use_conditional = condition_dim is not None

        # 如果使用双线性插值，则用它来上采样
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            if use_conditional:
                self.conv = ConditionalDoubleConv(in_channels, out_channels, condition_dim, in_channels // 2)
            else:
                self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, use_cbam=use_cbam)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            if use_conditional:
                self.conv = ConditionalDoubleConv(in_channels, out_channels, condition_dim)
            else:
                self.conv = DoubleConv(in_channels, out_channels, use_cbam=use_cbam)

    def forward(self, x1, x2, c=None):
        # x1是来自上采样路径的特征图，x2是来自编码器路径的跳跃连接特征图
        x1 = self.up(x1)
        
        # 拼接跳跃连接的特征
        x = torch.cat([x2, x1], dim=1)
        
        if isinstance(self.conv, ConditionalDoubleConv):
            return self.conv(x, c)
        else:
            return self.conv(x)

class ConditionalUNet(nn.Module):
    """
    条件化的U-Net模型。
    通过AdaIN层将条件向量注入到网络的每个卷积块中。
    """
    def __init__(self, in_channels, out_channels, condition_dim, base_channels=64, bilinear=True):
        """
        Args:
            in_channels (int): 输入图像通道数 (例如, 3 for RGB)。
            out_channels (int): 输出图像通道数 (例如, 3 for RGB)。
            condition_dim (int): 条件向量的维度。
            base_channels (int): 第一层卷积的通道数。
            bilinear (bool): 是否使用双线性插值进行上采样。
        """
        super(ConditionalUNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.condition_dim = condition_dim
        self.base_channels = base_channels
        self.bilinear = bilinear
        
        # 编码器（下采样路径）
        self.inc = ConditionalDoubleConv(in_channels, base_channels, condition_dim)
        self.down1 = Down(base_channels, base_channels * 2, condition_dim)
        self.down2 = Down(base_channels * 2, base_channels * 4, condition_dim)
        self.down3 = Down(base_channels * 4, base_channels * 8, condition_dim)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor, condition_dim)

        # 解码器（上采样路径）
        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, condition_dim, bilinear)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, condition_dim, bilinear)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, condition_dim, bilinear)
        self.up4 = Up(base_channels * 2, base_channels, condition_dim, bilinear)
        
        # 输出层
        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x, c):
        # 编码
        x1 = self.inc(x, c)
        _, x2 = self.down1(x1, c)
        _, x3 = self.down2(x2, c)
        _, x4 = self.down3(x3, c)
        _, x5 = self.down4(x4, c)
        
        # 解码
        x = self.up1(x5, x4, c)
        x = self.up2(x, x3, c)
        x = self.up3(x, x2, c)
        x = self.up4(x, x1, c)
        
        # 输出
        logits = self.outc(x)
        return logits

# --- 为了兼容旧的测试，保留原始的UNet类 ---
class UNet(nn.Module):
    """
    基础的U-Net模型，作为基线。
    不包含任何条件注入逻辑。
    """
    def __init__(self, in_channels, out_channels, base_channels=64, bilinear=True):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.bilinear = bilinear
        
        # 编码器（下采样路径）
        self.inc = DoubleConv(in_channels, base_channels, use_cbam=True)
        self.down1_block = Down(base_channels, base_channels * 2 ,use_cbam=True)
        self.down2_block = Down(base_channels * 2, base_channels * 4, use_cbam=True)
        self.down3_block = Down(base_channels * 4, base_channels * 8, use_cbam=True)
        factor = 2 if bilinear else 1
        self.down4_block = Down(base_channels * 8, base_channels * 16 // factor, use_cbam=True)

        # 解码器（上采样路径）
        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear=bilinear, use_cbam=True)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear=bilinear, use_cbam=True)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear=bilinear, use_cbam=True)
        self.up4 = Up(base_channels * 2, base_channels, bilinear=bilinear, use_cbam=True)
        
        # 输出层
        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # 编码
        x1 = self.inc(x)
        _, x2 = self.down1_block(x1)
        _, x3 = self.down2_block(x2)
        _, x4 = self.down3_block(x3)
        _, x5 = self.down4_block(x4)
        
        # 解码
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # 输出
        logits = self.outc(x)
        return logits
