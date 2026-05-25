"""
小波条件模块
- 纯PyTorch实现，不依赖pywt，支持GPU
- 对热成像做两级Haar小波分解
- 输出两套多尺度频率特征用于注入UNet Encoder
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class HaarWavelet2D(nn.Module):
    """
    可微分Haar小波分解
    输出四个子带：LL(低频), LH(横向边缘), HL(纵向边缘), HH(对角纹理)
    """
    def __init__(self):
        super().__init__()
        ll = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32) / 4.0
        lh = torch.tensor([[-1, -1], [1, 1]], dtype=torch.float32) / 4.0
        hl = torch.tensor([[-1, 1], [-1, 1]], dtype=torch.float32) / 4.0
        hh = torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32) / 4.0
        # 注册为buffer：随模型保存，不参与梯度
        self.register_buffer('ll_filter', ll.view(1, 1, 2, 2))
        self.register_buffer('lh_filter', lh.view(1, 1, 2, 2))
        self.register_buffer('hl_filter', hl.view(1, 1, 2, 2))
        self.register_buffer('hh_filter', hh.view(1, 1, 2, 2))

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        x_flat = x.view(B * C, 1, H, W)
        ll = F.conv2d(x_flat, self.ll_filter, stride=2).view(B, C, H//2, W//2)
        lh = F.conv2d(x_flat, self.lh_filter, stride=2).view(B, C, H//2, W//2)
        hl = F.conv2d(x_flat, self.hl_filter, stride=2).view(B, C, H//2, W//2)
        hh = F.conv2d(x_flat, self.hh_filter, stride=2).view(B, C, H//2, W//2)
        return ll, lh, hl, hh


class WaveletConditionModule(nn.Module):
    """
    对热成像做两级小波分解，输出两套频率特征：
      feat_64: [B, out_ch_64, 64, 64]  →  注入 Encoder 64×64 层
      feat_32: [B, out_ch_32, 32, 32]  →  注入 Encoder 32×32 层
    """
    def __init__(self, in_ch=3, out_ch_64=64, out_ch_32=64):
        super().__init__()
        self.dwt = HaarWavelet2D()
        hf_ch = in_ch * 3  # LH+HL+HH 拼接 = 9通道

        # Level1 高频 → 64通道
        self.proj_64 = nn.Sequential(
            nn.Conv2d(hf_ch, out_ch_64 * 2, 1, bias=False),
            nn.GroupNorm(8, out_ch_64 * 2),
            nn.SiLU(),
            nn.Conv2d(out_ch_64 * 2, out_ch_64, 1, bias=False),
            nn.GroupNorm(8, out_ch_64),
            nn.SiLU(),
        )
        # Level2 低频+高频 → 64通道
        self.proj_32 = nn.Sequential(
            nn.Conv2d(in_ch + hf_ch, out_ch_32 * 2, 1, bias=False),
            nn.GroupNorm(8, out_ch_32 * 2),
            nn.SiLU(),
            nn.Conv2d(out_ch_32 * 2, out_ch_32, 1, bias=False),
            nn.GroupNorm(8, out_ch_32),
            nn.SiLU(),
        )

    def convert_to_fp16(self):
        self.proj_64.half()
        self.proj_32.half()

    def convert_to_fp32(self):
        self.proj_64.float()
        self.proj_32.float()

    def forward(self, thermal):
        dtype = thermal.dtype
        x = thermal.float()  # 小波分解用fp32保证精度

        # Level1: 128→64
        ll1, lh1, hl1, hh1 = self.dwt(x)
        hf1 = torch.cat([lh1, hl1, hh1], dim=1)  # [B, 9, 64, 64]

        # Level2: 64→32
        ll2, lh2, hl2, hh2 = self.dwt(ll1)
        hf2 = torch.cat([lh2, hl2, hh2], dim=1)  # [B, 9, 32, 32]

        feat_64 = self.proj_64(hf1.to(dtype))
        feat_32 = self.proj_32(torch.cat([ll2, hf2], dim=1).to(dtype))
        return feat_64, feat_32


class WaveletChannelAttention(nn.Module):
    """
    轻量级通道注意力：用小波特征门控UNet特征
    不做全局token matching（ArcFace已经做了），只做通道维度的频率选择
    """
    def __init__(self, unet_ch, wavelet_ch, reduction=4):
        super().__init__()
        self.align = nn.Conv2d(wavelet_ch, unet_ch, 1, bias=False)
        mid_ch = max(unet_ch // reduction, 32)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(unet_ch, mid_ch),
            nn.SiLU(),
            nn.Linear(mid_ch, unet_ch),
            nn.Sigmoid(),
        )
        # 初始化bias=1让gate初始接近1，训练更稳定
        nn.init.constant_(self.gate[-2].bias, 1.0)

    def convert_to_fp16(self):
        self.align.half()
        for m in self.gate.modules():
            if isinstance(m, nn.Linear):
                m.half()

    def convert_to_fp32(self):
        self.align.float()
        for m in self.gate.modules():
            if isinstance(m, nn.Linear):
                m.float()

    def forward(self, unet_feat, wavelet_feat):
        B, C, H, W = unet_feat.shape
        dtype = unet_feat.dtype
        wavelet_feat = wavelet_feat.to(dtype)
        if wavelet_feat.shape[2:] != (H, W):
            wavelet_feat = F.interpolate(wavelet_feat, size=(H, W),
                                         mode='bilinear', align_corners=False)
        aligned = self.align(wavelet_feat)          # [B, unet_ch, H, W]
        gate = self.gate(aligned).view(B, C, 1, 1)  # [B, unet_ch, 1, 1]
        return (unet_feat + gate * aligned).contiguous()