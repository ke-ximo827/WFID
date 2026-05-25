"""
傅里叶全局增强模块
- 用全局2D FFT替代self-attention做全局建模（来自PW-FNet）
- 插在Encoder最后输出和Middle Block之间
- residual_scale从0初始化，训练稳定
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierEnhancementBlock(nn.Module):
    """
    流程：输入 → GroupNorm → FFT → 实部/虚部各做1×1卷积 → IFFT → 残差相加
    关键：residual_scale初始为0，让模块从零开始缓慢激活
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.channels = channels
        mid_ch = max(channels // reduction, 32)

        self.norm = nn.GroupNorm(32, channels)

        # 实部和虚部独立处理（物理意义不同）
        self.real_conv = nn.Sequential(
            nn.Conv2d(channels, mid_ch, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(mid_ch, channels, 1, bias=False),
        )
        self.imag_conv = nn.Sequential(
            nn.Conv2d(channels, mid_ch, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(mid_ch, channels, 1, bias=False),
        )
        self.out_norm = nn.GroupNorm(32, channels)

        # 初始为0：训练初期不干扰原始特征
        self.residual_scale = nn.Parameter(torch.zeros(1))

    def convert_to_fp16(self):
        self.norm.half()
        self.out_norm.half()
        for seq in [self.real_conv, self.imag_conv]:
            for m in seq.modules():
                if isinstance(m, nn.Conv2d):
                    m.half()

    def convert_to_fp32(self):
        self.norm.float()
        self.out_norm.float()
        for seq in [self.real_conv, self.imag_conv]:
            for m in seq.modules():
                if isinstance(m, nn.Conv2d):
                    m.float()

    def forward(self, x):
        residual = x
        dtype = x.dtype

        x_norm = self.norm(x)

        # FFT 用fp32保证数值稳定
        X = torch.fft.rfft2(x_norm.float(), norm='ortho')  # 复数

        # 频域卷积
        X_real_out = self.real_conv(X.real.to(dtype))
        X_imag_out = self.imag_conv(X.imag.to(dtype))

        # IFFT
        X_out = torch.complex(X_real_out.float(), X_imag_out.float())
        x_spatial = torch.fft.irfft2(
            X_out, s=(x.shape[2], x.shape[3]), norm='ortho'
        ).to(dtype)

        x_spatial = self.out_norm(x_spatial)

        return (residual + self.residual_scale * x_spatial).contiguous()