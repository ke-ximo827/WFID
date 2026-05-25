"""
Region-Level Cross-Attention - 修复版
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RegionCrossAttention(nn.Module):
    """
    Region-Level Cross-Attention
    在Decoder (32x32)注入局部区域特征
    """
    def __init__(self, unet_channels, thermal_channels, num_regions=5):
        super().__init__()
        
        self.unet_channels = unet_channels
        self.thermal_channels = thermal_channels
        self.num_regions = num_regions
        
        # Region-wise projections
        hidden_dim = 256
        self.hidden_dim = hidden_dim
        self.to_q = nn.Conv2d(unet_channels, hidden_dim, 1)
        self.to_k = nn.Conv2d(thermal_channels, hidden_dim, 1)
        self.to_v = nn.Conv2d(thermal_channels, hidden_dim, 1)
        
        self.to_out = nn.Conv2d(hidden_dim, unet_channels, 1)
        
        self.scale = hidden_dim ** -0.5
    
    def convert_to_fp16(self):
        """转换所有Conv2d为FP16"""
        self.to_q.half()
        self.to_k.half()
        self.to_v.half()
        self.to_out.half()
    
    def convert_to_fp32(self):
        """转换所有Conv2d为FP32"""
        self.to_q.float()
        self.to_k.float()
        self.to_v.float()
        self.to_out.float()

    def forward(self, unet_feat, thermal_feat):
        """
        Args:
            unet_feat: [B, 768, H, W] Decoder特征
            thermal_feat: [B, 512, H, W] 热图像layer2特征
        
        Returns:
            [B, 768, H, W] 注入区域信息后的特征
        """
        B, C, H, W = unet_feat.shape
        
        # 确保dtype一致
        dtype = unet_feat.dtype
        thermal_feat = thermal_feat.type(dtype)
        
        # Q, K, V
        q = self.to_q(unet_feat)  # [B, 256, H, W]
        k = self.to_k(thermal_feat)  # [B, 256, H, W]
        v = self.to_v(thermal_feat)  # [B, 256, H, W]
        
        # Reshape: [B, C, H, W] → [B, H*W, C]
        q = q.flatten(2).transpose(1, 2)  # [B, H*W, 256]
        k = k.flatten(2).transpose(1, 2)  # [B, H*W, 256]
        v = v.flatten(2).transpose(1, 2)  # [B, H*W, 256]
        
        # Attention
        attn = torch.bmm(q, k.transpose(1, 2)) * self.scale  # [B, H*W, H*W]
        attn = F.softmax(attn.float(), dim=-1).type(dtype)
        
        out = torch.bmm(attn, v)  # [B, H*W, 256]
        
        # Reshape back: [B, H*W, 256] → [B, 256, H, W]
        out = out.transpose(1, 2).reshape(B, self.hidden_dim, H, W)
        
        out = self.to_out(out)  # [B, 768, H, W]
        
        return (unet_feat + out).contiguous()