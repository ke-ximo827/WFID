"""
Identity-Level Cross-Attention - 修复版
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentityCrossAttention(nn.Module):
    """
    Identity-Level Cross-Attention
    在Middle Block (16x16)注入全局身份特征
    """
    def __init__(self, unet_channels, thermal_channels, hidden_dim=512):
        super().__init__()
        
        self.unet_channels = unet_channels
        self.thermal_channels = thermal_channels
        self.hidden_dim = hidden_dim
        
        # Query from UNet features
        self.to_q = nn.Conv2d(unet_channels, hidden_dim, 1)
        
        # Key & Value from thermal features
        self.to_k = nn.Conv2d(thermal_channels, hidden_dim, 1)
        self.to_v = nn.Conv2d(thermal_channels, hidden_dim, 1)
        
        # Output projection
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
            unet_feat: [B, 768, H, W] UNet middle block特征
            thermal_feat: [B, 1024, H, W] 热图像layer3特征
        
        Returns:
            [B, 768, H, W] 注入身份信息后的特征
        """
        B, C, H, W = unet_feat.shape
        
        # 确保dtype一致
        dtype = unet_feat.dtype
        thermal_feat = thermal_feat.type(dtype)
        
        # Query, Key, Value
        q = self.to_q(unet_feat)  # [B, 512, H, W]
        k = self.to_k(thermal_feat)  # [B, 512, H, W]
        v = self.to_v(thermal_feat)  # [B, 512, H, W]
        
        # Reshape for attention: [B, C, H, W] → [B, H*W, C]
        q = q.flatten(2).transpose(1, 2)  # [B, H*W, 512]
        k = k.flatten(2).transpose(1, 2)  # [B, H*W, 512]
        v = v.flatten(2).transpose(1, 2)  # [B, H*W, 512]
        
        # Attention
        attn = torch.bmm(q, k.transpose(1, 2)) * self.scale  # [B, H*W, H*W]
        attn = F.softmax(attn.float(), dim=-1).type(dtype)
        
        # Apply attention to values
        out = torch.bmm(attn, v)  # [B, H*W, 512]
        
        # Reshape back: [B, H*W, 512] → [B, 512, H, W]
        out = out.transpose(1, 2).reshape(B, self.hidden_dim, H, W)
        
        # Output projection
        out = self.to_out(out)  # [B, 768, H, W]
        
        # Residual connection
        return (unet_feat + out).contiguous()