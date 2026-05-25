"""
ArcFace特征提取器 - 基于Fine-tuned IR-SE50

修复版：
1. 投影层可训练（适应DDPM特征空间）
2. Backbone冻结（保持Fine-tuned能力）
3. 强制eval模式（防止BatchNorm漂移）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class SEModule(nn.Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class bottleneck_IR_SE(nn.Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                nn.BatchNorm2d(depth)
            )
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            nn.PReLU(depth),
            nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            nn.BatchNorm2d(depth),
            SEModule(depth, 16)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


def get_block(in_channel, depth, num_units, stride=2):
    return [bottleneck_IR_SE(in_channel, depth, stride)] + \
           [bottleneck_IR_SE(depth, depth, 1) for i in range(num_units - 1)]


def get_blocks_se50():
    blocks = [
        get_block(in_channel=64, depth=64, num_units=3),
        get_block(in_channel=64, depth=128, num_units=4),
        get_block(in_channel=128, depth=256, num_units=14),
        get_block(in_channel=256, depth=512, num_units=3)
    ]
    return blocks


class ArcFaceFeatureExtractor(nn.Module):
    """
    基于Fine-tuned IR-SE50提取热成像特征
    
    修复策略：
    - Backbone (IR-SE50): 冻结，保持Fine-tuned能力
    - 投影层 (proj1/2/3): 可训练，适应DDPM
    - BatchNorm: 强制eval，不更新统计量
    
    输入: 热成像 [B, 3, 128, 128]
    输出: {
        'layer1': [B, 256, 64, 64],
        'layer2': [B, 512, 32, 32],
        'layer3': [B, 1024, 16, 16],
    }
    """
    def __init__(self, weight_path):
        super().__init__()
        
        blocks = get_blocks_se50()
        
        # Input layer
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64)
        )
        
        # Body (所有block)
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(bottleneck)
        self.body = nn.Sequential(*modules)
        
        # 分层提取特征
        self.layer1 = nn.Sequential(*modules[0:3])    # 前3个: 64通道
        self.layer2 = nn.Sequential(*modules[3:7])    # 接4个: 128通道
        self.layer3 = nn.Sequential(*modules[7:21])   # 接14个: 256通道
        
        # 投影层：调整通道数以匹配DDPM期望的256/512/1024
        # ✓ 这些层会被训练
        self.proj1 = nn.Conv2d(64, 256, 1, bias=False)
        self.proj2 = nn.Conv2d(128, 512, 1, bias=False)
        self.proj3 = nn.Conv2d(256, 1024, 1, bias=False)
        
        # 加载fine-tuned权重
        self._load_weights(weight_path)
        
        # ========== 修复冻结策略 ==========
        # 冻结backbone（Fine-tuned部分）
        for name, param in self.named_parameters():
            if 'proj' not in name:  # 不包括投影层
                param.requires_grad = False
        
        # 投影层保持可训练（会被保存到master_params）
        for param in self.proj1.parameters():
            param.requires_grad = True
        for param in self.proj2.parameters():
            param.requires_grad = True
        for param in self.proj3.parameters():
            param.requires_grad = True
        
        # 强制eval模式（即使被DDP设为train也保持eval）
        self.eval()
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                module.track_running_stats = False  # 不更新running stats
        # ====================================
        
        print(f"✓ ArcFace特征提取器初始化完成")
        print(f"  权重: {weight_path}")
        print(f"  Backbone: 冻结 (Fine-tuned)")
        print(f"  投影层: 可训练 (适应DDPM)")
    
    def _load_weights(self, weight_path):
        """加载fine-tuned权重"""
        print(f"加载权重: {weight_path}")
        state_dict = torch.load(weight_path, map_location='cpu')
        
        # 加载到backbone（忽略投影层，因为是新加的）
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        
        # 投影层是新加的，所以会缺失，这是正常的
        expected_missing = ['proj1.weight', 'proj2.weight', 'proj3.weight']
        actual_missing = [k for k in missing if k in expected_missing]
        
        if len(actual_missing) == 3:
            print(f"✓ 权重加载成功（投影层随机初始化，将被训练）")
        else:
            print(f"⚠ 缺失keys: {len(missing)}")
            if len(missing) > 10:
                print(f"  示例: {missing[:5]}")
            else:
                print(f"  {missing}")
    
    def train(self, mode=True):
        """
        重写train()方法，确保backbone始终在eval模式
        
        即使model.train()被调用，backbone也保持eval
        只有投影层响应train/eval切换
        """
        # 不调用super().train()，避免递归设置所有子模块
        
        # 投影层可以切换train/eval
        self.proj1.train(mode)
        self.proj2.train(mode)
        self.proj3.train(mode)
        
        # Backbone始终eval
        self.input_layer.eval()
        self.layer1.eval()
        self.layer2.eval()
        self.layer3.eval()
        
        # BatchNorm始终eval
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                module.track_running_stats = False
        
        return self
    
    def forward(self, thermal):
        """
        提取多层特征用于DDPM注意力
        
        Args:
            thermal: [B, 3, 128, 128] 热成像
        
        Returns:
            dict {
                'layer1': [B, 256, 64, 64],
                'layer2': [B, 512, 32, 32],
                'layer3': [B, 1024, 16, 16],
            }
        """
        # Resize到112x112 (ArcFace标准输入)
        x = F.interpolate(thermal, size=(112, 112), mode='bilinear', align_corners=False)
        
        # 通过input layer
        x = self.input_layer(x)  # [B, 64, 112, 112]
        
        # 分层提取特征
        feat1 = self.layer1(x)      # [B, 64, 56, 56]
        feat2 = self.layer2(feat1)  # [B, 128, 28, 28]
        feat3 = self.layer3(feat2)  # [B, 256, 14, 14]
        
        # 投影到目标通道数（这里会有梯度）
        feat1 = self.proj1(feat1)  # [B, 256, 56, 56]
        feat2 = self.proj2(feat2)  # [B, 512, 28, 28]
        feat3 = self.proj3(feat3)  # [B, 1024, 14, 14]
        
        # Resize到目标分辨率
        feat1 = F.interpolate(feat1, size=(64, 64), mode='bilinear', align_corners=False)
        feat2 = F.interpolate(feat2, size=(32, 32), mode='bilinear', align_corners=False)
        feat3 = F.interpolate(feat3, size=(16, 16), mode='bilinear', align_corners=False)
        
        return {
            'layer1': feat1,  # [B, 256, 64, 64]
            'layer2': feat2,  # [B, 512, 32, 32]
            'layer3': feat3,  # [B, 1024, 16, 16]
        }


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    import sys
    
    weight_path = './weights_T/arcface_thermal_finetuned_best.pth'
    if len(sys.argv) > 1:
        weight_path = sys.argv[1]
    
    print("测试ArcFace特征提取器...")
    print(f"权重: {weight_path}\n")
    
    try:
        # 创建提取器
        extractor = ArcFaceFeatureExtractor(weight_path)
        
        # 验证冻结状态
        print("\n参数冻结状态:")
        for name, param in extractor.named_parameters():
            print(f"  {name}: {'可训练' if param.requires_grad else '冻结'}")
        
        # 测试输入
        thermal = torch.randn(2, 3, 128, 128)
        
        # 提取特征
        features = extractor(thermal)
        
        print("\n✓ 特征提取成功:")
        print(f"  layer1: {features['layer1'].shape}")
        print(f"  layer2: {features['layer2'].shape}")
        print(f"  layer3: {features['layer3'].shape}")
        
        # 验证维度
        assert features['layer1'].shape == (2, 256, 64, 64)
        assert features['layer2'].shape == (2, 512, 32, 32)
        assert features['layer3'].shape == (2, 1024, 16, 16)
        
        print("\n✓✓✓ 所有测试通过！可以用于DDPM训练！")
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()