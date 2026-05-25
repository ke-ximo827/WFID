"""
Fine-tune IR-SE50 on Thermal-Visible Paired Data

使用你下载的IR-SE50预训练权重
在热成像-可见光配对数据上fine-tune
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
from tqdm import tqdm
import argparse


# ============================================================================
# 模型定义 (与test_ir_se50.py相同)
# ============================================================================

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


class Backbone_IR_SE50(nn.Module):
    def __init__(self, drop_ratio=0.4):
        super(Backbone_IR_SE50, self).__init__()
        
        blocks = get_blocks_se50()
        
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64)
        )
        
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(bottleneck)
        
        self.body = nn.Sequential(*modules)
        
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(drop_ratio),
            Flatten(),
            nn.Linear(512 * 7 * 7, 512),
            nn.BatchNorm1d(512)
        )

    def forward(self, x):
        """返回512维embedding用于triplet loss"""
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        return l2_norm(x)


# ============================================================================
# 数据集
# ============================================================================

class PairedDataset(Dataset):
    def __init__(self, thermal_dir, visible_dir):
        self.thermal_dir = thermal_dir
        self.visible_dir = visible_dir
        
        # 获取所有文件
        self.files = sorted([f for f in os.listdir(visible_dir) 
                            if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        print(f"找到 {len(self.files)} 对图像")
        
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        fname = self.files[idx]
        
        # 读取图像
        thermal_path = os.path.join(self.thermal_dir, fname)
        visible_path = os.path.join(self.visible_dir, fname)
        
        thermal = Image.open(thermal_path).convert('RGB')
        visible = Image.open(visible_path).convert('RGB')
        
        thermal = self.transform(thermal)
        visible = self.transform(visible)
        
        # 提取person ID (假设格式: personID_xxx.png)
        try:
            person_id = int(fname.split('_')[0])
        except:
            person_id = idx  # 如果格式不对，用索引
        
        return thermal, visible, person_id


# ============================================================================
# 损失函数
# ============================================================================

def triplet_loss(anchor, positive, negative, margin=0.3):
    """
    Triplet Loss
    同一人的热成像-可见光应该接近，不同人的应该远离
    """
    pos_dist = (anchor - positive).pow(2).sum(1)
    neg_dist = (anchor - negative).pow(2).sum(1)
    loss = F.relu(pos_dist - neg_dist + margin)
    return loss.mean()


def get_hard_negative(feat_thermal, feat_visible, labels):
    """挖掘hard negative: 找最相似的负样本"""
    B = feat_thermal.size(0)
    sim = torch.mm(feat_thermal, feat_visible.t())
    
    hard_neg_indices = []
    for i in range(B):
        neg_mask = (labels != labels[i])
        
        if neg_mask.sum() == 0:
            hard_neg_indices.append((i + 1) % B)
        else:
            neg_sim = sim[i].clone()
            neg_sim[~neg_mask] = -1e9
            hard_neg_idx = neg_sim.argmax().item()
            hard_neg_indices.append(hard_neg_idx)
    
    return feat_visible[hard_neg_indices]


# ============================================================================
# 训练
# ============================================================================

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    # 数据集
    print("\n加载数据...")
    dataset = PairedDataset(args.thermal_dir, args.visible_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    print(f"Batch数: {len(dataloader)}")
    
    # 模型
    print("\n初始化IR-SE50...")
    model = Backbone_IR_SE50(drop_ratio=0.4).to(device)
    
    # 加载预训练
    print(f"加载预训练: {args.pretrained}")
    state_dict = torch.load(args.pretrained, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    print("✓ 预训练加载完成")
    
    # 优化器
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=5e-4
    )
    
    # 学习率调度
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[30, 60, 90],
        gamma=0.1
    )
    
    # 训练循环
    print(f"\n开始Fine-tune (共{args.epochs}个epoch)...")
    print("="*70)
    
    model.train()
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for thermal, visible, labels in pbar:
            thermal = thermal.to(device)
            visible = visible.to(device)
            labels = labels.to(device)
            
            # 提取特征
            feat_thermal = model(thermal)  # [B, 512]
            feat_visible = model(visible)  # [B, 512]
            
            # Hard negative mining
            feat_negative = get_hard_negative(feat_thermal, feat_visible, labels)
            
            # Triplet loss
            loss = triplet_loss(feat_thermal, feat_visible, feat_negative, args.margin)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # 更新学习率
        scheduler.step()
        
        # 打印epoch统计
        avg_loss = epoch_loss / len(dataloader)
        lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f} | LR: {lr:.6f}")
        
        # 保存checkpoint
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            save_path = args.output.replace('.pth', f'_epoch{epoch+1}.pth')
            torch.save(model.state_dict(), save_path)
            print(f"✓ 保存checkpoint: {save_path}")
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = args.output.replace('.pth', '_best.pth')
            torch.save(model.state_dict(), best_path)
            print(f"✓ 保存最佳模型: {best_path} (loss={avg_loss:.4f})")
    
    # 保存最终模型
    torch.save(model.state_dict(), args.output)
    print(f"\n{'='*70}")
    print(f"✓ Fine-tune完成！")
    print(f"✓ 最终模型: {args.output}")
    print(f"✓ 最佳模型: {best_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fine-tune IR-SE50 on Thermal-Visible Data')
    parser.add_argument('--pretrained', default='./weights/model_ir_se50.pth',
                       help='预训练权重路径')
    parser.add_argument('--thermal_dir', default='./data/train/TH/',
                       help='热成像目录')
    parser.add_argument('--visible_dir', default='./data/train/VIS/',
                       help='可见光目录')
    parser.add_argument('--output', default='./weights_T/arcface_thermal_finetuned.pth',
                       help='输出权重路径')
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch大小')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='学习率')
    parser.add_argument('--margin', type=float, default=0.3,
                       help='Triplet loss margin')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    
    # 检查输入
    assert os.path.exists(args.pretrained), f"预训练权重不存在: {args.pretrained}"
    assert os.path.exists(args.thermal_dir), f"热成像目录不存在: {args.thermal_dir}"
    assert os.path.exists(args.visible_dir), f"可见光目录不存在: {args.visible_dir}"
    
    print("配置:")
    print(f"  预训练: {args.pretrained}")
    print(f"  热成像: {args.thermal_dir}")
    print(f"  可见光: {args.visible_dir}")
    print(f"  输出: {args.output}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch: {args.batch_size}")
    print(f"  LR: {args.lr}")
    print()
    
    train(args)