import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import numpy as np
import random
import argparse
from tqdm import tqdm
import torch.nn.functional as F
from ViTExpert import ViTExpert
# 导入自定义模块（假设四个文件在同一目录下）
from Fusion_model import InterpretationFusionNet
from Loss import faithfulness_loss, compute_fusion_loss
from utils import apply_perturbation
from models import FullyConvModel

# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 定义完整的主模型（包含两个专家和一个融合网络）
class ExpertFusionModel(nn.Module):
    def __init__(self, num_classes, backbone_name1='resnet50',backbone_name2='vit_base_patch16_224', pretrained=True):
        super(ExpertFusionModel, self).__init__()
        # 构建两个专家（使用相同的结构，但参数独立）
        # 注意：这里使用FullyConvModel，它需要一个预训练的模型作为骨干
        # 我们需要先加载预训练的骨干，再传入FullyConvModel
        # 为了简化，我们直接使用torchvision的预训练模型作为骨干
        # 但FullyConvModel期望一个完整的模型，它内部会提取骨干和分类器
        # 所以我们需要实例化两个不同的模型对象
        self.expert1 = self._build_expert(backbone_name1, num_classes, pretrained)
        self.expert2 = self._build_expert(backbone_name2, num_classes, pretrained)
        
        # 融合网络：输入5通道（原图+两个cam），输出两个权重
        self.fusion_net = InterpretationFusionNet(in_channels=5, out_dim=2)
    
    def _build_expert(self, backbone_name, num_classes, pretrained):
        # 加载预训练的完整模型
        if backbone_name == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
        elif backbone_name == 'bagnet33':
            # 假设bagnet33在timm中可用，这里简单处理
            import timm
            model = timm.create_model('bagnet33', pretrained=pretrained)
        elif backbone_name=='vit_base_patch16_224':
            return ViTExpert(num_classes, model_name=backbone_name, pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        # 包装成FullyConvModel
        # 注意：FullyConvModel期望cfg和model，但这里我们只传入model和num_classes
        # 根据Loss.py中的FullyConvModel定义，它需要cfg和model，但cfg似乎只用来获取network名称
        # 我们修改一下：直接使用默认参数，或者传入一个模拟的cfg对象
        # 为了方便，我们创建一个简单的cfg对象
        class DummyCfg:
            train = type('', (), {})()
            train.network = backbone_name
        cfg = DummyCfg()
        return FullyConvModel(cfg, model, num_classes)
    
    def forward(self, x):
    # 专家1
        logits1, cam1 = self.expert1(x)   # cam1: (B, num_classes, H', W')
    # 专家2
        logits2, cam2 = self.expert2(x)
    
    # 取平均得到单通道热力图
        cam1_mean = cam1.mean(dim=1, keepdim=True)   # (B, 1, H', W')
        cam2_mean = cam2.mean(dim=1, keepdim=True)
    
    # 上采样到原始图像尺寸
        cam1_mean = F.interpolate(cam1_mean, size=x.shape[-2:], mode='bilinear', align_corners=False)
        cam2_mean = F.interpolate(cam2_mean, size=x.shape[-2:], mode='bilinear', align_corners=False)
    
    # 融合网络输入（尺寸已一致）
        weights = self.fusion_net(x, cam1_mean, cam2_mean)   # (B, 2)
    
    # 加权融合logits
        logits_fused = weights[:, 0:1] * logits1 + weights[:, 1:2] * logits2
    
        return logits_fused, logits1, logits2, cam1_mean, cam2_mean, weights
# 数据预处理
def get_transforms(input_size=224):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform

def load_data(data_root, batch_size, val_split=0.2):
    train_transform, val_transform = get_transforms()
    # 加载完整训练集
    full_dataset = datasets.ImageFolder(os.path.join(data_root, 'Training'), transform=train_transform)
    # 划分训练/验证
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    # 对验证集使用单独的transform（无数据增强）
    val_dataset.dataset.transform = val_transform
    
    # 测试集
    test_dataset = datasets.ImageFolder(os.path.join(data_root, 'Test'), transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # 类别映射
    class_names = full_dataset.classes
    num_classes = len(class_names)
    print(f"Classes: {class_names}")
    return train_loader, val_loader, test_loader, num_classes

def train_one_epoch(model, train_loader, optimizer, device, args):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        # 前向传播
        logits_fused, logits1, logits2, cam1, cam2, weights = model(images)
        
        # 计算损失
        loss = compute_fusion_loss(
            model=model, images=images, labels=labels,
            logits_fused=logits_fused, logits1=logits1, logits2=logits2,
            cam1=cam1, cam2=cam2, w1=weights[:,0], w2=weights[:,1],
            alpha=args.alpha, beta=args.beta, lambda_expert=args.lambda_expert,
            threshold=args.threshold, temp=args.temp
        )
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        _, pred = torch.max(logits_fused, 1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({'loss': loss.item(), 'acc': correct/total})
    
    epoch_loss = total_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            logits_fused, _, _, _, _, _ = model(images)
            loss = nn.CrossEntropyLoss()(logits_fused, labels)
            total_loss += loss.item() * images.size(0)
            _, pred = torch.max(logits_fused, 1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    val_loss = total_loss / total
    val_acc = correct / total
    return val_loss, val_acc

def main(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载数据
    train_loader, val_loader, test_loader, num_classes = load_data(args.data_root, args.batch_size, args.val_split)
    
    # 创建模型
    model = ExpertFusionModel(num_classes, backbone_name1=args.backbone1, backbone_name2=args.backbone2 ,pretrained=args.pretrained).to(device)
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    
    best_val_acc = 0.0
    best_model_path = os.path.join(args.save_dir, 'best_model.pth')
    patience_counter = 0
    patience = args.early_stop_patience  # 新增参数
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, args)
        val_loss, val_acc = validate(model, val_loader, device)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        scheduler.step(val_loss)

        # 保存最佳模型并检查早停
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with val_acc={val_acc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epoch(s)")
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        

    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_acc = validate(model, test_loader, device)
    print(f"\nTest Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data', help='Path to dataset root (contains Training/ and Test/)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save models')
    parser.add_argument('--backbone1', type=str, default='resnet50', choices=['resnet50', 'bagnet33'], help='Backbone network')
    parser.add_argument('--backbone2',type=str,default='vit_base_patch16_224',help='Backbone network')
    parser.add_argument('--pretrained', action='store_true', default=True, help='Use pretrained weights')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--early_stop_patience', type=int, default=2, help='Patience for early stopping')
    # 损失函数超参数
    parser.add_argument('--alpha', type=float, default=0.1, help='Weight for consistency loss')
    parser.add_argument('--beta', type=float, default=0.1, help='Weight for faithfulness loss')
    parser.add_argument('--lambda_expert', type=float, default=0.2, help='Weight for expert classification loss')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for perturbation mask')
    parser.add_argument('--temp', type=float, default=10.0, help='Temperature for sigmoid mask')
    
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    main(args)