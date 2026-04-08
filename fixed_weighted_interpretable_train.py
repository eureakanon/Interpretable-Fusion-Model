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
from models import FullyConvModel
from Loss import faithfulness_loss, compute_fusion_loss_fixed  # 注意：需要修改损失函数版本

# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 定义固定权重融合模型（无门控网络）
class FixedWeightFusionModel(nn.Module):
    def __init__(self, num_classes, backbone_name1='resnet50', backbone_name2='vit_base_patch16_224',
                 pretrained=True, fixed_weight1=0.5, fixed_weight2=0.5):
        super(FixedWeightFusionModel, self).__init__()
        self.expert1 = self._build_expert(backbone_name1, num_classes, pretrained)
        self.expert2 = self._build_expert(backbone_name2, num_classes, pretrained)
        self.fixed_weight1 = fixed_weight1
        self.fixed_weight2 = fixed_weight2
        # 注册为缓冲区，使其不参与梯度更新
        self.register_buffer('weight1', torch.tensor(fixed_weight1))
        self.register_buffer('weight2', torch.tensor(fixed_weight2))

    def _build_expert(self, backbone_name, num_classes, pretrained):
        if backbone_name == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
        elif backbone_name == 'bagnet33':
            import timm
            model = timm.create_model('bagnet33', pretrained=pretrained)
        elif backbone_name == 'vit_base_patch16_224':
            return ViTExpert(num_classes, model_name=backbone_name, pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        class DummyCfg:
            train = type('', (), {})()
            train.network = backbone_name
        cfg = DummyCfg()
        return FullyConvModel(cfg, model, num_classes)

    def forward(self, x):
        logits1, cam1 = self.expert1(x)
        logits2, cam2 = self.expert2(x)

        # 取平均得到单通道热力图（用于损失计算）
        cam1_mean = cam1.mean(dim=1, keepdim=True)
        cam2_mean = cam2.mean(dim=1, keepdim=True)

        # 上采样至原图尺寸
        cam1_mean = F.interpolate(cam1_mean, size=x.shape[-2:], mode='bilinear', align_corners=False)
        cam2_mean = F.interpolate(cam2_mean, size=x.shape[-2:], mode='bilinear', align_corners=False)

        # 固定权重融合
        weights = torch.stack([self.weight1.expand(x.size(0)),
                               self.weight2.expand(x.size(0))], dim=1)  # (B,2)
        logits_fused = self.weight1 * logits1 + self.weight2 * logits2

        # 返回格式与 ExpertFusionModel 一致，方便复用损失函数
        return logits_fused, logits1, logits2, cam1_mean, cam2_mean, weights


# 修改损失函数以支持固定权重（其实原损失函数已支持，但需确保传入的 w1,w2 正确）
# 这里直接复用 compute_fusion_loss，但需要导入时注意路径
# 为了安全，我们定义一个新的损失函数，内部使用固定权重



# 数据预处理与加载（与原始 train.py 完全相同）
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
    full_dataset = datasets.ImageFolder(os.path.join(data_root, 'Training'), transform=train_transform)
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset.transform = val_transform
    test_dataset = datasets.ImageFolder(os.path.join(data_root, 'Test'), transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
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
        logits_fused, logits1, logits2, cam1, cam2, weights = model(images)
        # 使用相同的损失函数（固定权重作为参数传入）
        loss = compute_fusion_loss_fixed(
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

    train_loader, val_loader, test_loader, num_classes = load_data(args.data_root, args.batch_size, args.val_split)

    model = FixedWeightFusionModel(
        num_classes,
        backbone_name1=args.backbone1,
        backbone_name2=args.backbone2,
        pretrained=args.pretrained,
        fixed_weight1=args.fixed_weight1,
        fixed_weight2=args.fixed_weight2
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    best_val_acc = 0.0
    best_model_path = os.path.join(args.save_dir, 'best_model.pth')
    patience_counter = 0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, args)
        val_loss, val_acc = validate(model, val_loader, device)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with val_acc={val_acc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epoch(s)")
            if patience_counter >= args.early_stop_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_acc = validate(model, test_loader, device)
    print(f"\nTest Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_fixed')
    parser.add_argument('--backbone1', type=str, default='resnet50', choices=['resnet50', 'bagnet33'])
    parser.add_argument('--backbone2', type=str, default='vit_base_patch16_224')
    parser.add_argument('--pretrained', action='store_true', default=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--early_stop_patience', type=int, default=2)
    # 固定权重参数
    parser.add_argument('--fixed_weight1', type=float, default=0.5, help='Fixed weight for expert1')
    parser.add_argument('--fixed_weight2', type=float, default=0.5, help='Fixed weight for expert2')
    # 损失函数超参数
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--lambda_expert', type=float, default=0.2)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--temp', type=float, default=10.0)

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    main(args)