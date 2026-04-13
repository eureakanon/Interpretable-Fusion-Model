import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np
import random
import argparse
from tqdm import tqdm
import timm

# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 数据预处理（与原始模型完全相同）
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
    # 对验证集使用单独的transform（无数据增强）
    val_dataset.dataset.transform = val_transform
    # 测试集
    test_dataset = datasets.ImageFolder(os.path.join(data_root, 'Test'), transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    class_names = full_dataset.classes
    num_classes = len(class_names)
    print(f"Classes: {class_names}")
    return train_loader, val_loader, test_loader, num_classes

# 定义原始 ViT 模型（无可解释性输出）
class RawViT(nn.Module):
    def __init__(self, num_classes, model_name='vit_base_patch16_224', pretrained=True):
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        logits = self.vit(x)
        return logits

def train_one_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        _, pred = torch.max(logits, 1)
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
            logits = model(images)
            loss = nn.CrossEntropyLoss()(logits, labels)
            total_loss += loss.item() * images.size(0)
            _, pred = torch.max(logits, 1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    val_loss = total_loss / total
    val_acc = correct / total
    return val_loss, val_acc

def main(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, num_classes = load_data(
        args.data_root, args.batch_size, args.val_split
    )

    model = RawViT(num_classes, model_name=args.model_name, pretrained=args.pretrained).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    best_val_acc = 0.0
    best_model_path = os.path.join(args.save_dir, 'best_model.pth')
    patience_counter = 0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
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

    # 加载最佳模型并测试
    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_acc = validate(model, test_loader, device)
    print(f"\nTest Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data', help='Path to dataset root')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_raw_vit', help='Directory to save models')
    parser.add_argument('--model_name', type=str, default='vit_base_patch16_224', help='ViT model name from timm')
    parser.add_argument('--pretrained', action='store_true', default=True, help='Use pretrained weights')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate (larger than fine-tuned)')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--early_stop_patience', type=int, default=3, help='Patience for early stopping')
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    main(args)