# vit_expert.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math

class ViTExpert(nn.Module):
    """
    Vision Transformer 专家，输出 logits 和可微的类无关热力图（类似 SoftCAM）。
    热力图通过分类器权重与 patch tokens 的加权平均生成，适合在训练阶段用于融合网络和忠实度损失。
    """
    def __init__(self, num_classes, model_name='vit_base_patch16_224', pretrained=True):
        super().__init__()
        # 加载预训练 ViT，确保 num_classes 匹配
        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        self.num_classes = num_classes
        self.patch_size = 16  # 假设标准 ViT，实际可从模型获取
        self.img_size = 224   # 默认输入尺寸
        self.num_patches = (self.img_size // self.patch_size) ** 2

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) 输入图像，H,W 应与模型训练尺寸一致（如224）。
        Returns:
            logits: (B, num_classes) 分类 logits
            cam: (B, 1, H, W) 可微热力图，值域 [0,1]
        """
        # 获取所有 token 的特征 (B, N, D)，其中 N = 1 + num_patches
        tokens = self.vit.forward_features(x)   # 注意：timm 的 VisionTransformer 支持此方法
        cls_token = tokens[:, 0, :]              # (B, D)
        patch_tokens = tokens[:, 1:, :]          # (B, P, D)

        # 分类 logits
        logits = self.vit.head(cls_token)        # (B, num_classes)

        # 分类器权重 (num_classes, D)
        weight = self.vit.head.weight
        # 可选：偏置，通常 CAM 不使用偏置
        # bias = self.vit.head.bias

        # 计算每个类别的激活图 (B, P, num_classes)
        # 使用 einsum: bpd, cd -> bpc
        cam_per_class = torch.einsum('bpd,cd->bpc', patch_tokens, weight)  # 未加偏置

        # 生成类无关的热力图：对所有类别取平均（可微，简单有效）
        cam = cam_per_class.mean(dim=-1)         # (B, P)

        # 或者使用预测类别的加权平均（通过 softmax 概率），也是可微的，但计算稍多
        # probs = F.softmax(logits, dim=1)       # (B, num_classes)
        # cam = torch.einsum('bpc,bc->bp', cam_per_class, probs)

        # 将序列 reshape 为空间网格 (B, 1, H, W)
        H = W = int(math.sqrt(self.num_patches))
        cam = cam.view(-1, H, W).unsqueeze(1)    # (B, 1, H, W)

        # 双线性上采样到原始输入尺寸
        cam = F.interpolate(cam, size=x.shape[-2:], mode='bilinear', align_corners=False)

        # 可选：将热力图归一化到 [0,1] 范围（有利于后续扰动）
        # cam_min = cam.view(cam.size(0), -1).min(dim=1, keepdim=True)[0].view(cam.size(0), 1, 1, 1)
        # cam_max = cam.view(cam.size(0), -1).max(dim=1, keepdim=True)[0].view(cam.size(0), 1, 1, 1)
        # cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        # 但注意：如果希望热力图为正值且幅度适当，也可不强制归一化，因为后续 sigmoid 会处理。

        return logits, cam