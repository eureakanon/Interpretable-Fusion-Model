import torch
import torch.nn.functional as F
import torch.nn as nn
from utils import apply_perturbation
def faithfulness_loss(model, images, labels, prob_orig, cam1, cam2, weights, threshold=0.5, temp=10.0):
    """
    计算可微分的忠实度损失。
    Args:
        model: 完整模型
        images: (B,3,H,W)
        labels: (B,)
        prob_orig: (B,) 原始图像在真实标签上的概率
        cam1, cam2: (B,1,H,W) 两个专家的热力图
        weights: (B,2) 融合权重
        threshold, temp: 扰动参数
    """
    # 扰动1
    perturbed1 = apply_perturbation(images, cam1, threshold=threshold, temp=temp)
    with torch.enable_grad():
        logits_fused_p1, _, _, _, _, _ = model(perturbed1)
        probs_p1 = F.softmax(logits_fused_p1, dim=1)
        prob_p1 = probs_p1[torch.arange(len(labels)), labels]
    delta1 = prob_orig - prob_p1
    # 释放中间变量
    del perturbed1, logits_fused_p1, probs_p1

    # 扰动2
    perturbed2 = apply_perturbation(images, cam2, threshold=threshold, temp=temp)
    with torch.enable_grad():
        logits_fused_p2, _, _, _, _, _ = model(perturbed2)
        probs_p2 = F.softmax(logits_fused_p2, dim=1)
        prob_p2 = probs_p2[torch.arange(len(labels)), labels]
    delta2 = prob_orig - prob_p2
    del perturbed2, logits_fused_p2, probs_p2

    w1 = weights[:, 0]
    w2 = weights[:, 1]
    loss_faith = (w1 * (1-delta1) + w2 * (1-delta2)).mean()
    return loss_faith
def compute_fusion_loss(model, images, labels,
                        logits_fused, logits1, logits2,
                        cam1, cam2, w1, w2,
                        alpha=0.1, beta=0.1, lambda_expert=0.2,
                        threshold=0.5, temp=10.0):
    # 1. Classification loss
    ce_fused = F.cross_entropy(logits_fused, labels)
    ce_expert1 = F.cross_entropy(logits1, labels)
    ce_expert2 = F.cross_entropy(logits2, labels)
    L_cls = ce_fused + lambda_expert * (ce_expert1 + ce_expert2)

    # 2. Consistency loss
    cam1_norm = cam1.view(cam1.size(0), -1)
    cam2_norm = cam2.view(cam2.size(0), -1)
    cam1_centered = cam1_norm - cam1_norm.mean(dim=1, keepdim=True)
    cam2_centered = cam2_norm - cam2_norm.mean(dim=1, keepdim=True)
    corr = (cam1_centered * cam2_centered).sum(dim=1) / (
        torch.sqrt((cam1_centered ** 2).sum(dim=1) + 1e-8) *
        torch.sqrt((cam2_centered ** 2).sum(dim=1) + 1e-8)
    )
    L_cons = (1 - corr).mean()

    # 3. Faithfulness loss
    # 计算原始图像在真实标签上的概率
    probs_orig = F.softmax(logits_fused, dim=1)
    prob_orig = probs_orig[torch.arange(len(labels)), labels]
    # 注意：cam1, cam2 已经与原始图像尺寸一致（需预先上采样）
    L_faith = faithfulness_loss(model, images, labels, prob_orig, cam1, cam2, 
                                torch.stack([w1, w2], dim=1), 
                                threshold=threshold, temp=temp)

    L_total = L_cls + alpha * L_cons + beta * L_faith
    return L_total