# training/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights

class PerceptualLoss(nn.Module):
    """
    感知损失 (Perceptual Loss)
    使用预训练的VGG19网络提取高维特征进行比较，能更好地衡量图像的感知相似性。
    """
    def __init__(self, feature_layers=[2, 7, 12, 21, 30]):
        """
        Args:
            feature_layers (list): VGG19中用于提取特征的层的索引。
                                   默认值为 [2, 7, 12, 21, 30] 对应 'relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1'
        """
        super().__init__()
        # 加载预训练的VGG19模型，并设置为评估模式
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features
        self.features = nn.ModuleList([vgg[i] for i in range(max(feature_layers) + 1)])
        for param in self.features.parameters():
            param.requires_grad = False # 冻结参数

        self.feature_layers = feature_layers
        
        # VGG网络使用ImageNet的均值和标准差进行归一化
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize(self, tensor):
        """对输入张量进行VGG标准化"""
        return (tensor - self.mean) / self.std

    def forward(self, pred, target):
        """
        计算预测图像和目标图像之间的感知损失。
        Args:
            pred (torch.Tensor): 预测图像批次, (B, C, H, W), 值域 [0, 1]。
            target (torch.Tensor): 真实图像批次, (B, C, H, W), 值域 [0, 1]。
        Returns:
            torch.Tensor: 感知损失值。
        """
        # 对输入进行VGG标准化
        pred = self.normalize(pred)
        target = self.normalize(target)
        
        perceptual_loss = 0.0
        for i, layer in enumerate(self.features):
            pred = layer(pred)
            target = layer(target)
            if i in self.feature_layers:
                perceptual_loss += F.l1_loss(pred, target)
        
        return perceptual_loss


class CombinedLoss(nn.Module):
    """
    组合多种损失函数。
    """
    def __init__(self, loss_weights):
        """
        Args:
            loss_weights (dict): 包含各类损失权重的字典, 例如 {'l1': 1.0, 'perceptual': 0.1}
        """
        super().__init__()
        self.loss_weights = loss_weights
        self.l1_loss = nn.L1Loss()
        
        # 如果需要，才实例化感知损失，避免不必要的显存占用
        if 'perceptual' in self.loss_weights and self.loss_weights['perceptual'] > 0:
            self.perceptual_loss = PerceptualLoss()
        else:
            self.perceptual_loss = None

    def forward(self, pred, target):
        """
        计算总损失。
        Args:
            pred (torch.Tensor): 预测图像批次, (B, C, H, W)。
            target (torch.Tensor): 真实图像批次, (B, C, H, W)。
        Returns:
            tuple: (总损失, 包含各项损失的字典)。
        """
        loss_dict = {}
        total_loss = 0.0
        
        # 计算 L1 损失
        if 'l1' in self.loss_weights and self.loss_weights['l1'] > 0:
            l1 = self.l1_loss(pred, target)
            loss_dict['l1_loss'] = l1
            total_loss += self.loss_weights['l1'] * l1
            
        # 计算感知损失
        if self.perceptual_loss is not None and self.loss_weights['perceptual'] > 0:
            perceptual = self.perceptual_loss(pred, target)
            loss_dict['perceptual_loss'] = perceptual
            total_loss += self.loss_weights['perceptual'] * perceptual
            
        loss_dict['total_loss'] = total_loss
        return total_loss, loss_dict
