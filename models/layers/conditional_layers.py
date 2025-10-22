# models/layers/conditional_layers.py

import torch
import torch.nn as nn

class ConditionalAdaIN(nn.Module):
    """
    条件化自适应实例归一化 (Conditional Adaptive Instance Normalization)
    
    该模块根据输入的条件向量 c, 动态地调整特征图 x 的均值和方差。
    它首先将 x 进行实例归一化，然后通过一个小型神经网络从 c 预测出
    仿射变换的参数（缩放因子 gamma 和平移因子 beta），最后应用这些参数。
    """
    def __init__(self, condition_dim, feature_channels):
        """
        Args:
            condition_dim (int): 条件向量的维度。
            feature_channels (int): 输入特征图的通道数。
        """
        super().__init__()
        
        # 神经网络，用于从条件向量预测 gamma 和 beta
        # 输出维度是 feature_channels 的两倍，一半给 gamma，一半给 beta
        self.mlp = nn.Sequential(
            nn.Linear(condition_dim, feature_channels),
            nn.ReLU(),
            nn.Linear(feature_channels, feature_channels * 2)
        )
        
        # 实例归一化
        self.instance_norm = nn.InstanceNorm2d(feature_channels, affine=False)

    def forward(self, x, c):
        """
        Args:
            x (torch.Tensor): 输入特征图, 形状 (B, C, H, W)。
            c (torch.Tensor): 条件向量, 形状 (B, condition_dim)。
        
        Returns:
            torch.Tensor: 经过调制的特征图。
        """
        # 1. 对特征图进行实例归一化
        normalized = self.instance_norm(x)
        
        # 2. 从条件向量 c 预测 gamma 和 beta
        # mlp_out 形状: (B, C * 2)
        mlp_out = self.mlp(c)
        
        # 3. 将 mlp_out 调整形状并切分
        # view 操作将其变为 (B, C, 2)
        # unsqueeze 操作增加 H, W 维度，以便进行广播 -> (B, C, 1, 1)
        gamma, beta = mlp_out.view(x.size(0), x.size(1), 2).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=2)

        # 4. 应用仿射变换
        # output = gamma * normalized_feature + beta
        return gamma * normalized + beta
