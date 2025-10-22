# experiments/test_data.py 

import sys
sys.path.append('..')

import yaml
import torch
import matplotlib.pyplot as plt
import numpy as np
from data.dataset import create_dataloaders

def test_dataloader():
    """测试数据加载器"""
    
    # 加载配置
    with open('../config/unet_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 修改配置以便快速测试
    config['data']['batch_size'] = 4
    config['data']['num_workers'] = 0  # 方便调试
    
    # 创建数据加载器
    train_loader, val_loader = create_dataloaders(config, val_split_ratio=0.1)
    
    # 获取一个batch
    batch = next(iter(train_loader))
    
    print("=" * 50)
    print("数据批次信息:")
    print(f"Degraded shape: {batch['degraded'].shape}")
    print(f"Original shape: {batch['original'].shape}")
    print(f"Condition shape: {batch['condition'].shape}")
    print(f"Batch size: {batch['degraded'].shape[0]}")
    print("=" * 50)
    
    # 打印退化信息
    for i in range(len(batch['degradation_type'])):
        print(f"\n样本 {i}:")
        print(f"  退化类型: {batch['degradation_type'][i]}")
        print(f"  退化索引: {batch['degradation_idx'][i]}")
        print(f"  条件向量: {batch['condition'][i].tolist()}")
        print(f"  参数详情: {batch['degradation_params'][i]}")
    
    # 可视化前2个样本
    visualize_batch(batch, num_samples=2)
    
    print("\n✓ 数据加载测试成功!")
    
def visualize_batch(batch, num_samples=4):
    """可视化一个batch的样本"""
    
    num_samples = min(num_samples, batch['degraded'].shape[0])
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # 转换为numpy并调整到[0, 1]
        degraded = batch['degraded'][i].permute(1, 2, 0).cpu().numpy()
        original = batch['original'][i].permute(1, 2, 0).cpu().numpy()
        
        # 计算差异图
        diff = np.abs(original - degraded)
        
        # 获取退化类型
        deg_type = batch['degradation_type'][i]
        params = batch['degradation_params'][i]
        
        # 显示图像
        axes[i, 0].imshow(np.clip(original, 0, 1))
        axes[i, 0].set_title(f"Original")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(np.clip(degraded, 0, 1))
        axes[i, 1].set_title(f"Degraded: {deg_type}\nParams: {[f'{p:.2f}' for p in params[:2]]}")
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(diff, cmap='hot')
        axes[i, 2].set_title(f"Difference (MAE: {diff.mean():.4f})")
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('../outputs/visualizations/data_test.png', dpi=150, bbox_inches='tight')
    print("\n可视化结果已保存到: outputs/visualizations/data_test.png")
    plt.show()

if __name__ == '__main__':
    test_dataloader()
