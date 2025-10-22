# experiments/test_data.py 

import os
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
    
    # --- 为快速测试修改配置 ---
    config['data']['batch_size'] = 4
    config['data']['num_workers'] = 0  # 方便调试
    # 使用一个较小的子集来加速测试
    SUBSET_SIZE_FOR_TEST = 100 
    
    # --- 关键修改：接收所有三个数据加载器 ---
    print("正在创建数据加载器（使用100个样本的子集进行测试）...")
    train_loader, val_loader, test_loader = create_dataloaders(config, subset_size=SUBSET_SIZE_FOR_TEST)
    
    # 从训练数据加载器中获取一个批次来检查数据增强
    print("\n从 train_loader 中获取一个批次进行测试...")
    batch = next(iter(train_loader))
    
    print("=" * 50)
    print("数据批次信息:")
    print(f"Degraded shape: {batch['degraded'].shape}")
    print(f"Original shape: {batch['original'].shape}")
    print(f"Condition shape: {batch['condition'].shape}")
    print(f"Batch size: {batch['degraded'].shape[0]}")
    # 验证图像尺寸是否正确
    assert batch['degraded'].shape[2] == config['data']['image_size']
    assert batch['degraded'].shape[3] == config['data']['image_size']
    print(f"图像尺寸 (H, W): ({batch['degraded'].shape[2]}, {batch['degraded'].shape[3]}) - 检查通过")
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
    
    print("\n✓ 数据加载和增强测试成功!")
    
def visualize_batch(batch, num_samples=4):
    """可视化一个batch的样本"""
    
    num_samples = min(num_samples, batch['degraded'].shape[0])
    
    # 修正：确保在只有一个样本时，axes也能被正确索引
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle("数据加载器与数据增强测试", fontsize=16)
    
    for i in range(num_samples):
        # 转换为numpy并调整到[0, 1]
        degraded = batch['degraded'][i].permute(1, 2, 0).cpu().numpy()
        original = batch['original'][i].permute(1, 2, 0).cpu().numpy()
        
        # 计算差异图
        diff = np.abs(original - degraded)
        
        # 获取退化类型和参数
        deg_type = batch['degradation_type'][i]
        params = batch['degradation_params'][i]
        
        # 显示原图（经过数据增强后）
        axes[i, 0].imshow(np.clip(original, 0, 1))
        axes[i, 0].set_title(f"Original (Augmented)")
        axes[i, 0].axis('off')
        
        # 显示退化后的图像
        axes[i, 1].imshow(np.clip(degraded, 0, 1))
        axes[i, 1].set_title(f"Degraded: {deg_type}\nParams: {[f'{p:.2f}' for p in params if p != 0]}")
        axes[i, 1].axis('off')
        
        # 显示差异图
        axes[i, 2].imshow(diff.mean(axis=2), cmap='viridis') # 使用单通道可视化差异
        axes[i, 2].set_title(f"Difference (MAE: {diff.mean():.4f})")
        axes[i, 2].axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 调整布局以适应主标题
    
    # 确保输出目录存在
    output_dir = '../outputs/visualizations/'
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'data_augmentation_test.png')

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n可视化结果已保存到: {save_path}")
    plt.show()

if __name__ == '__main__':
    test_dataloader()
