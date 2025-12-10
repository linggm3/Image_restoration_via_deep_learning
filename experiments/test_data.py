# experiments/test_data.py 

import os
import sys
sys.path.append('..')

import yaml
import torch
import matplotlib.pyplot as plt
import numpy as np
from data.dataset import create_dataloaders
from collections import Counter

def test_dataloader():
    """测试数据加载器和退化链功能"""
    
    # 加载配置
    with open('../config/unet_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 为快速测试修改配置
    config['data']['batch_size'] = 4
    config['data']['num_workers'] = 0
    SUBSET_SIZE_FOR_TEST = 100 
    
    print("=" * 70)
    print("正在创建数据加载器（使用100个样本的子集进行测试）...")
    print("=" * 70)
    
    train_loader, val_loader, test_loader = create_dataloaders(
        config, 
        subset_size=SUBSET_SIZE_FOR_TEST
    )
    
    # 从训练数据加载器中获取多个批次进行统计
    print("\n" + "=" * 70)
    print("测试退化链功能 - 统计退化类型分布")
    print("=" * 70)
    
    num_batches_to_test = 1
    all_degradations = []
    
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= num_batches_to_test:
            break
            
        print(f"\n批次 {batch_idx + 1}:")
        print(f"  Degraded shape: {batch['degraded'].shape}")
        print(f"  Original shape: {batch['original'].shape}")
        print(f"  Condition shape: {batch['condition'].shape}")
        
        # 收集退化信息
        for i, info in enumerate(batch['degradation_info']):
            deg_names = info['degradation_names']
            num_degs = info['num_degradations']
            all_degradations.extend(deg_names)
            
            if i < 2:  # 只打印前2个样本的详细信息
                print(f"\n  样本 {i}:")
                print(f"    应用的退化数量: {num_degs}")
                print(f"    退化链: {' -> '.join(deg_names)}")
                for deg in info['applied_degradations']:
                    print(f"      - {deg['name']}: {deg['params']}")
    
    # 统计退化类型分布
    print("\n" + "=" * 70)
    print("退化类型统计（共 {} 个批次，{} 个样本）:".format(
        num_batches_to_test, 
        num_batches_to_test * config['data']['batch_size']
    ))
    print("=" * 70)
    
    degradation_counter = Counter(all_degradations)
    total_degradations = sum(degradation_counter.values())
    
    for deg_name, count in degradation_counter.most_common():
        percentage = (count / total_degradations) * 100
        print(f"  {deg_name:20s}: {count:3d} 次 ({percentage:5.1f}%)")
    
    # 可视化测试
    print("\n" + "=" * 70)
    print("生成可视化样本...")
    print("=" * 70)
    
    batch = next(iter(train_loader))
    visualize_batch(batch, num_samples=4, config=config)
    
    print("\n✅ 数据加载和退化链测试成功!")
    print("=" * 70)


def visualize_batch(batch, num_samples=4, config=None):
    """可视化一个batch的样本，展示退化效果"""
    
    num_samples = min(num_samples, batch['degraded'].shape[0])
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle("degrade", fontsize=18, fontweight='bold')
    
    for i in range(num_samples):
        # 转换为numpy并调整到[0, 1]
        degraded = batch['degraded'][i].permute(1, 2, 0).cpu().numpy()
        original = batch['original'][i].permute(1, 2, 0).cpu().numpy()
        
        # 计算差异图
        diff = np.abs(original - degraded)
        
        # 获取退化信息
        info = batch['degradation_info'][i]
        deg_names = info['degradation_names']
        num_degs = info['num_degradations']
        
        # 构造详细的退化描述
        deg_description = f"{num_degs}:\n"
        for deg in info['applied_degradations']:
            name = deg['name']
            params = deg['params']
            
            # 格式化参数显示
            if name == 'motion_blur':
                param_str = f"kernel={params['kernel_size']:.0f}, angle={params['angle']:.0f}°"
            elif name == 'gaussian_blur':
                param_str = f"σ={params['sigma']:.2f}"
            elif name == 'gaussian_noise':
                param_str = f"std={params['noise_std']:.1f}"
            elif name == 'jpeg_compression':
                param_str = f"quality={params['quality']:.0f}"
            elif name == 'downsampling':
                param_str = f"scale={params['scale']:.2f}"
            else:
                param_str = str(params)
            
            deg_description += f"  • {name.replace('_', ' ').title()}: {param_str}\n"
        
        # 显示原图
        axes[i, 0].imshow(np.clip(original, 0, 1))
        axes[i, 0].set_title("original\n", fontsize=12, fontweight='bold')
        axes[i, 0].axis('off')
        
        # 显示退化后的图像
        axes[i, 1].imshow(np.clip(degraded, 0, 1))
        #axes[i, 1].set_title(f"degrade\n{deg_description}", 
        #                    fontsize=10, ha='center')
        axes[i, 1].axis('off')
        
        # 显示差异图
        diff_mean = diff.mean(axis=2)
        im = axes[i, 2].imshow(diff_mean, cmap='hot', vmin=0, vmax=0.3)
        axes[i, 2].set_title(f"差异热力图\n平均差异: {diff.mean():.4f}", 
                            fontsize=12, fontweight='bold')
        axes[i, 2].axis('off')
        
        # 添加色条
        plt.colorbar(im, ax=axes[i, 2], fraction=0.046, pad=0.04)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # 保存结果
    output_dir = '../outputs/visualizations/'
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'degradation_chain_test.png')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 可视化结果已保存到: {save_path}")
    
    # 也尝试显示图像
    try:
        plt.show()
    except:
        print("   (无法显示图像窗口，但文件已保存)")
    finally:
        plt.close(fig)


def test_individual_degradations():
    """测试每种退化的独立效果"""
    print("\n" + "=" * 70)
    print("测试单个退化效果")
    print("=" * 70)
    
    from data.transforms import (
        MotionBlurTransform, 
        GaussianBlurTransform,
        GaussianNoiseTransform,
        JPEGCompressionTransform,
        DownsamplingTransform
    )
    from PIL import Image
    
    # 创建一个测试图像
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    transforms = {
        'Motion Blur': MotionBlurTransform(),
        'Gaussian Blur': GaussianBlurTransform(),
        'Gaussian Noise': GaussianNoiseTransform(),
        'JPEG Compression': JPEGCompressionTransform(),
        'Downsampling': DownsamplingTransform()
    }
    
    print("\n测试每种退化:")
    for name, transform in transforms.items():
        try:
            degraded, params = transform.apply(test_image.copy())
            print(f"  ✓ {name:20s}: 成功, 参数={params}")
        except Exception as e:
            print(f"  ✗ {name:20s}: 失败 - {e}")
    
    print("\n✅ 单个退化测试完成!")


if __name__ == '__main__':
    # 运行主测试
    test_dataloader()
    
    # 运行单个退化测试
    test_individual_degradations()
    
    print("\n" + "=" * 70)
    print("🎉 所有测试完成!")
    print("=" * 70)
