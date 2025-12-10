# experiments/visualize_degradations.py

"""
可视化所有退化类型的效果
用于理解和调试退化参数
"""

import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

from data.transforms import (
    MotionBlurTransform,
    GaussianBlurTransform,
    GaussianNoiseTransform,
    JPEGCompressionTransform,
    DownsamplingTransform
)

def create_test_image(size=256):
    """创建一个测试图像（棋盘格和渐变）"""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    
    # 棋盘格背景
    square_size = size // 8
    for i in range(8):
        for j in range(8):
            if (i + j) % 2 == 0:
                img[i*square_size:(i+1)*square_size, 
                    j*square_size:(j+1)*square_size] = 200
    
    # 添加彩色渐变
    gradient = np.linspace(0, 255, size).astype(np.uint8)
    img[size//4:3*size//4, size//4:3*size//4, 0] = np.tile(gradient, (size//2, 1))[:size//2, :size//2]
    img[size//4:3*size//4, size//4:3*size//4, 1] = np.tile(gradient.reshape(-1, 1), (1, size//2))[:size//2, :size//2]
    img[size//4:3*size//4, size//4:3*size//4, 2] = 128
    
    return img


def visualize_all_degradations():
    """可视化所有退化类型的效果"""
    print("=" * 70)
    print("🎨 可视化所有退化类型")
    print("=" * 70)
    
    # 创建测试图像
    print("\n✓ 创建测试图像...")
    test_image = create_test_image(256)
    
    # 定义所有退化及其参数
    degradations = [
        {
            'name': 'Motion Blur',
            'transform': MotionBlurTransform(kernel_size_range=[15, 15], angle_range=[45, 45]),
            'description': 'Kernel=15, Angle=45°'
        },
        {
            'name': 'Gaussian Blur',
            'transform': GaussianBlurTransform(sigma_range=[4.0, 4.0]),
            'description': 'σ=4.0'
        },
        {
            'name': 'Gaussian Noise',
            'transform': GaussianNoiseTransform(noise_std_range=[30, 30]),
            'description': 'Std=30'
        },
        {
            'name': 'JPEG Compression',
            'transform': JPEGCompressionTransform(quality_range=[20, 20]),
            'description': 'Quality=20'
        },
        {
            'name': 'Downsampling',
            'transform': DownsamplingTransform(scale_range=[0.5, 0.5]),
            'description': 'Scale=0.5'
        }
    ]
    
    # 创建可视化
    num_degs = len(degradations)
    fig, axes = plt.subplots(2, num_degs + 1, figsize=(4 * (num_degs + 1), 8))
    
    fig.suptitle('退化类型效果对比', fontsize=20, fontweight='bold', y=0.98)
    
    # 第一行：原图和各种退化
    axes[0, 0].imshow(test_image)
    axes[0, 0].set_title('原始图像', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    degraded_images = []
    
    for i, deg_info in enumerate(degradations):
        print(f"✓ 应用 {deg_info['name']}...")
        degraded, params = deg_info['transform'].apply(test_image.copy())
        degraded_images.append(degraded)
        
        axes[0, i + 1].imshow(degraded)
        axes[0, i + 1].set_title(
            f"{deg_info['name']}\n{deg_info['description']}", 
            fontsize=12
        )
        axes[0, i + 1].axis('off')
    
    # 第二行：差异图
    axes[1, 0].text(0.5, 0.5, '差异热力图\n(相对原图)', 
                     ha='center', va='center', fontsize=14, fontweight='bold',
                     transform=axes[1, 0].transAxes)
    axes[1, 0].axis('off')
    
    for i, degraded in enumerate(degraded_images):
        # 计算差异
        diff = np.abs(test_image.astype(float) - degraded.astype(float)).mean(axis=2)
        
        im = axes[1, i + 1].imshow(diff, cmap='hot', vmin=0, vmax=100)
        axes[1, i + 1].set_title(f'平均差异: {diff.mean():.2f}', fontsize=11)
        axes[1, i + 1].axis('off')
        
        # 添加色条
        plt.colorbar(im, ax=axes[1, i + 1], fraction=0.046, pad=0.04)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    
    # 保存结果
    output_dir = '../outputs/visualizations/'
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'degradation_types_comparison.png')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ 可视化结果已保存到: {save_path}")
    
    try:
        plt.show()
    except:
        print("  (无法显示图像窗口，但文件已保存)")
    finally:
        plt.close()


def visualize_degradation_severity():
    """可视化不同严重程度的退化"""
    print("\n" + "=" * 70)
    print("🎨 可视化退化严重程度")
    print("=" * 70)
    
    test_image = create_test_image(256)
    
    # 定义不同严重程度的参数
    severity_configs = [
        {
            'name': 'Motion Blur',
            'severities': [
                ('轻微', MotionBlurTransform(kernel_size_range=[5, 5], angle_range=[0, 0])),
                ('中等', MotionBlurTransform(kernel_size_range=[15, 15], angle_range=[45, 45])),
                ('严重', MotionBlurTransform(kernel_size_range=[25, 25], angle_range=[90, 90]))
            ]
        },
        {
            'name': 'Gaussian Blur',
            'severities': [
                ('轻微', GaussianBlurTransform(sigma_range=[1.0, 1.0])),
                ('中等', GaussianBlurTransform(sigma_range=[4.0, 4.0])),
                ('严重', GaussianBlurTransform(sigma_range=[8.0, 8.0]))
            ]
        },
        {
            'name': 'Gaussian Noise',
            'severities': [
                ('轻微', GaussianNoiseTransform(noise_std_range=[10, 10])),
                ('中等', GaussianNoiseTransform(noise_std_range=[30, 30])),
                ('严重', GaussianNoiseTransform(noise_std_range=[50, 50]))
            ]
        }
    ]
    
    for config in severity_configs:
        print(f"\n✓ 处理 {config['name']}...")
        
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        fig.suptitle(f"{config['name']} - 不同严重程度", fontsize=16, fontweight='bold')
        
        # 原图
        axes[0].imshow(test_image)
        axes[0].set_title('原始图像', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # 不同严重程度
        for i, (severity_name, transform) in enumerate(config['severities']):
            degraded, params = transform.apply(test_image.copy())
            axes[i + 1].imshow(degraded)
            axes[i + 1].set_title(f'{severity_name}\n{params}', fontsize=11)
            axes[i + 1].axis('off')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # 保存
        output_dir = '../outputs/visualizations/'
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(
            output_dir, 
            f'severity_{config["name"].lower().replace(" ", "_")}.png'
        )
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ 保存到: {save_path}")
        plt.close()


def visualize_degradation_chain():
    """可视化退化链的效果"""
    print("\n" + "=" * 70)
    print("🎨 可视化退化链效果")
    print("=" * 70)
    
    test_image = create_test_image(256)
    
    # 定义退化链
    chain_steps = [
        ('原图', None, test_image.copy()),
        ('运动模糊', MotionBlurTransform(kernel_size_range=[15, 15], angle_range=[45, 45]), None),
        ('+ 高斯噪声', GaussianNoiseTransform(noise_std_range=[20, 20]), None),
        ('+ JPEG压缩', JPEGCompressionTransform(quality_range=[30, 30]), None),
        ('+ 下采样', DownsamplingTransform(scale_range=[0.6, 0.6]), None)
    ]
    
    # 应用退化链
    print("✓ 应用退化链...")
    current_image = test_image.copy()
    
    for i in range(1, len(chain_steps)):
        step_name, transform, _ = chain_steps[i]
        if transform is not None:
            current_image, params = transform.apply(current_image)
            chain_steps[i] = (step_name, transform, current_image.copy())
            print(f"  {i}. {step_name}: {params}")
    
    # 可视化
    num_steps = len(chain_steps)
    fig, axes = plt.subplots(1, num_steps, figsize=(4 * num_steps, 4))
    
    fig.suptitle('退化链效果演示', fontsize=18, fontweight='bold', y=0.98)
    
    for i, (step_name, _, image) in enumerate(chain_steps):
        if i == 0:
            axes[i].imshow(image)
            axes[i].set_title(step_name, fontsize=12, fontweight='bold')
        else:
            axes[i].imshow(image)
            diff = np.abs(test_image.astype(float) - image.astype(float)).mean()
            axes[i].set_title(f'{step_name}\n平均差异: {diff:.2f}', fontsize=11)
        axes[i].axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 保存
    output_dir = '../outputs/visualizations/'
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'degradation_chain_demo.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ 退化链可视化保存到: {save_path}")
    
    try:
        plt.show()
    except:
        print("  (无法显示图像窗口，但文件已保存)")
    finally:
        plt.close()


if __name__ == '__main__':
    print("\n" + "🎨" * 35)
    print("退化效果可视化工具")
    print("🎨" * 35)
    
    try:
        # 运行所有可视化
        visualize_all_degradations()
        visualize_degradation_severity()
        visualize_degradation_chain()
        
        print("\n" + "=" * 70)
        print("✅ 所有可视化完成!")
        print("=" * 70)
        print("\n💡 提示:")
        print("  - 查看 ../outputs/visualizations/ 目录中的所有生成图像")
        print("  - 根据可视化结果调整配置文件中的退化参数")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ 可视化过程出错: {e}")
        import traceback
        traceback.print_exc()
