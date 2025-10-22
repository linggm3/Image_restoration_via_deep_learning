# experiments/evaluate_unet.py

import sys
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# 将项目根目录添加到Python路径
sys.path.append('..')

from data.dataset import create_dataloaders
from models.unet import UNet
from evaluation.metrics import Evaluator, calculate_psnr, calculate_ssim, calculate_lpips

def evaluate_model(config, model_path):
    """
    在测试集上评估模型并可视化结果
    """
    # 1. 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"将使用设备: {device}")

    # 2. 创建测试数据加载器
    print("正在创建测试数据加载器...")
    config['data']['batch_size'] = 16 
    _, _, test_loader = create_dataloaders(config)
    print(f"测试集样本总数: {len(test_loader.dataset)}")

    # 3. 初始化模型并加载权重
    print("正在初始化U-Net模型...")
    model = UNet(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        base_channels=config['model']['base_channels'],
        bilinear=True
    ).to(device)
    
    print(f"正在从 '{model_path}' 加载模型权重...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("模型加载成功！")

    # 4. 初始化评估器
    metrics = config['evaluation']['metrics']
    degraded_evaluator = Evaluator(metrics=metrics)
    restored_evaluator = Evaluator(metrics=metrics)
    
    # --- CHANGE: 按退化类型收集可视化样本 ---
    vis_samples = {
        'gaussian_blur': [],
        'histogram_equalization': [],
        'gamma_correction': []
    }
    num_vis_per_type = 5

    # 5. 在测试集上进行评估
    print("\n开始在测试集上进行评估...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            degraded = batch['degraded'].to(device)
            original = batch['original'].to(device)
            
            restored = model(degraded)
            
            degraded_evaluator.update(degraded, original)
            restored_evaluator.update(restored, original)
            
            # --- CHANGE: 收集特定类型的样本 ---
            for i in range(degraded.size(0)):
                deg_type = batch['degradation_type'][i]
                if deg_type in vis_samples and len(vis_samples[deg_type]) < num_vis_per_type:
                    # 为这个样本单独计算所有指标
                    original_s = original[i]
                    degraded_s = degraded[i]
                    restored_s = restored[i]

                    metrics_deg = {
                        'psnr': calculate_psnr(degraded_s, original_s),
                        'ssim': calculate_ssim(degraded_s, original_s),
                        'lpips': calculate_lpips(degraded_s, original_s)
                    }
                    metrics_res = {
                        'psnr': calculate_psnr(restored_s, original_s),
                        'ssim': calculate_ssim(restored_s, original_s),
                        'lpips': calculate_lpips(restored_s, original_s)
                    }

                    vis_samples[deg_type].append({
                        'degraded': degraded_s.cpu(),
                        'original': original_s.cpu(),
                        'restored': restored_s.cpu(),
                        'params': batch['degradation_params'][i],
                        'metrics_deg': metrics_deg,
                        'metrics_res': metrics_res
                    })

    # 6. 打印最终评估结果
    degraded_metrics = degraded_evaluator.get_mean_results()
    restored_metrics = restored_evaluator.get_mean_results()

    print("\n================ 评估结果 ================")
    print(f"{'Metric':<10} | {'Degraded vs Original':<25} | {'Restored vs Original':<25}")
    print("-" * 65)
    for metric in metrics:
        deg_val = degraded_metrics.get(metric, 0.0)
        res_val = restored_metrics.get(metric, 0.0)
        print(f"{metric.upper():<10} | {deg_val:<25.4f} | {res_val:<25.4f}")
    print("==========================================")
    
    # 7. 可视化结果
    print("\n正在生成可视化结果...")
    visualize_results_per_type(vis_samples, config)


def visualize_results_per_type(samples_dict, config):
    """为每种退化类型生成单独的可视化图"""
    base_output_dir = f"../outputs/{config['model']['name']}/visualizations/"
    os.makedirs(base_output_dir, exist_ok=True)

    for deg_type, samples in samples_dict.items():
        if not samples:
            print(f"未找到 '{deg_type}' 类型的样本进行可视化。")
            continue

        num_samples = len(samples)
        fig, axes = plt.subplots(num_samples, 3, figsize=(18, 5 * num_samples))
        
        if num_samples == 1:
            axes = axes.reshape(1, -1)
            
        fig.suptitle(f"U-Net Test Results for: {deg_type.replace('_', ' ').title()}", fontsize=20)
        
        for i, sample in enumerate(samples):
            original = sample['original'].permute(1, 2, 0).numpy()
            degraded = sample['degraded'].permute(1, 2, 0).numpy()
            restored = sample['restored'].permute(1, 2, 0).numpy()
            
            metrics_deg = sample['metrics_deg']
            metrics_res = sample['metrics_res']

            # 原图
            axes[i, 0].imshow(np.clip(original, 0, 1))
            axes[i, 0].set_title("Original")
            axes[i, 0].axis('off')
            
            # 退化图
            title_deg = (f"Degraded\n"
                         f"PSNR: {metrics_deg['psnr']:.2f}, SSIM: {metrics_deg['ssim']:.4f}, LPIPS: {metrics_deg['lpips']:.4f}")
            axes[i, 1].imshow(np.clip(degraded, 0, 1))
            axes[i, 1].set_title(title_deg)
            axes[i, 1].axis('off')
            
            # 复原图
            title_res = (f"Restored by U-Net\n"
                         f"PSNR: {metrics_res['psnr']:.2f}, SSIM: {metrics_res['ssim']:.4f}, LPIPS: {metrics_res['lpips']:.4f}")
            axes[i, 2].imshow(np.clip(restored, 0, 1))
            axes[i, 2].set_title(title_res)
            axes[i, 2].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        
        save_path = os.path.join(base_output_dir, f'test_results_{deg_type}.png')
        plt.savefig(save_path, dpi=150)
        plt.close(fig) # 关闭图形，防止在内存中累积
        print(f"'{deg_type}' 的可视化结果已保存到: {save_path}")


if __name__ == '__main__':
    try:
        with open('../config/unet_config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("配置文件 'unet_config.yaml' 加载成功。")
    except FileNotFoundError:
        print("错误：无法找到配置文件 '../config/unet_config.yaml'。")
        sys.exit(1)
        
    config['model']['name'] = "unet_augmented"
    checkpoint_dir = f"../outputs/{config['model']['name']}/checkpoints/"
    model_path = os.path.join(checkpoint_dir, 'best.pth')

    if not os.path.exists(model_path):
        print(f"错误：在 '{model_path}' 未找到模型权重。")
        sys.exit(1)

    evaluate_model(config, model_path)
