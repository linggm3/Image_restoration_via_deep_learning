# experiments/evaluate_unet.py

import sys
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import operator

# 将项目根目录添加到Python路径
sys.path.append('..')

from data.dataset import create_dataloaders
from models.unet import UNet
from evaluation.metrics import Evaluator, calculate_psnr, calculate_ssim, calculate_lpips

def get_degradation_severity(deg_type, params):
    """根据退化类型和参数计算严重程度得分"""
    if deg_type == 'gaussian_blur':
        # sigma值越大越严重
        return params[1]  # sigma 在条件向量的第二个位置
    elif deg_type == 'gamma_correction':
        # gamma值离1.0越远越严重
        return abs(params[1] - 1.0) # gamma 在第二个位置
    elif deg_type == 'histogram_equalization':
        # 没有严重程度之分
        return 0
    return 0

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
    
    # --- CHANGE: 收集所有样本及其严重程度 ---
    all_vis_samples = {
        'gaussian_blur': [],
        'histogram_equalization': [],
        'gamma_correction': []
    }

    # 5. 在测试集上进行评估
    print("\n开始在测试集上进行评估...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            degraded = batch['degraded'].to(device)
            original = batch['original'].to(device)
            
            restored = model(degraded)
            
            degraded_evaluator.update(degraded, original)
            restored_evaluator.update(restored, original)
            
            # --- CHANGE: 收集所有样本用于后续筛选 ---
            for i in range(degraded.size(0)):
                deg_type = batch['degradation_type'][i]
                params = batch['degradation_params'][i]
                if deg_type in all_vis_samples:
                    severity = get_degradation_severity(deg_type, params)
                    all_vis_samples[deg_type].append({
                        'degraded': degraded[i].cpu(),
                        'original': original[i].cpu(),
                        'restored': restored[i].cpu(),
                        'params': params,
                        'severity': severity
                    })

    # --- CHANGE: 筛选出要可视化的最终样本 ---
    final_vis_samples = {}
    num_vis_per_type = 5

    for deg_type, samples in all_vis_samples.items():
        if not samples:
            continue
        # 按严重程度从高到低排序
        samples.sort(key=operator.itemgetter('severity'), reverse=True)
        # 选取最严重的N个样本
        final_vis_samples[deg_type] = samples[:num_vis_per_type]

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
    visualize_final_samples(final_vis_samples, config)


def visualize_final_samples(samples_dict, config):
    """为筛选出的最终样本生成可视化图"""
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
            
        fig.suptitle(f"U-Net Test Results for: {deg_type.replace('_', ' ').title()} (Medium to Severe)", fontsize=20)
        
        for i, sample in enumerate(samples):
            original = sample['original']
            degraded = sample['degraded']
            restored = sample['restored']
            
            # 单独计算指标
            metrics_deg = {
                'psnr': calculate_psnr(degraded, original),
                'ssim': calculate_ssim(degraded, original),
                'lpips': calculate_lpips(degraded, original)
            }
            metrics_res = {
                'psnr': calculate_psnr(restored, original),
                'ssim': calculate_ssim(restored, original),
                'lpips': calculate_lpips(restored, original)
            }
            
            # 准备图像用于显示
            original_np = original.permute(1, 2, 0).numpy()
            degraded_np = degraded.permute(1, 2, 0).numpy()
            restored_np = restored.permute(1, 2, 0).numpy()

            # 原图
            axes[i, 0].imshow(np.clip(original_np, 0, 1))
            axes[i, 0].set_title("Original")
            axes[i, 0].axis('off')
            
            # --- CHANGE: 更新标题以包含退化参数 ---
            param_str = ""
            if deg_type == 'gaussian_blur':
                param_str = f" (sigma={sample['params'][1]:.2f})"
            elif deg_type == 'gamma_correction':
                param_str = f" (gamma={sample['params'][1]:.2f})"

            # 退化图
            title_deg = (f"Degraded{param_str}\n"
                         f"PSNR: {metrics_deg['psnr']:.2f}, SSIM: {metrics_deg['ssim']:.4f}, LPIPS: {metrics_deg['lpips']:.4f}")
            axes[i, 1].imshow(np.clip(degraded_np, 0, 1))
            axes[i, 1].set_title(title_deg)
            axes[i, 1].axis('off')
            
            # 复原图
            title_res = (f"Restored by U-Net\n"
                         f"PSNR: {metrics_res['psnr']:.2f}, SSIM: {metrics_res['ssim']:.4f}, LPIPS: {metrics_res['lpips']:.4f}")
            axes[i, 2].imshow(np.clip(restored_np, 0, 1))
            axes[i, 2].set_title(title_res)
            axes[i, 2].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        
        save_path = os.path.join(base_output_dir, f'test_results_{deg_type}_severe.png')
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"'{deg_type}' 的严重退化样本可视化结果已保存到: {save_path}")


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
