# experiments/evaluation_unet.py

import sys
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from collections import defaultdict

sys.path.append('..')

from data.dataset import create_dataloaders
from models.unet import UNet, ConditionalUNet
from evaluation.metrics import Evaluator, calculate_psnr, calculate_ssim, calculate_lpips

def evaluate_model(config, model_path):
    """
    在测试集上评估模型并可视化结果
    支持条件和非条件模型
    """
    print("=" * 70)
    print("🔍 开始模型评估")
    print("=" * 70)
    
    # 1. 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ 使用设备: {device}")

    # 2. 创建测试数据加载器
    print("\n📊 正在创建测试数据加载器...")
    config['data']['batch_size'] = 16
    _, _, test_loader = create_dataloaders(config)
    print(f"✓ 测试集样本总数: {len(test_loader.dataset)}")

    # 3. 初始化模型并加载权重
    print(f"\n🏗️  正在初始化模型...")
    use_conditional = config['model'].get('use_conditional', False)
    
    if use_conditional:
        print("✓ 使用条件U-Net模型（使用退化信息指导恢复）")
        model = ConditionalUNet(
            in_channels=config['model']['in_channels'],
            out_channels=config['model']['out_channels'],
            condition_dim=config['model']['condition_dim'],
            base_channels=config['model']['base_channels'],
            bilinear=True
        ).to(device)
    else:
        print("✓ 使用基础U-Net模型（盲恢复模式）")
        model = UNet(
            in_channels=config['model']['in_channels'],
            out_channels=config['model']['out_channels'],
            base_channels=config['model']['base_channels'],
            bilinear=True
        ).to(device)
    
    print(f"\n📂 正在从 '{model_path}' 加载模型权重...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✓ 模型加载成功！")
    
    if 'epoch' in checkpoint:
        print(f"  训练轮次: {checkpoint['epoch'] + 1}")
    if 'best_val_metric' in checkpoint:
        print(f"  最佳验证LPIPS: {checkpoint['best_val_metric']:.4f}")

    # 4. 初始化评估器
    print("\n📈 初始化评估指标...")
    metrics = config['evaluation']['metrics']
    print(f"  评估指标: {', '.join([m.upper() for m in metrics])}")
    
    degraded_evaluator = Evaluator(metrics=metrics)
    restored_evaluator = Evaluator(metrics=metrics)
    
    # 按退化类型分别统计
    degradation_evaluators = defaultdict(lambda: {
        'degraded': Evaluator(metrics=metrics),
        'restored': Evaluator(metrics=metrics)
    })
    
    # 收集可视化样本
    vis_samples = defaultdict(list)
    max_samples_per_type = 5

    # 5. 在测试集上进行评估
    print("\n🧪 开始在测试集上进行评估...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            degraded = batch['degraded'].to(device)
            original = batch['original'].to(device)
            
            # 模型推理
            if use_conditional:
                condition = batch['condition'].to(device)
                restored = model(degraded, condition)
            else:
                restored = model(degraded)
            
            # 全局评估
            degraded_evaluator.update(degraded, original)
            restored_evaluator.update(restored, original)
            
            # 按退化类型评估
            for i in range(degraded.size(0)):
                info = batch['degradation_info'][i]
                deg_names = info['degradation_names']
                
                # 为每种应用的退化类型更新统计
                for deg_name in deg_names:
                    degradation_evaluators[deg_name]['degraded'].update(
                        degraded[i:i+1], original[i:i+1]
                    )
                    degradation_evaluators[deg_name]['restored'].update(
                        restored[i:i+1], original[i:i+1]
                    )
                
                # 收集可视化样本
                primary_deg = deg_names[0] if deg_names else 'unknown'
                if len(vis_samples[primary_deg]) < max_samples_per_type:
                    vis_samples[primary_deg].append({
                        'degraded': degraded[i].cpu(),
                        'original': original[i].cpu(),
                        'restored': restored[i].cpu(),
                        'info': info
                    })

    # 6. 打印全局评估结果
    degraded_metrics = degraded_evaluator.get_mean_results()
    restored_metrics = restored_evaluator.get_mean_results()

    print("\n" + "=" * 70)
    print("📊 全局评估结果")
    print("=" * 70)
    print(f"{'指标':<12} | {'退化图 vs 原图':<20} | {'复原图 vs 原图':<20} | {'改善':<10}")
    print("-" * 70)
    
    for metric in metrics:
        deg_val = degraded_metrics.get(metric, 0.0)
        res_val = restored_metrics.get(metric, 0.0)
        
        # 计算改善（LPIPS越小越好，PSNR/SSIM越大越好）
        if metric == 'lpips':
            improvement = ((deg_val - res_val) / deg_val * 100) if deg_val > 0 else 0
            improvement_str = f"{improvement:+.1f}%"
        else:
            improvement = ((res_val - deg_val) / deg_val * 100) if deg_val > 0 else 0
            improvement_str = f"{improvement:+.1f}%"
        
        print(f"{metric.upper():<12} | {deg_val:<20.4f} | {res_val:<20.4f} | {improvement_str:<10}")
    
    print("=" * 70)
    
    # 7. 打印按退化类型的评估结果
    if degradation_evaluators:
        print("\n" + "=" * 70)
        print("📊 按退化类型的评估结果")
        print("=" * 70)
        
        for deg_name in sorted(degradation_evaluators.keys()):
            evaluators = degradation_evaluators[deg_name]
            deg_metrics = evaluators['degraded'].get_mean_results()
            res_metrics = evaluators['restored'].get_mean_results()
            
            print(f"\n{deg_name.replace('_', ' ').title()}:")
            print(f"  {'PSNR':<8}: {deg_metrics.get('psnr', 0):.2f} → {res_metrics.get('psnr', 0):.2f}")
            print(f"  {'SSIM':<8}: {deg_metrics.get('ssim', 0):.4f} → {res_metrics.get('ssim', 0):.4f}")
            print(f"  {'LPIPS':<8}: {deg_metrics.get('lpips', 0):.4f} → {res_metrics.get('lpips', 0):.4f}")
        
        print("=" * 70)
    
    # 8. 可视化结果
    print("\n🎨 正在生成可视化结果...")
    visualize_results(vis_samples, config, restored_metrics)
    
    # 9. 保存评估报告
    save_evaluation_report(
        config, 
        degraded_metrics, 
        restored_metrics, 
        degradation_evaluators
    )
    
    print("\n✅ 评估完成！")


def visualize_results(vis_samples, config, overall_metrics):
    """生成可视化结果"""
    base_output_dir = f"../outputs/{config['model']['name']}/visualizations/"
    os.makedirs(base_output_dir, exist_ok=True)
    
    for deg_type, samples in vis_samples.items():
        if not samples:
            continue
        
        num_samples = len(samples)
        fig, axes = plt.subplots(num_samples, 3, figsize=(18, 5 * num_samples))
        
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(
            f"评估结果 - {deg_type.replace('_', ' ').title()}\n"
            f"全局PSNR: {overall_metrics.get('psnr', 0):.2f} | "
            f"SSIM: {overall_metrics.get('ssim', 0):.4f} | "
            f"LPIPS: {overall_metrics.get('lpips', 0):.4f}",
            fontsize=16,
            fontweight='bold'
        )
        
        for i, sample in enumerate(samples):
            original = sample['original']
            degraded = sample['degraded']
            restored = sample['restored']
            info = sample['info']
            
            # 计算单样本指标
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
            
            # 准备图像
            original_np = original.permute(1, 2, 0).numpy()
            degraded_np = degraded.permute(1, 2, 0).numpy()
            restored_np = restored.permute(1, 2, 0).numpy()
            
            # 构建退化描述
            deg_desc = "退化链: " + " → ".join(info['degradation_names'])
            
            # 原图
            axes[i, 0].imshow(np.clip(original_np, 0, 1))
            axes[i, 0].set_title("原始图像", fontsize=12, fontweight='bold')
            axes[i, 0].axis('off')
            
            # 退化图
            title_deg = (f"退化图像\n{deg_desc}\n"
                        f"PSNR: {metrics_deg['psnr']:.2f} | "
                        f"SSIM: {metrics_deg['ssim']:.4f} | "
                        f"LPIPS: {metrics_deg['lpips']:.4f}")
            axes[i, 1].imshow(np.clip(degraded_np, 0, 1))
            axes[i, 1].set_title(title_deg, fontsize=10)
            axes[i, 1].axis('off')
            
            # 复原图
            title_res = (f"复原图像\n"
                        f"PSNR: {metrics_res['psnr']:.2f} (↑{metrics_res['psnr']-metrics_deg['psnr']:+.2f}) | "
                        f"SSIM: {metrics_res['ssim']:.4f} (↑{metrics_res['ssim']-metrics_deg['ssim']:+.4f}) | "
                        f"LPIPS: {metrics_res['lpips']:.4f} (↓{metrics_deg['lpips']-metrics_res['lpips']:+.4f})")
            axes[i, 2].imshow(np.clip(restored_np, 0, 1))
            axes[i, 2].set_title(title_res, fontsize=10)
            axes[i, 2].axis('off')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        save_path = os.path.join(base_output_dir, f'eval_results_{deg_type}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ '{deg_type}' 结果保存到: {save_path}")


def save_evaluation_report(config, degraded_metrics, restored_metrics, degradation_evaluators):
    """保存评估报告到文本文件"""
    report_dir = f"../outputs/{config['model']['name']}/reports/"
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, 'evaluation_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("模型评估报告\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"模型名称: {config['model']['name']}\n")
        f.write(f"图像尺寸: {config['data']['image_size']}x{config['data']['image_size']}\n")
        f.write(f"使用条件模型: {config['model'].get('use_conditional', False)}\n\n")
        
        f.write("全局评估结果:\n")
        f.write("-" * 70 + "\n")
        for metric in config['evaluation']['metrics']:
            deg_val = degraded_metrics.get(metric, 0.0)
            res_val = restored_metrics.get(metric, 0.0)
            f.write(f"{metric.upper():<12}: {deg_val:.4f} → {res_val:.4f}\n")
        
        f.write("\n按退化类型的评估结果:\n")
        f.write("-" * 70 + "\n")
        for deg_name in sorted(degradation_evaluators.keys()):
            evaluators = degradation_evaluators[deg_name]
            deg_metrics = evaluators['degraded'].get_mean_results()
            res_metrics = evaluators['restored'].get_mean_results()
            
            f.write(f"\n{deg_name.replace('_', ' ').title()}:\n")
            for metric in config['evaluation']['metrics']:
                deg_val = deg_metrics.get(metric, 0.0)
                res_val = res_metrics.get(metric, 0.0)
                f.write(f"  {metric.upper():<8}: {deg_val:.4f} → {res_val:.4f}\n")
        
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"\n✓ 评估报告保存到: {report_path}")


if __name__ == '__main__':
    try:
        with open('../config/unet_config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("✓ 配置文件加载成功")
    except FileNotFoundError:
        print("✗ 错误：无法找到配置文件 '../config/unet_config.yaml'")
        sys.exit(1)
    
    checkpoint_dir = f"../outputs/{config['model']['name']}/checkpoints/"
    model_path = os.path.join(checkpoint_dir, 'best.pth')
    
    if not os.path.exists(model_path):
        print(f"✗ 错误：在 '{model_path}' 未找到模型权重")
        print(f"请先运行 'python train_unet.py' 训练模型")
        sys.exit(1)
    
    evaluate_model(config, model_path)
