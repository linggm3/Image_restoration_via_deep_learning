# experiments/test_metrics.py
import sys
sys.path.append('..') # 将项目根目录添加到Python路径

import torch
from evaluation.metrics import calculate_psnr, calculate_ssim, calculate_lpips, Evaluator

def test_metrics():
    """测试评估指标函数"""
    print("=" * 50)
    print("正在测试评估指标...")
    
    # 创建两个随机的图像张量 (C, H, W)
    img1 = torch.rand(3, 256, 256)
    img2 = torch.rand(3, 256, 256)
    img3 = img1.clone() # 创建一个完全相同的图像
    
    # 测试单个函数
    psnr_val = calculate_psnr(img1, img2)
    ssim_val = calculate_ssim(img1, img2)
    lpips_val = calculate_lpips(img1, img2)
    
    print(f"随机图像 vs 随机图像:")
    print(f"  PSNR: {psnr_val:.4f}")
    print(f"  SSIM: {ssim_val:.4f}")
    print(f"  LPIPS: {lpips_val:.4f}")

    psnr_same = calculate_psnr(img1, img3)
    ssim_same = calculate_ssim(img1, img3)
    lpips_same = calculate_lpips(img1, img3)

    print(f"\n相同图像 vs 相同图像:")
    print(f"  PSNR: {psnr_same:.4f} (期望值: inf)")
    print(f"  SSIM: {ssim_same:.4f} (期望值: 1.0)")
    print(f"  LPIPS: {lpips_same:.4f} (期望值: 0.0)")

    # 测试评估器类
    print("\n测试Evaluator类...")
    evaluator = Evaluator(metrics=['psnr', 'ssim', 'lpips'])
    pred_batch = torch.rand(4, 3, 64, 64)
    target_batch = torch.rand(4, 3, 64, 64)
    evaluator.update(pred_batch, target_batch)
    results = evaluator.get_mean_results()
    print(f"批次评估结果: {results}")
    
    print("\n✓ 评估指标测试成功!")
    print("=" * 50)

if __name__ == '__main__':
    test_metrics()
