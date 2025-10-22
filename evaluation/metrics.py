# evaluation/metrics.py

import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as ski_psnr
from skimage.metrics import structural_similarity as ski_ssim
import lpips

# 使用GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lpips_alex = lpips.LPIPS(net='alex').to(device)

def calculate_psnr(img1, img2, data_range=1.0):
    """
    计算两张图像之间的PSNR值。
    Args:
        img1 (torch.Tensor or np.ndarray): 图像1, 值域 [0, 1]。
        img2 (torch.Tensor or np.ndarray): 图像2, 值域 [0, 1]。
        data_range (float): 图像数据范围。
    Returns:
        float: PSNR值。
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.permute(1, 2, 0).cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.permute(1, 2, 0).cpu().numpy()
        
    return ski_psnr(img1, img2, data_range=data_range)

def calculate_ssim(img1, img2, data_range=1.0, multichannel=True):
    """
    计算两张图像之间的SSIM值。
    Args:
        img1 (torch.Tensor or np.ndarray): 图像1, 值域 [0, 1]。
        img2 (torch.Tensor or np.ndarray): 图像2, 值域 [0, 1]。
        data_range (float): 图像数据范围。
        multichannel (bool): 是否为多通道图像。
    Returns:
        float: SSIM值。
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.permute(1, 2, 0).cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.permute(1, 2, 0).cpu().numpy()
        
    # 对于 skimage >= 0.19, multichannel 参数应改为 channel_axis=-1
    try:
        return ski_ssim(img1, img2, data_range=data_range, channel_axis=-1, win_size=7)
    except TypeError:
        # 兼容旧版本
        return ski_ssim(img1, img2, data_range=data_range, multichannel=True, win_size=7)


def calculate_lpips(img1, img2):
    """
    计算两张图像之间的LPIPS值。
    Args:
        img1 (torch.Tensor): 图像1, (C, H, W), 值域 [0, 1]。
        img2 (torch.Tensor): 图像2, (C, H, W), 值域 [0, 1]。
    Returns:
        float: LPIPS值。
    """
    # LPIPS期望输入范围在[-1, 1]
    img1_norm = img1 * 2 - 1
    img2_norm = img2 * 2 - 1
    
    # 增加batch维度并移动到设备
    img1_norm = img1_norm.unsqueeze(0).to(device)
    img2_norm = img2_norm.unsqueeze(0).to(device)
    
    with torch.no_grad():
        dist = lpips_alex.forward(img1_norm, img2_norm)
        
    return dist.item()

class Evaluator:
    """
    评估器类，用于批量计算和管理多个指标。
    """
    def __init__(self, metrics=['psnr', 'ssim', 'lpips']):
        self.metrics = metrics
        self.results = {metric: [] for metric in self.metrics}

    def update(self, pred_batch, target_batch):
        """
        用一个批次的数据更新评估结果。
        Args:
            pred_batch (torch.Tensor): 预测图像批次, (B, C, H, W), 值域 [0, 1]。
            target_batch (torch.Tensor): 真实图像批次, (B, C, H, W), 值域 [0, 1]。
        """
        pred_batch = torch.clamp(pred_batch, 0, 1)
        
        for i in range(pred_batch.shape[0]):
            pred_img = pred_batch[i]
            target_img = target_batch[i]
            
            if 'psnr' in self.metrics:
                self.results['psnr'].append(calculate_psnr(pred_img, target_img))
            if 'ssim' in self.metrics:
                self.results['ssim'].append(calculate_ssim(pred_img, target_img))
            if 'lpips' in self.metrics:
                self.results['lpips'].append(calculate_lpips(pred_img, target_img))

    def get_mean_results(self):
        """
        计算所有指标的平均值。
        Returns:
            dict: 包含各指标平均值的字典。
        """
        mean_results = {}
        for metric, values in self.results.items():
            if not values:
                mean_results[metric] = 0
                continue
            
            if metric == 'psnr':
                # --- FIX: 计算平均PSNR时，排除inf值 ---
                finite_values = [v for v in values if np.isfinite(v)]
                if finite_values:
                    mean_results[metric] = np.mean(finite_values)
                else:
                    # 如果所有值都是inf，则结果也为inf
                    mean_results[metric] = float('inf')
            else:
                mean_results[metric] = np.mean(values)
                
        return mean_results

    def reset(self):
        """重置所有结果。"""
        self.results = {metric: [] for metric in self.metrics}
