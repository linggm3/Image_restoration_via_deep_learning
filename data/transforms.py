# data/transforms.py

import torch
import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image
import cv2

class DegradationTransform:
    """退化算子基类"""
    def __init__(self):
        self.name = "base"
        
    def apply(self, image):
        """
        应用退化
        Args:
            image: PIL Image或numpy array (H, W, C), RGB格式, 值域[0, 255]
        Returns:
            degraded_image: numpy array (H, W, C)
            condition_vector: 条件向量
        """
        raise NotImplementedError
        
    def get_condition_dim(self):
        """返回条件向量的维度"""
        raise NotImplementedError


class GaussianBlurTransform(DegradationTransform):
    """高斯模糊退化"""
    def __init__(self, sigma_range=(0.5, 5.0)):
        super().__init__()
        self.name = "gaussian_blur"
        self.sigma_range = sigma_range
        
    def apply(self, image):
        """
        应用高斯模糊
        """
        # 转换为numpy array
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # 随机采样sigma值
        sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])
        
        # 对每个通道分别应用高斯模糊
        degraded = np.zeros_like(image, dtype=np.float32)
        for c in range(image.shape[2]):
            degraded[:, :, c] = gaussian_filter(image[:, :, c], sigma=sigma)
        
        degraded = np.clip(degraded, 0, 255).astype(np.uint8)
        
        # 构造条件向量: [op_type_id, sigma, 0, 0]
        # op_type_id=0 表示高斯模糊
        condition = np.array([0.0, sigma, 0.0, 0.0], dtype=np.float32)
        
        return degraded, condition
    
    def get_condition_dim(self):
        return 4  # [op_type, param1, param2, param3]


class HistogramEqualizationTransform(DegradationTransform):
    """直方图均衡化退化"""
    def __init__(self, method='global'):
        super().__init__()
        self.name = "histogram_equalization"
        self.method = method
        
    def apply(self, image):
        """
        应用直方图均衡化
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # 转换到YCrCb色彩空间，只对Y通道均衡化
        img_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        
        if self.method == 'global':
            # 全局直方图均衡化
            img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        elif self.method == 'adaptive':
            # 自适应直方图均衡化（CLAHE）
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
        
        # 转换回RGB
        degraded = cv2.cvtColor(img_yuv, cv2.COLOR_YCrCb2RGB)
        
        # 条件向量: [op_type_id, method_id, 0, 0]
        # op_type_id=1 表示直方图均衡
        # method_id: 0=global, 1=adaptive
        method_id = 0.0 if self.method == 'global' else 1.0
        condition = np.array([1.0, method_id, 0.0, 0.0], dtype=np.float32)
        
        return degraded, condition
    
    def get_condition_dim(self):
        return 4


class GammaCorrectionTransform(DegradationTransform):
    """伽马校正退化"""
    def __init__(self, gamma_range=(0.4, 2.5)):
        super().__init__()
        self.name = "gamma_correction"
        self.gamma_range = gamma_range
        
    def apply(self, image):
        """
        应用伽马校正
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # 随机采样gamma值
        gamma = np.random.uniform(self.gamma_range[0], self.gamma_range[1])
        
        # 归一化到[0, 1]
        img_normalized = image.astype(np.float32) / 255.0
        
        # 应用伽马校正
        degraded = np.power(img_normalized, gamma)
        
        # 转换回[0, 255]
        degraded = (degraded * 255).astype(np.uint8)
        
        # 条件向量: [op_type_id, gamma, 0, 0]
        # op_type_id=2 表示伽马校正
        condition = np.array([2.0, gamma, 0.0, 0.0], dtype=np.float32)
        
        return degraded, condition
    
    def get_condition_dim(self):
        return 4


class RandomDegradation:
    """随机选择一种退化算子应用"""
    def __init__(self, transforms_list):
        """
        Args:
            transforms_list: 退化算子列表
        """
        self.transforms = transforms_list
        self.num_transforms = len(transforms_list)
        
    def __call__(self, image):
        """
        随机应用一种退化
        Returns:
            degraded_image: torch.Tensor (C, H, W), 值域[0, 1]
            original_image: torch.Tensor (C, H, W), 值域[0, 1]
            condition: torch.Tensor (condition_dim,)
            info: dict, 包含退化类型和参数信息（用于分析）
        """
        # 随机选择一个退化算子
        transform_idx = np.random.randint(0, self.num_transforms)
        transform = self.transforms[transform_idx]
        
        # 保存原图
        if isinstance(image, Image.Image):
            original = np.array(image)
        else:
            original = image.copy()
        
        # 应用退化
        degraded, condition = transform.apply(image)
        
        # 转换为torch tensor并归一化到[0, 1]
        degraded_tensor = torch.from_numpy(degraded).float() / 255.0
        original_tensor = torch.from_numpy(original).float() / 255.0
        
        # 转换为CHW格式
        degraded_tensor = degraded_tensor.permute(2, 0, 1)
        original_tensor = original_tensor.permute(2, 0, 1)
        
        condition_tensor = torch.from_numpy(condition)
        
        # 记录退化信息（用于分析）
        info = {
            'degradation_type': transform.name,
            'degradation_idx': transform_idx,
            'condition': condition.tolist()
        }
        
        return degraded_tensor, original_tensor, condition_tensor, info


# 便捷函数：创建所有退化算子
def create_degradation_transforms(config=None):
    """
    根据配置创建退化算子列表
    """
    if config is None:
        # 默认配置
        transforms = [
            GaussianBlurTransform(sigma_range=(0.5, 5.0)),
            HistogramEqualizationTransform(method='global'),
            GammaCorrectionTransform(gamma_range=(0.4, 2.5))
        ]
    else:
        transforms = []
        for deg_config in config['degradations']:
            if deg_config['type'] == 'gaussian_blur':
                sigma_range = deg_config['params']['sigma_range']
                transforms.append(GaussianBlurTransform(sigma_range=sigma_range))
            elif deg_config['type'] == 'histogram_equalization':
                method = deg_config['params']['method']
                transforms.append(HistogramEqualizationTransform(method=method))
            elif deg_config['type'] == 'gamma_correction':
                gamma_range = deg_config['params']['gamma_range']
                transforms.append(GammaCorrectionTransform(gamma_range=gamma_range))
    
    return transforms
