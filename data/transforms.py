# data/transforms.py

import torch
import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image
import cv2
from io import BytesIO

class DegradationTransform:
    """退化算子基类"""
    def __init__(self):
        self.name = "base"
        self.degradation_id = -1
        
    def apply(self, image, params=None):
        """
        应用退化
        Args:
            image: numpy array (H, W, C), RGB格式, 值域[0, 255]
            params: 可选的退化参数，如果为None则随机采样
        Returns:
            degraded_image: numpy array (H, W, C)
            actual_params: 实际使用的退化参数
        """
        raise NotImplementedError


class MotionBlurTransform(DegradationTransform):
    """运动模糊退化（模拟手抖、相机移动等）"""
    def __init__(self, kernel_size_range=(5, 25), angle_range=(0, 180)):
        super().__init__()
        self.name = "motion_blur"
        self.degradation_id = 0
        self.kernel_size_range = kernel_size_range
        self.angle_range = angle_range
        
    def _get_motion_blur_kernel(self, kernel_size, angle):
        """生成运动模糊核"""
        # 确保kernel_size为奇数
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        kernel = np.zeros((kernel_size, kernel_size))
        
        # 在中心绘制一条线
        center = kernel_size // 2
        angle_rad = np.deg2rad(angle)
        
        for i in range(kernel_size):
            offset = i - center
            x = int(center + offset * np.cos(angle_rad))
            y = int(center + offset * np.sin(angle_rad))
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                kernel[y, x] = 1
        
        # 归一化
        kernel = kernel / np.sum(kernel) if np.sum(kernel) > 0 else kernel
        return kernel
        
    def apply(self, image, params=None):
        """应用运动模糊"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # 参数采样
        if params is None:
            kernel_size = np.random.randint(self.kernel_size_range[0], self.kernel_size_range[1] + 1)
            angle = np.random.uniform(self.angle_range[0], self.angle_range[1])
        else:
            kernel_size = int(params.get('kernel_size', 15))
            angle = params.get('angle', 45.0)
        
        # 生成运动模糊核
        kernel = self._get_motion_blur_kernel(kernel_size, angle)
        
        # 对每个通道应用卷积
        degraded = cv2.filter2D(image, -1, kernel)
        degraded = np.clip(degraded, 0, 255).astype(np.uint8)
        
        actual_params = {
            'kernel_size': float(kernel_size),
            'angle': float(angle)
        }
        
        return degraded, actual_params


class GaussianBlurTransform(DegradationTransform):
    """高斯模糊退化（模拟失焦）"""
    def __init__(self, sigma_range=(0.5, 8.0)):
        super().__init__()
        self.name = "gaussian_blur"
        self.degradation_id = 1
        self.sigma_range = sigma_range
        
    def apply(self, image, params=None):
        """应用高斯模糊"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # 参数采样
        if params is None:
            sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])
        else:
            sigma = params.get('sigma', 2.0)
        
        # 应用高斯模糊
        degraded = cv2.GaussianBlur(image, (0, 0), sigma)
        degraded = np.clip(degraded, 0, 255).astype(np.uint8)
        
        actual_params = {'sigma': float(sigma)}
        
        return degraded, actual_params


class GaussianNoiseTransform(DegradationTransform):
    """高斯噪声退化（模拟低光环境）"""
    def __init__(self, noise_std_range=(5, 50)):
        super().__init__()
        self.name = "gaussian_noise"
        self.degradation_id = 2
        self.noise_std_range = noise_std_range
        
    def apply(self, image, params=None):
        """应用高斯噪声"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # 参数采样
        if params is None:
            noise_std = np.random.uniform(self.noise_std_range[0], self.noise_std_range[1])
        else:
            noise_std = params.get('noise_std', 15.0)
        
        # 生成高斯噪声
        noise = np.random.normal(0, noise_std, image.shape)
        degraded = image.astype(np.float32) + noise
        degraded = np.clip(degraded, 0, 255).astype(np.uint8)
        
        actual_params = {'noise_std': float(noise_std)}
        
        return degraded, actual_params


class JPEGCompressionTransform(DegradationTransform):
    """JPEG压缩退化（模拟画质损失）"""
    def __init__(self, quality_range=(10, 75)):
        super().__init__()
        self.name = "jpeg_compression"
        self.degradation_id = 3
        self.quality_range = quality_range
        
    def apply(self, image, params=None):
        """应用JPEG压缩"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # 参数采样
        if params is None:
            quality = np.random.randint(self.quality_range[0], self.quality_range[1] + 1)
        else:
            quality = int(params.get('quality', 50))
        
        # 转换为PIL Image
        pil_image = Image.fromarray(image)
        
        # 使用BytesIO模拟JPEG压缩
        buffer = BytesIO()
        pil_image.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        compressed_image = Image.open(buffer)
        degraded = np.array(compressed_image)
        
        actual_params = {'quality': float(quality)}
        
        return degraded, actual_params


class DownsamplingTransform(DegradationTransform):
    """下采样退化（模拟分辨率不足）"""
    def __init__(self, scale_range=(0.25, 0.75)):
        super().__init__()
        self.name = "downsampling"
        self.degradation_id = 4
        self.scale_range = scale_range
        
    def apply(self, image, params=None):
        """应用下采样"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        h, w = image.shape[:2]
        
        # 参数采样
        if params is None:
            scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        else:
            scale = params.get('scale', 0.5)
        
        # 下采样
        new_h, new_w = int(h * scale), int(w * scale)
        downsampled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # 上采样回原始尺寸
        degraded = cv2.resize(downsampled, (w, h), interpolation=cv2.INTER_LINEAR)
        degraded = np.clip(degraded, 0, 255).astype(np.uint8)
        
        actual_params = {'scale': float(scale)}
        
        return degraded, actual_params


class DegradationChain:
    """
    退化链：按顺序应用多种退化
    随机选择1-3种退化进行应用，防止过度退化
    """
    def __init__(self, transforms_dict, probabilities=None):
        """
        Args:
            transforms_dict: 字典，{退化名称: 退化实例}
            probabilities: 字典，{退化名称: 应用概率}，如果为None则使用默认概率
        """
        self.transforms_dict = transforms_dict
        self.transform_names = list(transforms_dict.keys())
        
        # 设置默认概率
        if probabilities is None:
            self.probabilities = {name: 0.5 for name in self.transform_names}
        else:
            self.probabilities = probabilities
            
    def __call__(self, image):
        """
        应用退化链
        Returns:
            degraded_image: torch.Tensor (C, H, W), 值域[0, 1]
            original_image: torch.Tensor (C, H, W), 值域[0, 1]
            condition: torch.Tensor (max_condition_dim,)
            info: dict，包含详细的退化信息
        """
        # 保存原图
        if isinstance(image, Image.Image):
            original = np.array(image)
        else:
            original = image.copy()
        
        degraded = original.copy()
        applied_degradations = []
        all_params = {}
        
        # --- 随机选择 1-3 种退化 ---
        # 概率设置: 1种(适中), 2种(最大), 3种(最小)
        # P(1)=0.3, P(2)=0.5, P(3)=0.2
        num_choices = [1, 2, 3]
        probs = [0.4, 0.5, 0.1]
        
        # 1. 确定要应用几种退化
        num_to_apply = np.random.choice(num_choices, p=probs)
        # 确保不超过实际可用的退化种类总数
        num_to_apply = min(num_to_apply, len(self.transform_names))
        
        # 2. 确定具体应用哪些退化
        # 使用配置中的 probabilities 作为权重，这样保留了配置文件的倾向性
        weights = np.array([self.probabilities.get(name, 0.5) for name in self.transform_names])
        if weights.sum() > 0:
            weights = weights / weights.sum() # 归一化
        else:
            weights = None # 均匀分布
            
        # 不放回采样
        selected_names = np.random.choice(
            self.transform_names, 
            size=num_to_apply, 
            replace=False, 
            p=weights
        )
        selected_set = set(selected_names)
        
        # 3. 按原始顺序应用选中的退化
        # 保持原始顺序(如先模糊后噪声)通常更符合物理规律
        for name in self.transform_names:
            if name in selected_set:
                transform = self.transforms_dict[name]
                degraded, params = transform.apply(degraded)
                applied_degradations.append({
                    'name': name,
                    'id': transform.degradation_id,
                    'params': params
                })
                all_params[name] = params
        
        # 保底机制：理论上 num_to_apply >= 1 不会触发，保留以防万一
        if len(applied_degradations) == 0 and len(self.transform_names) > 0:
            fallback_name = np.random.choice(self.transform_names)
            transform = self.transforms_dict[fallback_name]
            degraded, params = transform.apply(original)
            applied_degradations.append({
                'name': fallback_name,
                'id': transform.degradation_id,
                'params': params
            })
            all_params[fallback_name] = params
        
        # 构造条件向量
        # 格式: [deg1_flag, deg1_param1, deg1_param2, ..., deg2_flag, deg2_param1, ...]
        condition = self._build_condition_vector(applied_degradations)
        
        # 转换为torch tensor
        degraded_tensor = torch.from_numpy(degraded).float() / 255.0
        original_tensor = torch.from_numpy(original).float() / 255.0
        
        # 转换为CHW格式
        degraded_tensor = degraded_tensor.permute(2, 0, 1)
        original_tensor = original_tensor.permute(2, 0, 1)
        
        condition_tensor = torch.from_numpy(condition).float()
        
        # 记录退化信息
        info = {
            'applied_degradations': applied_degradations,
            'num_degradations': len(applied_degradations),
            'degradation_names': [d['name'] for d in applied_degradations],
            'all_params': all_params,
            'condition': condition.tolist()
        }
        
        return degraded_tensor, original_tensor, condition_tensor, info
    
    def _build_condition_vector(self, applied_degradations):
        """
        构建条件向量
        格式：每种退化占3个位置 [flag, param1, param2]
        总维度: 5 * 3 = 15
        """
        condition = np.zeros(15, dtype=np.float32)
        
        # 为每种退化类型分配位置
        degradation_positions = {
            'motion_blur': 0,      # [0:3]
            'gaussian_blur': 3,    # [3:6]
            'gaussian_noise': 6,   # [6:9]
            'jpeg_compression': 9, # [9:12]
            'downsampling': 12     # [12:15]
        }
        
        for deg_info in applied_degradations:
            name = deg_info['name']
            params = deg_info['params']
            pos = degradation_positions[name]
            
            # 设置flag为1表示该退化被应用
            condition[pos] = 1.0
            
            # 填充参数（归一化到合理范围）
            if name == 'motion_blur':
                condition[pos + 1] = params['kernel_size'] / 25.0  # 归一化到[0, 1]
                condition[pos + 2] = params['angle'] / 180.0
            elif name == 'gaussian_blur':
                condition[pos + 1] = params['sigma'] / 8.0
            elif name == 'gaussian_noise':
                condition[pos + 1] = params['noise_std'] / 50.0
            elif name == 'jpeg_compression':
                condition[pos + 1] = params['quality'] / 100.0
            elif name == 'downsampling':
                condition[pos + 1] = params['scale']
        
        return condition


def create_degradation_chain(config=None):
    """
    根据配置创建退化链
    """
    if config is None:
        # 默认配置
        transforms_dict = {
            'motion_blur': MotionBlurTransform(),
            'gaussian_blur': GaussianBlurTransform(),
            'gaussian_noise': GaussianNoiseTransform(),
            'jpeg_compression': JPEGCompressionTransform(),
            'downsampling': DownsamplingTransform()
        }
        probabilities = {name: 0.5 for name in transforms_dict.keys()}
    else:
        transforms_dict = {}
        probabilities = {}
        
        for deg_config in config.get('degradations', []):
            deg_type = deg_config['type']
            params = deg_config.get('params', {})
            prob = deg_config.get('probability', 0.5)
            
            if deg_type == 'motion_blur':
                transform = MotionBlurTransform(
                    kernel_size_range=params.get('kernel_size_range', [5, 25]),
                    angle_range=params.get('angle_range', [0, 180])
                )
            elif deg_type == 'gaussian_blur':
                transform = GaussianBlurTransform(
                    sigma_range=params.get('sigma_range', [0.5, 8.0])
                )
            elif deg_type == 'gaussian_noise':
                transform = GaussianNoiseTransform(
                    noise_std_range=params.get('noise_std_range', [5, 50])
                )
            elif deg_type == 'jpeg_compression':
                transform = JPEGCompressionTransform(
                    quality_range=params.get('quality_range', [10, 75])
                )
            elif deg_type == 'downsampling':
                transform = DownsamplingTransform(
                    scale_range=params.get('scale_range', [0.25, 0.75])
                )
            else:
                continue
            
            transforms_dict[deg_type] = transform
            probabilities[deg_type] = prob
    
    return DegradationChain(transforms_dict, probabilities)


# 兼容旧版本的函数
def create_degradation_transforms(config=None):
    """保持向后兼容"""
    return create_degradation_chain(config)
