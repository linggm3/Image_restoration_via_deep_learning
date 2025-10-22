# data/dataset.py

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from .transforms import RandomDegradation, create_degradation_transforms

class DynamicDegradationDataset(Dataset):
    """
    动态生成退化图像的数据集
    每次__getitem__都实时应用随机退化
    """
    def __init__(self, 
                 root_dir, 
                 image_size=256,
                 degradation_config=None,
                 split='train'):
        """
        Args:
            root_dir: 图像文件夹路径（如COCO的train2014文件夹）
            image_size: 输出图像尺寸
            degradation_config: 退化配置字典
            split: 'train' or 'val'
        """
        self.root_dir = root_dir
        self.image_size = image_size
        self.split = split
        
        # 获取所有图像文件路径
        self.image_paths = self._load_image_paths()
        
        print(f"加载了 {len(self.image_paths)} 张图像 ({split})")
        
        # 创建退化算子
        self.degradations = create_degradation_transforms(degradation_config)
        self.random_degradation = RandomDegradation(self.degradations)
        
        # 图像预处理（调整大小和中心裁剪）
        self.transform = T.Compose([
            T.Resize(int(image_size * 1.12)),  # 稍微大一点再裁剪
            T.CenterCrop(image_size),
        ])
        
    def _load_image_paths(self):
        """加载所有图像路径"""
        image_paths = []
        
        # 支持的图像格式
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        for fname in os.listdir(self.root_dir):
            ext = os.path.splitext(fname)[1].lower()
            if ext in valid_extensions:
                image_paths.append(os.path.join(self.root_dir, fname))
        
        return sorted(image_paths)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Returns:
            dict: {
                'degraded': 退化图像 (C, H, W), 值域[0, 1]
                'original': 原始图像 (C, H, W), 值域[0, 1]
                'condition': 条件向量 (condition_dim,)
                'degradation_type': 退化类型字符串
                'degradation_params': 退化参数列表
                'image_path': 图像路径字符串
            }
        """
        # 读取图像
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"读取图像失败: {img_path}, 错误: {e}")
            # 返回下一张图像
            return self.__getitem__((idx + 1) % len(self))
        
        # 预处理（调整大小）
        image = self.transform(image)
        
        # 应用随机退化
        degraded, original, condition, info = self.random_degradation(image)
        
        # 返回字典，注意这里直接返回字符串和标量，而不是嵌套字典
        return {
            'degraded': degraded,
            'original': original,
            'condition': condition,
            'degradation_type': info['degradation_type'],  # 字符串
            'degradation_idx': info['degradation_idx'],    # 整数
            'degradation_params': info['condition'],       # 列表
            'image_path': img_path                         # 字符串
        }


# 自定义collate函数，正确处理batch
def custom_collate_fn(batch):
    """
    自定义collate函数，正确处理包含字符串的batch
    """
    # 分离tensor和非tensor数据
    degraded = torch.stack([item['degraded'] for item in batch])
    original = torch.stack([item['original'] for item in batch])
    condition = torch.stack([item['condition'] for item in batch])
    
    # 保持字符串和列表为列表形式
    degradation_types = [item['degradation_type'] for item in batch]
    degradation_indices = [item['degradation_idx'] for item in batch]
    degradation_params = [item['degradation_params'] for item in batch]
    image_paths = [item['image_path'] for item in batch]
    
    return {
        'degraded': degraded,
        'original': original,
        'condition': condition,
        'degradation_type': degradation_types,
        'degradation_idx': degradation_indices,
        'degradation_params': degradation_params,
        'image_path': image_paths
    }


def create_dataloaders(config, val_split_ratio=0.1):
    """
    创建训练和验证数据加载器
    
    Args:
        config: 配置字典
        val_split_ratio: 验证集比例
    
    Returns:
        train_loader, val_loader
    """
    from torch.utils.data import DataLoader, random_split
    
    # 创建完整数据集
    full_dataset = DynamicDegradationDataset(
        root_dir=config['data']['data_root'],
        image_size=config['data']['image_size'],
        degradation_config=config['data'],
        split='full'
    )
    
    # 划分训练集和验证集
    val_size = int(len(full_dataset) * val_split_ratio)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # 固定随机种子保证可复现
    )
    
    # 创建数据加载器（使用自定义collate函数）
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate_fn  # 关键：使用自定义collate
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        collate_fn=custom_collate_fn  # 关键：使用自定义collate
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    return train_loader, val_loader
