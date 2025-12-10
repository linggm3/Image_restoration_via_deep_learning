# data/dataset.py

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import numpy as np
from .transforms import create_degradation_chain

class DynamicDegradationDataset(Dataset):
    """
    动态生成退化图像的数据集
    每次__getitem__都实时应用随机退化链
    """
    def __init__(self,
                 root_dir,
                 image_paths,
                 image_size=256,
                 degradation_config=None,
                 split='train',
                 augmentations=None):
        """
        Args:
            root_dir: 图像文件夹根路径
            image_paths: 该数据集使用的图像路径列表
            image_size: 输出图像尺寸
            degradation_config: 退化配置字典
            split: 'train', 'val', or 'test'
            augmentations: 应用于训练集的数据增强变换
        """
        self.root_dir = root_dir
        self.image_paths = image_paths
        self.image_size = image_size
        self.split = split
        self.augmentations = augmentations
        
        # 创建退化链
        self.degradation_chain = create_degradation_chain(degradation_config)
        
        # 图像预处理（调整大小和中心裁剪）- 用于验证和测试
        self.base_transform = T.Compose([
            T.Resize(int(image_size * 1.12)),
            T.CenterCrop(image_size),
        ])
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"读取图像失败: {img_path}, 错误: {e}")
            # 如果当前图像读取失败，则尝试读取下一个
            return self.__getitem__((idx + 1) % len(self))
        
        # 根据数据集类型应用不同的变换
        if self.split == 'train' and self.augmentations:
            image = self.augmentations(image)
        else:
            image = self.base_transform(image)

        # 实时应用退化链
        degraded, original, condition, info = self.degradation_chain(image)
        
        return {
            'degraded': degraded,
            'original': original,
            'condition': condition,
            'degradation_info': info,
            'image_path': img_path
        }


def custom_collate_fn(batch):
    """自定义的collate函数，用于批处理"""
    degraded = torch.stack([item['degraded'] for item in batch])
    original = torch.stack([item['original'] for item in batch])
    condition = torch.stack([item['condition'] for item in batch])
    degradation_info = [item['degradation_info'] for item in batch]
    image_paths = [item['image_path'] for item in batch]
    
    return {
        'degraded': degraded,
        'original': original,
        'condition': condition,
        'degradation_info': degradation_info,
        'image_path': image_paths
    }


def create_dataloaders(config, subset_size=None):
    """
    创建训练、验证和测试数据加载器 (8:1:1 划分)
    """
    
    # 1. 加载所有图像路径
    root_dir = config['data']['data_root']
    all_image_paths = []
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    print("正在加载所有图像路径...")
    all_files = sorted(os.listdir(root_dir))
    
    if subset_size is not None:
        print(f"警告：将使用 {subset_size} 个样本的子集进行操作。")
        np.random.seed(42)
        np.random.shuffle(all_files)
        all_files = all_files[:subset_size]

    for fname in all_files:
        ext = os.path.splitext(fname)[1].lower()
        if ext in valid_extensions:
            all_image_paths.append(os.path.join(root_dir, fname))
    
    print(f"找到 {len(all_image_paths)} 张有效图像")
    
    # 2. 8:1:1 划分数据集
    num_images = len(all_image_paths)
    num_test = int(num_images * 0.1)
    num_val = int(num_images * 0.1)
    num_train = num_images - num_val - num_test
    
    # 使用固定的随机种子确保划分可复现
    np.random.seed(42)
    np.random.shuffle(all_image_paths)
    
    train_paths = all_image_paths[:num_train]
    val_paths = all_image_paths[num_train : num_train + num_val]
    test_paths = all_image_paths[num_train + num_val:]

    print(f"数据集划分完成: 训练集={len(train_paths)}, 验证集={len(val_paths)}, 测试集={len(test_paths)}")

    # 3. 创建训练集数据增强
    aug_config = config['data'].get('augmentations', {})
    train_augmentations = T.Compose([
        T.RandomResizedCrop(
            config['data']['image_size'], 
            scale=aug_config.get('random_crop_scale', [0.8, 1.0])
        ),
        T.RandomHorizontalFlip(p=aug_config.get('horizontal_flip_prob', 0.5)),
    ])

    # 4. 创建各个数据集实例
    train_dataset = DynamicDegradationDataset(
        root_dir=root_dir,
        image_paths=train_paths,
        image_size=config['data']['image_size'],
        degradation_config=config['data'],
        split='train',
        augmentations=train_augmentations
    )
    
    val_dataset = DynamicDegradationDataset(
        root_dir=root_dir,
        image_paths=val_paths,
        image_size=config['data']['image_size'],
        degradation_config=config['data'],
        split='val',
        augmentations=None
    )
    
    test_dataset = DynamicDegradationDataset(
        root_dir=root_dir,
        image_paths=test_paths,
        image_size=config['data']['image_size'],
        degradation_config=config['data'],
        split='test',
        augmentations=None
    )

    # 5. 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        collate_fn=custom_collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    return train_loader, val_loader, test_loader
