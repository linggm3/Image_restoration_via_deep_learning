# experiments/train_unet.py

import sys
import yaml
import torch
import torch.optim as optim
from torch.optim import lr_scheduler

# 将项目根目录添加到Python路径，确保可以正确导入其他模块
sys.path.append('..')

from data.dataset import create_dataloaders
from models.unet import UNet
from training.trainer import Trainer
from training.losses import CombinedLoss

def main():
    """
    U-Net模型的完整训练与评估流程
    """
    print("=" * 50)
    print("开始执行U-Net模型训练流程...")

    # 1. 加载配置文件
    try:
        with open('../config/unet_config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("配置文件 'unet_config.yaml' 加载成功。")
    except FileNotFoundError:
        print("错误：无法找到配置文件 '../config/unet_config.yaml'。请确保文件存在。")
        return
        
    # 指定模型名称为 'unet'，以确保Trainer使用非条件模型的逻辑
    config['model']['name'] = "unet_augmented" # 可以给新实验起个名字
    
    # 2. 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"将使用设备: {device}")

    # 3. 创建数据加载器 (现在会返回三个)
    print("正在创建数据加载器...")
    # 请确保 config['data']['data_root'] 指向你的 MS-COCO_2014 数据集路径
    train_loader, val_loader, test_loader = create_dataloaders(config)
    print(f"测试集样本数: {len(test_loader.dataset)}")
    
    # 4. 初始化模型
    print("正在初始化U-Net模型...")
    model = UNet(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        base_channels=config['model']['base_channels'],
        bilinear=True
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型初始化完成。参数量: {num_params / 1e6:.2f} M")

    # 5. 初始化损失函数
    print("正在初始化损失函数...")
    loss_fn = CombinedLoss(loss_weights=config['training']['loss_weights']).to(device)

    # 6. 初始化优化器和学习率调度器
    print("正在初始化优化器和学习率调度器...")
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs'],
        eta_min=1e-6
    )

    # 7. 初始化训练器
    print("正在初始化训练器...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device
    )

    # 8. 开始训练
    print("=" * 50)
    print(f"训练即将开始，共计 {config['training']['epochs']} 个 Epoch。")
    print("日志将保存在: ", trainer.log_dir)
    print("检查点将保存在: ", trainer.checkpoint_dir)
    print("=" * 50)
    
    trainer.train()

    print("=" * 50)
    print("训练流程已全部完成！")
    print("后续步骤：可以在 test_loader 上进行最终的模型评估。")
    print("=" * 50)

if __name__ == '__main__':
    main()
