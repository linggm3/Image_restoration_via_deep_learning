# experiments/train_unet.py

import sys
import yaml
import torch
import torch.optim as optim
from torch.optim import lr_scheduler

sys.path.append('..')

from data.dataset import create_dataloaders
from models.unet import UNet, ConditionalUNet
from training.trainer import Trainer
from training.losses import CombinedLoss

def main():
    """
    U-Net模型的完整训练与评估流程
    支持条件模型和非条件模型
    """
    print("=" * 70)
    print("🚀 开始执行U-Net模型训练流程...")
    print("=" * 70)

    # 1. 加载配置文件
    try:
        with open('../config/unet_config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("✓ 配置文件 'unet_config.yaml' 加载成功")
    except FileNotFoundError:
        print("✗ 错误：无法找到配置文件 '../config/unet_config.yaml'")
        return
    
    # 2. 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ 使用设备: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # 3. 创建数据加载器
    print("\n" + "=" * 70)
    print("📊 正在创建数据加载器...")
    print("=" * 70)
    
    # 可选：使用子集进行快速测试
    # subset_size = 1000  # 取消注释以使用子集
    subset_size = None
    
    train_loader, val_loader, test_loader = create_dataloaders(config, subset_size=subset_size)
    
    print(f"✓ 数据加载器创建成功")
    print(f"  训练集: {len(train_loader.dataset)} 样本")
    print(f"  验证集: {len(val_loader.dataset)} 样本")
    print(f"  测试集: {len(test_loader.dataset)} 样本")
    
    # 4. 初始化模型
    print("\n" + "=" * 70)
    print("🏗️  正在初始化模型...")
    print("=" * 70)
    
    use_conditional = config['model'].get('use_conditional', False)
    
    if use_conditional:
        print("✓ 使用条件U-Net模型（支持退化链）")
        model = ConditionalUNet(
            in_channels=config['model']['in_channels'],
            out_channels=config['model']['out_channels'],
            condition_dim=config['model']['condition_dim'],
            base_channels=config['model']['base_channels'],
            bilinear=True
        ).to(device)
    else:
        print("✓ 使用基础U-Net模型")
        model = UNet(
            in_channels=config['model']['in_channels'],
            out_channels=config['model']['out_channels'],
            base_channels=config['model']['base_channels'],
            bilinear=True
        ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  模型参数量: {num_params / 1e6:.2f} M")

    # 5. 初始化损失函数
    print("\n✓ 正在初始化损失函数...")
    loss_fn = CombinedLoss(loss_weights=config['training']['loss_weights']).to(device)
    print(f"  L1 权重: {config['training']['loss_weights']['l1']}")
    print(f"  感知损失权重: {config['training']['loss_weights']['perceptual']}")

    # 6. 初始化优化器和学习率调度器
    print("\n✓ 正在初始化优化器和学习率调度器...")
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
    
    print(f"  初始学习率: {config['training']['learning_rate']}")
    print(f"  权重衰减: {config['training']['weight_decay']}")

    # 7. 初始化训练器
    print("\n✓ 正在初始化训练器...")
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
    
    print(f"  日志目录: {trainer.log_dir}")
    print(f"  检查点目录: {trainer.checkpoint_dir}")
    print(f"  混合精度训练: {config['training']['mixed_precision']}")
    print(f"  梯度裁剪: {config['training']['gradient_clip']}")

    # 8. 可选：从检查点恢复训练
    resume_from = None  # 设置为检查点路径以恢复训练
    if resume_from and os.path.exists(resume_from):
        print(f"\n✓ 从检查点恢复训练: {resume_from}")
        trainer.load_checkpoint(resume_from)

    # 9. 开始训练
    print("\n" + "=" * 70)
    print(f"🎯 开始训练 - 共 {config['training']['epochs']} 个 Epoch")
    print("=" * 70)
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n⚠️  训练被用户中断")
        save_path = os.path.join(trainer.checkpoint_dir, 'interrupted.pth')
        trainer.save_checkpoint(trainer.start_epoch, 'interrupted.pth')
        print(f"✓ 中断时的模型已保存: {save_path}")
    except Exception as e:
        print(f"\n✗ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("✅ 训练流程已全部完成！")
    print(f"📁 模型保存在: {trainer.checkpoint_dir}")
    print(f"📊 日志保存在: {trainer.log_dir}")
    print("\n💡 后续步骤：")
    print("1. 运行 'python evaluation_unet.py' 在测试集上评估模型")
    print("2. 使用 'tensorboard --logdir={}'.format(trainer.log_dir)")
    print("     查看训练曲线和指标")
    print("=" * 70)


if __name__ == '__main__':
    import os
    main()
