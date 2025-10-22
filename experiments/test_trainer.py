# experiments/train_unet.py

import sys
sys.path.append('..')
import yaml
import torch
from data.dataset import create_dataloaders
from models.unet import UNet  # 导入基础UNet
from training.trainer import Trainer
from training.losses import CombinedLoss

def main():
    """主训练函数"""
    
    # 加载配置
    with open('../config/unet_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    # --- 为快速测试，修改配置 ---
    print("--- 正在使用测试配置 ---")
    config['model']['name'] = "unet" # 明确指定为非条件模型
    config['data']['batch_size'] = 2
    config['data']['num_workers'] = 0
    config['training']['epochs'] = 2 # 只训练2个epoch
    config['evaluation']['metrics'] = ["psnr", "lpips"] # 减少计算量
    # --------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 1. 创建数据加载器
    train_loader, val_loader = create_dataloaders(config, val_split_ratio=0.1)
    
    # 2. 初始化模型
    model = UNet(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        base_channels=config['model']['base_channels']
    ).to(device)
    
    # 3. 初始化损失函数
    loss_fn = CombinedLoss(loss_weights=config['training']['loss_weights']).to(device)
    
    # 4. 初始化优化器和学习率调度器
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config['training']['epochs']
    )
    
    # 5. 初始化训练器
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
    
    # 6. 开始训练
    print("开始训练...")
    trainer.train()
    print("测试训练流程结束。")

if __name__ == '__main__':
    main()
