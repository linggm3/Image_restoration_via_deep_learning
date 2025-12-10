# experiments/test_trainer.py

import sys
sys.path.append('..')
import yaml
import torch
from data.dataset import create_dataloaders
from models.unet import UNet, ConditionalUNet
from training.trainer import Trainer
from training.losses import CombinedLoss

def test_training_pipeline(use_conditional=True):
    """
    测试完整的训练流程
    Args:
        use_conditional: 是否使用条件模型
    """
    print("=" * 70)
    print(f"测试训练流程 - {'条件模型' if use_conditional else '基础模型'}")
    print("=" * 70)
    
    # 加载配置
    try:
        with open('../config/unet_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print("✓ 配置文件加载成功")
    except FileNotFoundError:
        print("✗ 错误：无法找到配置文件")
        return False
    
    # 修改配置以快速测试
    print("\n📝 配置测试参数...")
    config['model']['use_conditional'] = use_conditional
    config['model']['name'] = f"test_{'conditional' if use_conditional else 'basic'}_unet"
    config['data']['batch_size'] = 4
    config['data']['num_workers'] = 0
    config['training']['epochs'] = 2
    config['training']['log_interval'] = 5
    config['training']['mixed_precision'] = False  # 禁用以便调试
    config['evaluation']['metrics'] = ["psnr", "lpips"]
    
    print(f"  批大小: {config['data']['batch_size']}")
    print(f"  训练轮次: {config['training']['epochs']}")
    print(f"  混合精度: {config['training']['mixed_precision']}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n✓ 使用设备: {device}")
    
    try:
        # 1. 创建数据加载器
        print("\n📊 创建数据加载器...")
        train_loader, val_loader, _ = create_dataloaders(config, subset_size=50)
        print(f"✓ 训练集: {len(train_loader.dataset)} 样本")
        print(f"✓ 验证集: {len(val_loader.dataset)} 样本")
        
        # 2. 初始化模型
        print("\n🏗️  初始化模型...")
        if use_conditional:
            model = ConditionalUNet(
                in_channels=config['model']['in_channels'],
                out_channels=config['model']['out_channels'],
                condition_dim=config['model']['condition_dim'],
                base_channels=32,  # 使用较小的模型加快测试
                bilinear=True
            ).to(device)
        else:
            model = UNet(
                in_channels=config['model']['in_channels'],
                out_channels=config['model']['out_channels'],
                base_channels=32,
                bilinear=True
            ).to(device)
        
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✓ 模型参数量: {num_params / 1e6:.2f} M")
        
        # 3. 初始化损失函数
        print("\n📉 初始化损失函数...")
        loss_fn = CombinedLoss(loss_weights=config['training']['loss_weights']).to(device)
        print("✓ 损失函数初始化完成")
        
        # 4. 初始化优化器和调度器
        print("\n⚙️  初始化优化器...")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['epochs']
        )
        print(f"✓ 学习率: {config['training']['learning_rate']}")
        
        # 5. 初始化训练器
        print("\n🎯 初始化训练器...")
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
        print("✓ 训练器初始化完成")
        
        # 6. 开始训练
        print("\n🚀 开始训练...")
        trainer.train()
        
        # 7. 检查输出
        print("\n🔍 检查训练输出...")
        import os
        
        # 检查日志目录
        if os.path.exists(trainer.log_dir):
            print(f"✓ 日志目录已创建: {trainer.log_dir}")
        else:
            print(f"✗ 日志目录未创建")
            return False
        
        # 检查检查点目录
        if os.path.exists(trainer.checkpoint_dir):
            print(f"✓ 检查点目录已创建: {trainer.checkpoint_dir}")
            
            # 检查是否保存了最佳模型
            best_path = os.path.join(trainer.checkpoint_dir, 'best.pth')
            if os.path.exists(best_path):
                print(f"✓ 最佳模型已保存: {best_path}")
            else:
                print(f"⚠️  最佳模型未保存")
        else:
            print(f"✗ 检查点目录未创建")
            return False
        
        print("\n✅ 训练流程测试成功!")
        return True
        
    except Exception as e:
        print(f"\n✗ 训练流程测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_checkpoint_save_load():
    """测试检查点的保存和加载"""
    print("\n" + "=" * 70)
    print("测试检查点保存和加载")
    print("=" * 70)
    
    try:
        # 创建一个简单的模型和优化器
        model = UNet(in_channels=3, out_channels=3, base_channels=32)
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        
        # 保存状态
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'test_checkpoint.pth')
            
            print(f"✓ 临时目录: {tmpdir}")
            
            # 保存
            state = {
                'epoch': 5,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_metric': 0.123
            }
            torch.save(state, checkpoint_path)
            print(f"✓ 检查点已保存")
            
            # 加载
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            print(f"✓ 检查点已加载")
            print(f"  Epoch: {checkpoint['epoch']}")
            print(f"  Best metric: {checkpoint['best_val_metric']}")
        
        print("\n✅ 检查点保存/加载测试成功!")
        return True
        
    except Exception as e:
        print(f"\n✗ 检查点测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_single_batch_training():
    """测试单个批次的训练"""
    print("\n" + "=" * 70)
    print("测试单批次训练")
    print("=" * 70)
    
    try:
        # 创建虚拟数据
        batch_size = 2
        img_size = 128
        condition_dim = 15
        
        degraded = torch.randn(batch_size, 3, img_size, img_size)
        original = torch.randn(batch_size, 3, img_size, img_size)
        condition = torch.randn(batch_size, condition_dim)
        
        # 创建模型
        model = ConditionalUNet(
            in_channels=3,
            out_channels=3,
            condition_dim=condition_dim,
            base_channels=32
        )
        
        # 创建损失函数和优化器
        loss_fn = CombinedLoss(loss_weights={'l1': 1.0, 'perceptual': 0.0})
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        print("✓ 模型和优化器初始化完成")
        
        # 训练一个批次
        model.train()
        optimizer.zero_grad()
        
        restored = model(degraded, condition)
        total_loss, loss_dict = loss_fn(restored, original)
        
        print(f"✓ 前向传播完成，损失: {total_loss.item():.4f}")
        
        total_loss.backward()
        optimizer.step()
        
        print("✓ 反向传播和优化完成")
        
        # 再次前向传播，检查损失是否变化
        with torch.no_grad():
            restored2 = model(degraded, condition)
            total_loss2, _ = loss_fn(restored2, original)
        
        print(f"✓ 优化后损失: {total_loss2.item():.4f}")
        print(f"  损失变化: {total_loss.item() - total_loss2.item():+.6f}")
        
        print("\n✅ 单批次训练测试成功!")
        return True
        
    except Exception as e:
        print(f"\n✗ 单批次训练测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("\n" + "🧪" * 35)
    print("开始训练流程单元测试")
    print("🧪" * 35 + "\n")
    
    results = []
    
    # 运行所有测试
    results.append(("单批次训练", test_single_batch_training()))
    results.append(("检查点保存/加载", test_checkpoint_save_load()))
    results.append(("基础模型训练", test_training_pipeline(use_conditional=False)))
    results.append(("条件模型训练", test_training_pipeline(use_conditional=True)))
    
    # 总结
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name:25s}: {status}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过!")
    else:
        print(f"\n⚠️  {total - passed} 个测试失败")
    
    print("=" * 70)
