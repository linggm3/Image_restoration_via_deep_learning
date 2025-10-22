# experiments/test_losses.py
import sys
sys.path.append('..')
import torch
from training.losses import CombinedLoss

def test_loss_calculation():
    """测试损失函数计算"""
    print("=" * 50)
    print("正在测试损失函数...")

    # 损失权重配置
    loss_weights = {'l1': 1.0, 'perceptual': 0.1}
    
    # 创建损失函数实例
    loss_fn = CombinedLoss(loss_weights=loss_weights)
    
    # 创建随机的输入和目标张量
    batch_size = 2
    img_size = 64
    pred_tensor = torch.rand(batch_size, 3, img_size, img_size)
    target_tensor = torch.rand(batch_size, 3, img_size, img_size)
    
    # 计算损失
    try:
        total_loss, loss_dict = loss_fn(pred_tensor, target_tensor)
        
        print(f"总损失: {total_loss.item():.4f}")
        for name, value in loss_dict.items():
            print(f"  - {name}: {value.item():.4f}")
            
        # 检查损失是否为正数
        assert total_loss.item() > 0
        assert 'l1_loss' in loss_dict
        assert 'perceptual_loss' in loss_dict
        
        print("\n✓ 损失函数测试成功!")
        
    except Exception as e:
        print(f"\n✗ 损失函数测试失败: {e}")
        
    print("=" * 50)

if __name__ == '__main__':
    test_loss_calculation()
