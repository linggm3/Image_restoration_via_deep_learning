# experiments/test_unet.py
import sys
sys.path.append('..') # 将项目根目录添加到Python路径

import torch
from models.unet import UNet

def test_unet_forward_pass():
    """测试U-Net模型的前向传播"""
    print("=" * 50)
    print("正在测试基础U-Net模型...")
    
    # 模型参数
    in_channels = 3
    out_channels = 3
    batch_size = 2
    img_size = 256
    
    # 创建模型实例
    model = UNet(in_channels=in_channels, out_channels=out_channels)
    
    # 打印模型参数量
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {num_params / 1e6:.2f} M")
    
    # 创建一个随机输入张量
    input_tensor = torch.randn(batch_size, in_channels, img_size, img_size)
    print(f"输入张量形状: {input_tensor.shape}")
    
    # 执行前向传播
    try:
        output_tensor = model(input_tensor)
        print(f"输出张量形状: {output_tensor.shape}")
        
        # 检查输出形状是否与输入形状匹配
        assert output_tensor.shape == input_tensor.shape
        
        print("\n✓ U-Net前向传播测试成功!")
        
    except Exception as e:
        print(f"\n✗ U-Net前向传播测试失败: {e}")
        
    print("=" * 50)

if __name__ == '__main__':
    test_unet_forward_pass()
