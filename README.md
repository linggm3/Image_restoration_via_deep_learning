# Image restoration via deep learning

## Project Structure

```
image-restoration-project/
├── README.md
├── requirements.txt
├── config/
│   ├── unet_config.yaml          # U-Net配置
│   └── transformer_config.yaml    # Transformer配置
│
├── data/
│   ├── __init__.py
│   ├── dataset.py                 # 核心：动态退化数据集类
│   ├── transforms.py              # 退化算子实现
│   └── dataloader.py              # 数据加载逻辑
│
├── models/
│   ├── __init__.py
│   ├── unet.py                    # 条件化U-Net
│   ├── transformer.py             # 条件化Transformer (SwinIR-like)
│   ├── base_model.py              # 模型基类
│   └── layers/
│       ├── __init__.py
│       ├── conditional_layers.py  # AdaIN等条件注入层
│       └── attention.py           # 注意力模块
│
├── training/
│   ├── __init__.py
│   ├── trainer.py                 # 训练循环
│   ├── losses.py                  # 损失函数集合
│   └── optimizer.py               # 优化器配置
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                 # PSNR, SSIM, LPIPS实现
│   └── evaluator.py               # 评估流程
│
├── utils/
│   ├── __init__.py
│   ├── logger.py                  # 日志和可视化
│   ├── checkpoint.py              # 模型保存/加载
│   └── visualization.py           # 结果可视化
│
├── experiments/
│   ├── train_unet.py              # U-Net训练脚本
│   ├── train_transformer.py       # Transformer训练脚本
│   └── inference.py               # 推理脚本
│
├── notebooks/
│   ├── data_exploration.ipynb     # 数据探索
│   └── results_analysis.ipynb     # 结果分析
│
└── outputs/
    ├── checkpoints/               # 模型权重
    ├── logs/                      # 训练日志
    └── visualizations/            # 可视化结果
```

## Implementation plan

#### `data/transform.py`
```python
核心功能：
- GaussianBlur(sigma_range=[0.5, 5.0])
- HistogramEqualization(mode=['global', 'adaptive'])  
- GammaCorrection(gamma_range=[0.4, 2.5])
- 可扩展到更多算子（锐化、噪声等）

设计要点：
- 每个算子类返回：(退化图像, 条件向量)
- 条件向量统一格式：[op_type_onehot, param_values]
  例如：[1,0,0, 2.5, 0, 0] → 高斯模糊sigma=2.5
```

#### `data/dataset.py`
```python
class DynamicDegradationDataset(Dataset):
    核心逻辑：
    1. __getitem__时读取COCO原图
    2. 随机选择一种退化算子
    3. 随机采样该算子的参数
    4. 实时应用退化生成图像对
    5. 返回：{
           'degraded': 退化图像,
           'original': 原图,
           'condition': 条件向量,
           'degradation_info': 元数据(可选，用于分析)
       }
    
    优势：
    - 无需预存储，节省磁盘空间
    - 每个epoch数据都不同，天然防过拟合
    - 易于调整退化参数分布
```

#### `models/conditional_layers.py`

```
方案A：简单拼接
输入: [B, 3, H, W]
条件: [B, C_cond] → 扩展为 [B, C_cond, H, W]
拼接: concat → [B, 3+C_cond, H, W]
送入网络首层

方案B：AdaIN调制
class ConditionalAdaIN(nn.Module):
    在每个残差块或Transformer块中：
    1. 特征图归一化
    2. 从条件向量预测γ和β
    3. 仿射变换：output = γ * normalized + β
    
    优势：动态调制整个网络，表达能力更强
```


#### `models/unet.py`
```
架构设计：
- Encoder: 5层下采样 (64→128→256→512→512)
- Decoder: 5层上采样 + 跳跃连接
- 条件注入点：
  · 输入层拼接
  · 每个残差块使用AdaIN
  · Bottleneck额外条件融合

特点：
- 轻量级（~30M参数）
- 训练快速
- 适合作为Baseline
```

#### `models/transformer.py` 
```
参考架构：SwinIR / Restormer

设计要点：
- Patch Embedding层接受条件输入
- 使用Shifted Window Attention（计算高效）
- 条件通过Cross-Attention或AdaIN注入每个Transformer块
- 保留浅层卷积特征提取

配置建议：
- 6-8个Transformer块
- 窗口大小8x8
- 多头注意力(6-8 heads)
- 约50M参数
```

#### `training/losses.py`
```python
推荐组合：
L_total = λ1*L_L1 + λ2*L_Perceptual + λ3*L_GAN(可选)

1. L1 Loss (λ1=1.0)
   - 像素级保真度基础

2. Perceptual Loss (λ2=0.1)
   - VGG-19的relu1_1, relu2_1, relu3_1特征
   - 保证纹理和结构相似性

3. GAN Loss (λ3=0.01, 可选)
   - 仅用于U-Net，提升真实感
   - PatchGAN判别器（70x70感受野）

注意：Transformer通常不需要GAN，其自身重建能力已足够强
```

#### 'training/trainer.py'
```python
核心功能：
- 支持混合精度训练（AMP）
- 学习率Warmup + CosineAnnealing
- 梯度裁剪（防止训练不稳定）
- 每N步验证集评估
- TensorBoard日志
- 自动保存最佳模型

训练策略：
- Batch size: 16-32（取决于GPU）
- 优化器: AdamW (lr=2e-4, weight_decay=1e-4)
- Epochs: 100-200
- 验证频率: 每1000步
```

#### `evaluation/metrics.py`
```python
必须实现：
1. PSNR (Peak Signal-to-Noise Ratio)
   - 报告总体均值
   - 按退化类型分别统计

2. SSIM (Structural Similarity)
   - 使用skimage或pytorch-msssim

3. LPIPS (Learned Perceptual Image Patch Similarity)
   - 使用官方实现
   - 这是感知质量的金标准

额外分析：
- 按退化类型细分指标
- 按退化参数强度细分（如不同sigma值）
- 可视化对比（原图/退化/恢复）
```

##  Sequence of implementatio
```
第1步: 环境配置和项目初始化
第2步: 实现退化算子（transforms.py）
第3步: 实现动态数据集（dataset.py）
第4步: 测试数据流程
第5步: 实现评估指标（metrics.py）
第6步: 实现基础U-Net（不带条件）
第7步: 实现条件化层（conditional_layers.py）
第8步: 实现条件化U-Net
第9步: 实现训练器（trainer.py）
第10步: 实现损失函数（losses.py）
第11步: 训练和评估U-Net
第12步: 实现Transformer模型
第13步: 完整实验和分析
```
