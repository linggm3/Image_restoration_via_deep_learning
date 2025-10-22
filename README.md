# Image restoration via deep learning

## 项目结构规划

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
