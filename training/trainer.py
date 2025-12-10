# training/trainer.py

import torch
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from evaluation.metrics import Evaluator

class Trainer:
    """
    通用训练器类，支持条件和非条件模型
    """
    def __init__(self, model, train_loader, val_loader, loss_fn, optimizer, scheduler, config, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        
        # 混合精度训练
        self.use_amp = config['training']['mixed_precision']
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        
        # 日志和保存路径
        self.base_dir = f"../outputs/{config['model']['name']}"
        self.log_dir = os.path.join(self.base_dir, 'logs')
        self.checkpoint_dir = os.path.join(self.base_dir, 'checkpoints')
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.writer = SummaryWriter(self.log_dir)
        self.evaluator = Evaluator(metrics=config['evaluation']['metrics'])
        
        self.start_epoch = 0
        self.global_step = 0
        self.best_val_metric = float('inf')  # LPIPS越小越好
        
        # 统计信息
        self.degradation_stats = {
            'motion_blur': 0,
            'gaussian_blur': 0,
            'gaussian_noise': 0,
            'jpeg_compression': 0,
            'downsampling': 0
        }

    def _is_conditional(self):
        """判断模型是否为条件模型"""
        return self.config['model'].get('use_conditional', False) or 'conditional' in self.config['model']['name']

    def _update_degradation_stats(self, batch):
        """更新退化统计信息"""
        for info in batch.get('degradation_info', []):
            for deg_name in info.get('degradation_names', []):
                if deg_name in self.degradation_stats:
                    self.degradation_stats[deg_name] += 1

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_loader, 
            desc=f"Epoch {epoch+1}/{self.config['training']['epochs']}", 
            leave=False
        )
        
        for batch in progress_bar:
            degraded = batch['degraded'].to(self.device)
            original = batch['original'].to(self.device)
            
            # 更新退化统计
            self._update_degradation_stats(batch)
            
            self.optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                # 区分条件模型和非条件模型
                if self._is_conditional():
                    condition = batch['condition'].to(self.device)
                    restored = self.model(degraded, condition)
                else:
                    restored = self.model(degraded)
                
                total_loss, loss_dict = self.loss_fn(restored, original)
            
            self.scaler.scale(total_loss).backward()
            
            # 梯度裁剪
            if self.config['training']['gradient_clip'] > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['training']['gradient_clip']
                )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            epoch_loss += total_loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'avg_loss': f"{epoch_loss/num_batches:.4f}"
            })
            
            # 记录日志
            if self.global_step % self.config['training']['log_interval'] == 0:
                self.writer.add_scalar('Loss/train_total', total_loss.item(), self.global_step)
                for name, value in loss_dict.items():
                    if name != 'total_loss':
                        self.writer.add_scalar(f'Loss/train_{name}', value.item(), self.global_step)
                self.writer.add_scalar('LearningRate', self.optimizer.param_groups[0]['lr'], self.global_step)

            self.global_step += 1
        
        avg_loss = epoch_loss / num_batches
        return avg_loss

    def validate_epoch(self, epoch):
        """验证一个epoch"""
        self.model.eval()
        self.evaluator.reset()
        val_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating", leave=False):
                degraded = batch['degraded'].to(self.device)
                original = batch['original'].to(self.device)

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    if self._is_conditional():
                        condition = batch['condition'].to(self.device)
                        restored = self.model(degraded, condition)
                    else:
                        restored = self.model(degraded)
                    
                    # 计算验证损失
                    total_loss, _ = self.loss_fn(restored, original)
                    val_loss += total_loss.item()
                    num_batches += 1
                
                # 更新评估指标
                self.evaluator.update(restored, original)
        
        # 计算平均指标
        mean_results = self.evaluator.get_mean_results()
        avg_val_loss = val_loss / num_batches
        
        # 记录到TensorBoard
        self.writer.add_scalar('Loss/val_total', avg_val_loss, self.global_step)
        for metric, value in mean_results.items():
            self.writer.add_scalar(f'Metrics/val_{metric}', value, self.global_step)
        
        # 打印结果
        print(f"\nEpoch {epoch+1} Validation Results:")
        print(f"  Loss: {avg_val_loss:.4f}")
        for metric, value in mean_results.items():
            print(f"  {metric.upper()}: {value:.4f}")
        
        # 保存最佳模型 (以LPIPS为准，越小越好)
        current_metric = mean_results.get('lpips', float('inf'))
        if current_metric < self.best_val_metric:
            self.best_val_metric = current_metric
            self.save_checkpoint(epoch, "best.pth")
            print(f"  ✓ New best model saved! LPIPS: {current_metric:.4f}")
        
        return mean_results
    
    def train(self):
        """完整的训练流程"""
        print("\n" + "=" * 70)
        print(f"开始训练 - 模型: {self.config['model']['name']}")
        print(f"条件模型: {self._is_conditional()}")
        print(f"训练集大小: {len(self.train_loader.dataset)}")
        print(f"验证集大小: {len(self.val_loader.dataset)}")
        print("=" * 70 + "\n")
        
        for epoch in range(self.start_epoch, self.config['training']['epochs']):
            # 训练
            avg_loss = self.train_epoch(epoch)
            
            # 验证
            val_results = self.validate_epoch(epoch)
            
            # 更新学习率
            self.scheduler.step()
            
            # 打印退化统计
            if (epoch + 1) % 5 == 0:
                print(f"\n退化类型统计（累计）:")
                total = sum(self.degradation_stats.values())
                for deg_name, count in sorted(self.degradation_stats.items()):
                    percentage = (count / total * 100) if total > 0 else 0
                    print(f"  {deg_name:20s}: {count:6d} ({percentage:5.1f}%)")
            
            # 定期保存检查点
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch, f"epoch_{epoch+1}.pth")
            
            print("-" * 70)
        
        # 训练结束
        self.writer.close()
        print("\n" + "=" * 70)
        print("🎉 训练完成!")
        print(f"最佳验证 LPIPS: {self.best_val_metric:.4f}")
        print(f"检查点保存在: {self.checkpoint_dir}")
        print("=" * 70 + "\n")

    def save_checkpoint(self, epoch, filename):
        """保存模型检查点"""
        state = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_metric': self.best_val_metric,
            'config': self.config,
            'degradation_stats': self.degradation_stats
        }
        save_path = os.path.join(self.checkpoint_dir, filename)
        torch.save(state, save_path)
        
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_val_metric = checkpoint['best_val_metric']
        if 'degradation_stats' in checkpoint:
            self.degradation_stats = checkpoint['degradation_stats']
        print(f"✓ 检查点已加载: {checkpoint_path}")
        print(f"  从 Epoch {self.start_epoch} 继续训练")
