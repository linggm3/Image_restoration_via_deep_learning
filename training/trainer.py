# training/trainer.py

import torch
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from evaluation.metrics import Evaluator

class Trainer:
    """
    通用训练器类
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
        self.best_val_metric = float('inf') # LPIPS越小越好

    def _is_conditional(self):
        """判断模型是否为条件模型"""
        return 'conditional' in self.config['model']['name']

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config['training']['epochs']}", leave=False)
        
        for batch in progress_bar:
            degraded = batch['degraded'].to(self.device)
            original = batch['original'].to(self.device)
            
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
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['gradient_clip'])
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            progress_bar.set_postfix(loss=f"{total_loss.item():.4f}")
            
            # 记录日志
            if self.global_step % self.config['training']['log_interval'] == 0:
                self.writer.add_scalar('Loss/train_total', total_loss.item(), self.global_step)
                for name, value in loss_dict.items():
                    self.writer.add_scalar(f'Loss/train_{name}', value.item(), self.global_step)
                self.writer.add_scalar('LearningRate', self.scheduler.get_last_lr()[0], self.global_step)

            self.global_step += 1

    def validate_epoch(self, epoch):
        """验证一个epoch"""
        self.model.eval()
        self.evaluator.reset()
        
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
                
                self.evaluator.update(restored, original)
        
        mean_results = self.evaluator.get_mean_results()
        
        # 记录到TensorBoard
        for metric, value in mean_results.items():
            self.writer.add_scalar(f'Metrics/val_{metric}', value, self.global_step)
        
        print(f"Epoch {epoch} Validation Results: {mean_results}")
        
        # 保存最佳模型 (以LPIPS为准)
        current_metric = mean_results.get('lpips', float('inf'))
        if current_metric < self.best_val_metric:
            self.best_val_metric = current_metric
            self.save_checkpoint(epoch, "best.pth")
            print(f"New best model saved at epoch {epoch} with LPIPS: {current_metric:.4f}")
    
    def train(self):
        """完整的训练流程"""
        for epoch in range(self.start_epoch, self.config['training']['epochs']):
            self.train_epoch(epoch)
            self.validate_epoch(epoch)
            self.scheduler.step()
            
            # 定期保存检查点
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch, f"epoch_{epoch}.pth")
        
        self.writer.close()
        print("训练完成!")

    def save_checkpoint(self, epoch, filename):
        """保存模型检查点"""
        state = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_metric': self.best_val_metric,
        }
        torch.save(state, os.path.join(self.checkpoint_dir, filename))
        print(f"Checkpoint saved to {os.path.join(self.checkpoint_dir, filename)}")
