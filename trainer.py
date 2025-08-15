import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup


class Trainer:
    def __init__(self, model, train_loader, test_loader, tokenizer,
                 epochs=10, save_dir='models', device='cpu'):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.tokenizer = tokenizer
        self.epochs = epochs
        self.save_dir = save_dir
        self.device = device

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        # 初始化优化器和学习率调度器
        self.optimizer = AdamW(model.parameters(), lr=5e-5, eps=1e-8)
        total_steps = len(train_loader) * epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        # 记录最佳性能
        self.best_val_loss = float('inf')

    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0

        # 使用tqdm显示进度
        progress_bar = tqdm(self.train_loader, desc='训练中')
        for batch in progress_bar:
            # 将数据移动到GPU
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # 清零梯度
            self.model.zero_grad()

            # 前向传播
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            total_loss += loss.item()

            # 反向传播和优化
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # 梯度裁剪
            self.optimizer.step()
            self.scheduler.step()

            # 更新进度条
            progress_bar.set_postfix({'batch_loss': loss.item()})

        # 计算平均损失
        avg_train_loss = total_loss / len(self.train_loader)
        return avg_train_loss

    def evaluate(self):
        """评估模型性能"""
        self.model.eval()
        total_loss = 0

        # 在评估时不计算梯度
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc='评估中'):
                # 将数据移动到GPU
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                # 前向传播
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                total_loss += loss.item()

        # 计算平均损失
        avg_val_loss = total_loss / len(self.test_loader)
        return avg_val_loss

    def train(self):
        """完整训练流程"""
        for epoch in range(self.epochs):
            print(f"\n===== 第 {epoch + 1}/{self.epochs} 轮 =====")

            # 训练
            train_loss = self.train_epoch()
            print(f"训练损失: {train_loss:.4f}")

            # 评估
            val_loss = self.evaluate()
            print(f"验证损失: {val_loss:.4f}")

            # 保存最佳模型
            if val_loss < self.best_val_loss:
                print(f"验证损失降低 ({self.best_val_loss:.4f} → {val_loss:.4f})，保存模型...")
                self.best_val_loss = val_loss
                self.model.save_pretrained(f"{self.save_dir}/best_model")
                self.tokenizer.save_pretrained(f"{self.save_dir}/best_model")

            # 保存最后一轮模型
            self.model.save_pretrained(f"{self.save_dir}/last_model")
            self.tokenizer.save_pretrained(f"{self.save_dir}/last_model")

        print(f"\n训练完成! 最佳验证损失: {self.best_val_loss:.4f}")
