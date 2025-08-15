import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split


class LegalSpellingDataset(Dataset):
    """法律文书拼写纠错数据集"""

    def __init__(self, source_texts, target_texts, tokenizer, max_len=128):
        self.source_texts = source_texts
        self.target_texts = target_texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.source_texts)

    def __getitem__(self, idx):
        source_text = str(self.source_texts[idx])
        target_text = str(self.target_texts[idx])

        # 对源文本和目标文本进行编码
        source_encoding = self.tokenizer(
            source_text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # 转换为张量并压缩维度
        input_ids = source_encoding['input_ids'].flatten()
        attention_mask = source_encoding['attention_mask'].flatten()
        labels = target_encoding['input_ids'].flatten()

        # 将填充部分的标签设为-100，这样在计算损失时会被忽略
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


def load_legal_dataset(file_path):
    """加载法律文书数据集"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 分割列（处理可能包含空格的文本）
            parts = line.split('\t', 2)  # 只分割前两部分
            if len(parts) == 3:
                data.append({
                    'error_count': parts[0],
                    'source': parts[1],
                    'target': parts[2]
                })
    return pd.DataFrame(data)


def get_data_loaders(train_path, test_path, batch_size=8, max_len=128):
    """获取训练和测试数据加载器"""
    # 加载数据集
    train_df = load_legal_dataset(train_path)
    test_df = load_legal_dataset(test_path)

    # 初始化tokenizer
    tokenizer = BertTokenizer.from_pretrained('fnlp/bart-base-chinese')

    # 创建数据集实例
    train_dataset = LegalSpellingDataset(
        train_df['source'].tolist(),
        train_df['target'].tolist(),
        tokenizer,
        max_len=max_len
    )

    test_dataset = LegalSpellingDataset(
        test_df['source'].tolist(),
        test_df['target'].tolist(),
        tokenizer,
        max_len=max_len
    )

    # 自动选择是否使用GPU加速的数据加载
    device_count = torch.cuda.device_count()
    pin_memory = device_count > 0  # 如果有GPU则启用内存锁定

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4 if device_count > 0 else 0,  # GPU时使用多进程加载
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4 if device_count > 0 else 0,
        pin_memory=pin_memory
    )

    return train_loader, test_loader, tokenizer
