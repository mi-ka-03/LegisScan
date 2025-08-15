import torch
from transformers import BartForConditionalGeneration, BertTokenizer


class LegalSpellingCorrector:
    def __init__(self, model_path, device='cpu'):
        """初始化拼写纠错器"""
        self.device = device
        # 加载模型和tokenizer
        self.model = BartForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)

        # 将模型移动到指定设备并设置为评估模式
        self.model = self.model.to(device)
        self.model.eval()

    def correct_text(self, text, max_length=128):
        """纠正文本中的拼写错误"""
        # 对输入文本进行编码
        inputs = self.tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # 将输入移动到GPU
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        # 生成纠正后的文本（不计算梯度以加速）
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=5,  # 束搜索提高生成质量
                early_stopping=True
            )

        # 解码生成的文本
        corrected_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        return corrected_text

    def compare_text(self, original, corrected):
        """比较原始文本和纠正后的文本，找出差异"""
        # 简单的差异比较（实际应用中可使用更复杂的算法）
        differences = []
        if original != corrected:
            # 这里使用简单的字符级比较，实际应用可使用编辑距离算法
            min_len = min(len(original), len(corrected))
            i = 0
            while i < min_len:
                if original[i] != corrected[i]:
                    # 找到差异起始位置
                    start = i
                    # 找到差异结束位置
                    while i < min_len and original[i] != corrected[i]:
                        i += 1
                    end = i
                    # 提取差异部分
                    original_diff = original[start:end]
                    corrected_diff = corrected[start:end]
                    differences.append({
                        'start': start,
                        'end': end,
                        'original': original_diff,
                        'corrected': corrected_diff
                    })
                else:
                    i += 1

        return {
            'original': original,
            'corrected': corrected,
            'differences': differences
        }
