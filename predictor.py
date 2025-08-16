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
        """
        对比原始文本与纠错文本，精准提取差异（支持不等长、多差异场景）
        :param original: 原始输入文本
        :param corrected: 纠错后文本
        :return: 包含原始文本、纠错文本、差异详情的字典
        """
        differences = []
        # 处理空文本特殊情况
        if not original and not corrected:
            return {"original": original, "corrected": corrected, "differences": differences}
        if not original:
            differences.append({"type": "新增", "content": corrected, "start": 0, "end": len(corrected)})
            return {"original": original, "corrected": corrected, "differences": differences}
        if not corrected:
            differences.append({"type": "删除", "content": original, "start": 0, "end": len(original)})
            return {"original": original, "corrected": corrected, "differences": differences}

        # 双指针遍历找差异（兼容不等长文本）
        i = j = 0  # i: 原始文本指针  j: 纠错文本指针
        len_original, len_corrected = len(original), len(corrected)
        while i < len_original and j < len_corrected:
            if original[i] == corrected[j]:
                i += 1
                j += 1
            else:
                # 标记差异起始位置
                start_original, start_corrected = i, j
                # 寻找差异结束位置（向长文本遍历）
                while i < len_original and j < len_corrected and original[i] != corrected[j]:
                    i += 1
                    j += 1
                # 处理剩余未匹配内容（原始文本有剩余则为删除，纠错文本有剩余则为新增）
                end_original, end_corrected = i, j
                # 提取差异片段
                original_diff = original[start_original:end_original] if start_original < end_original else ""
                corrected_diff = corrected[start_corrected:end_corrected] if start_corrected < end_corrected else ""

                # 判定差异类型（删除/新增/替换）
                diff_type = ""
                if original_diff and corrected_diff:
                    diff_type = "替换"
                elif original_diff:
                    diff_type = "删除"
                else:
                    diff_type = "新增"

                differences.append({
                    "type": diff_type,
                    "original": original_diff,
                    "corrected": corrected_diff,
                    "start_original": start_original,
                    "end_original": end_original,
                    "start_corrected": start_corrected,
                    "end_corrected": end_corrected
                })

        # 处理原始文本剩余内容（删除操作）
        while i < len_original:
            differences.append({
                "type": "删除",
                "original": original[i:],
                "corrected": "",
                "start_original": i,
                "end_original": len_original,
                "start_corrected": j,
                "end_corrected": j
            })
            i += 1

        # 处理纠错文本剩余内容（新增操作）
        while j < len_corrected:
            differences.append({
                "type": "新增",
                "original": "",
                "corrected": corrected[j:],
                "start_original": i,
                "end_original": i,
                "start_corrected": j,
                "end_corrected": len_corrected
            })
            j += 1

        return {
            "original": original,
            "corrected": corrected,
            "differences": differences
        }
