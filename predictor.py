import torch
import re
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
        """纠正文本中的拼写错误（移除所有空格）"""
        # 对输入文本进行编码
        inputs = self.tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        # 将输入移动到指定设备
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
        # 核心修改：移除所有空格（包括空格、制表符、换行符等空白字符）
        corrected_text = re.sub(r'\s', '', corrected_text)
        return corrected_text

    def compare_text(self, original, corrected):
        """
        基于最长公共子序列（LCS）对比原始文本与纠错文本，精准提取差异
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

        # 计算LCS及其在原始文本和纠正文本中的索引
        lcs, lcs_indices = self._get_lcs_with_indices(original, corrected)
        if not lcs:
            # 无公共子序列，视为整体替换
            differences.append({
                "type": "替换",
                "original": original,
                "corrected": corrected,
                "start_original": 0,
                "end_original": len(original),
                "start_corrected": 0,
                "end_corrected": len(corrected)
            })
            return {"original": original, "corrected": corrected, "differences": differences}

        # 基于LCS索引划分差异片段
        i = j = k = 0  # i: original指针, j: corrected指针, k: lcs指针
        len_original, len_corrected = len(original), len(corrected)
        len_lcs = len(lcs)

        while k < len_lcs:
            # 下一个LCS字符在original和corrected中的位置
            orig_pos = lcs_indices["original"][k]
            corr_pos = lcs_indices["corrected"][k]

            # 处理original中[当前i, orig_pos)的删除部分
            if i < orig_pos:
                differences.append({
                    "type": "删除",
                    "original": original[i:orig_pos],
                    "corrected": "",
                    "start_original": i,
                    "end_original": orig_pos,
                    "start_corrected": j,
                    "end_corrected": j
                })
                i = orig_pos

            # 处理corrected中[当前j, corr_pos)的新增部分
            if j < corr_pos:
                differences.append({
                    "type": "新增",
                    "original": "",
                    "corrected": corrected[j:corr_pos],
                    "start_original": i,
                    "end_original": i,
                    "start_corrected": j,
                    "end_corrected": corr_pos
                })
                j = corr_pos

            # 移动指针到LCS字符后
            i += 1
            j += 1
            k += 1

        # 处理original剩余部分（删除）
        if i < len_original:
            differences.append({
                "type": "删除",
                "original": original[i:],
                "corrected": "",
                "start_original": i,
                "end_original": len_original,
                "start_corrected": j,
                "end_corrected": j
            })

        # 处理corrected剩余部分（新增）
        if j < len_corrected:
            differences.append({
                "type": "新增",
                "original": "",
                "corrected": corrected[j:],
                "start_original": i,
                "end_original": i,
                "start_corrected": j,
                "end_corrected": len_corrected
            })

        # 合并连续的同类型差异（优化显示）
        merged_diffs = self._merge_differences(differences)
        return {
            "original": original,
            "corrected": corrected,
            "differences": merged_diffs
        }

    def _get_lcs_with_indices(self, s1, s2):
        """计算最长公共子序列（LCS），并返回其在s1和s2中的索引"""
        m, n = len(s1), len(s2)
        # 构建DP表
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        # 回溯获取LCS及索引
        lcs = []
        i, j = m, n
        indices_s1 = []  # LCS字符在s1中的索引
        indices_s2 = []  # LCS字符在s2中的索引

        while i > 0 and j > 0:
            if s1[i - 1] == s2[j - 1]:
                lcs.append(s1[i - 1])
                indices_s1.append(i - 1)
                indices_s2.append(j - 1)
                i -= 1
                j -= 1
            elif dp[i - 1][j] > dp[i][j - 1]:
                i -= 1
            else:
                j -= 1

        # 反转以恢复顺序
        lcs.reverse()
        indices_s1.reverse()
        indices_s2.reverse()

        return lcs, {"original": indices_s1, "corrected": indices_s2}

    def _merge_differences(self, differences):
        """合并连续的同类型差异（优化显示）"""
        if not differences:
            return []
        merged = [differences[0]]
        for curr in differences[1:]:
            last = merged[-1]
            # 若类型相同且位置连续，合并
            if (last["type"] == curr["type"] and
                last["end_original"] == curr["start_original"] and
                last["end_corrected"] == curr["start_corrected"]):
                merged[-1] = {
                    "type": last["type"],
                    "original": last["original"] + curr["original"],
                    "corrected": last["corrected"] + curr["corrected"],
                    "start_original": last["start_original"],
                    "end_original": curr["end_original"],
                    "start_corrected": last["start_corrected"],
                    "end_corrected": curr["end_corrected"]
                }
            else:
                merged.append(curr)
        return merged
