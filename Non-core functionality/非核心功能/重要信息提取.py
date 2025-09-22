import re
import pymupdf
from docx import Document
from collections import defaultdict


class AdvancedContractExtractor:
    def __init__(self):
        # 同义词库
        self.synonym_dict = {
            "甲方": ["甲方", "委托人"],
            "乙方": ["乙方", "受托人"],
            "合同金额": ["金额", "合同金额", "总金额", "价款"],
            "履行期限": ["委托期限", "合同期限", "有效期", "履行期限"],
            "违约责任": ["违约责任", "违约条款"]
        }

        # 关键词权重
        self.keyword_weights = {
            "甲方": {"甲方": 1.0, "委托人": 0.9},
            "乙方": {"乙方": 1.0, "受托人": 0.9},
            "合同金额": {"金额": 0.8, "合同金额": 1.0, "总金额": 0.9, "价款": 0.7},
            "履行期限": {"委托期限": 1.0, "合同期限": 0.9, "有效期": 0.8, "履行期限": 0.95},
            "违约责任": {"违约责任": 1.0, "违约条款": 0.9}
        }

        self.confidence_threshold = 0.7

    # 文件读取方法
    def read_word_file(self, file_path):
        try:
            doc = Document(file_path)
            return '\n'.join([para.text for para in doc.paragraphs])
        except Exception as e:
            print(f"读取Word文件出错: {str(e)}")
            return ""

    def read_pdf_file(self, file_path):
        try:
            doc = pymupdf.open(file_path)
            return '\n'.join([page.get_text() for page in doc])
        except Exception as e:
            print(f"读取PDF文件出错: {str(e)}")
            return ""

    def read_contract_file(self, file_path):
        if file_path.lower().endswith('.docx'):
            return self.read_word_file(file_path)
        elif file_path.lower().endswith('.pdf'):
            return self.read_pdf_file(file_path)
        else:
            print("不支持的文件格式，仅支持.docx和.pdf")
            return ""

    def extract_entities(self, text):
        """优化后的提取逻辑，修复已知问题"""
        results = {}

        # 1. 提取甲方/乙方（保持不变）
        party_pattern = re.compile(r'(甲方|乙方)(\（[^）]*\）)?\s*[:：]\s*([^\n]+)')
        parties = party_pattern.findall(text)
        for label, _, name in parties:
            name = name.strip()
            if 0 < len(name) < 100:
                if label == "甲方":
                    results["甲方"] = {"value": name, "confidence": 1.00}
                elif label == "乙方":
                    results["乙方"] = {"value": name, "confidence": 1.00}
                if "甲方" in results and "乙方" in results:
                    break

        # 2. 提取合同金额（保持不变）
        amount_pattern = re.compile(r'(金额|合同金额|总金额|价款)\s*[:：]?\s*([^\n]+)')
        amount_match = amount_pattern.search(text)
        if amount_match:
            value = amount_match.group(2).strip()
            num_match = re.search(r'(\d+(?:,\d+)*(?:\.\d+)?)', value)
            if num_match:
                results["合同金额"] = {"value": num_match.group(1) + "元", "confidence": 0.80}
            else:
                results["合同金额"] = {"value": value, "confidence": 0.80}

        # 3. 优化履行期限提取（支持空值提示 + 更灵活的格式）
        term_pattern = re.compile(
            r'(委托期限|合同期限|有效期|履行期限)\s*[:：]?\s*自\s*([^\n]*?)\s*至\s*([^\n]+?)[。止]?',
            re.IGNORECASE
        )
        term_match = term_pattern.search(text)
        if term_match:
            start_date = term_match.group(2).strip() or "未填写"
            end_date = term_match.group(3).strip() or "未填写"
            results["履行期限"] = {
                "value": {"start_date": start_date, "end_date": end_date},
                "confidence": 1.00 if (start_date != "未填写" or end_date != "未填写") else 0.5
            }

        # 4. 优化违约责任提取（限制到下一条款前）
        liability_pattern = re.compile(
            r'(违约责任|违约条款)\s*[:：]?\s*([\s\S]*?)(?=第\d+条\s*(?!违约)|甲方（签名|乙方（签名|$)',
            re.IGNORECASE
        )
        liability_match = liability_pattern.search(text)
        if liability_match:
            liability_content = liability_match.group(2).strip()
            liability_content = re.sub(r'\s+', ' ', liability_content)  # 清理多余空白
            # 移除后续条款内容
            liability_content = re.split(r'第\d+条\s*(?!违约)', liability_content)[0]
            results["违约责任"] = {
                "value": liability_content,
                "confidence": 0.90
            }

        # 5. 修复签名提取（支持甲方签名 + 更宽松的格式）
        signature_pattern = re.compile(
            r'甲方（签名/盖章）：\s*([^\n]*?)\s*'
            r'乙方（签名/盖章）：\s*([^\n]*?)\s*',
            re.DOTALL
        )
        signature_match = signature_pattern.search(text)
        if signature_match:
            # 甲方签名（支持空值）
            if signature_match.group(1).strip() or signature_match.group(1).strip() == "":
                results["甲方签名"] = {
                    "value": signature_match.group(1).strip() or "未签名",
                    "confidence": 0.95 if signature_match.group(1).strip() else 0.6
                }
            # 乙方签名
            if signature_match.group(2).strip() or signature_match.group(2).strip() == "":
                results["乙方签名"] = {
                    "value": signature_match.group(2).strip() or "未签名",
                    "confidence": 0.95 if signature_match.group(2).strip() else 0.6
                }

        # 6. 补充法定代表人、委托代理人和日期提取
        rep_pattern = re.compile(
            r'法定代表人：\s*([^\n]*?)\s*'
            r'法定代表人：\s*([^\n]*?)\s*'
            r'委托代理人：\s*([^\n]*?)\s*'
            r'委托代理人：\s*([^\n]*?)\s*'
            r'(\d*)\s*年\s*(\d*)\s*月\s*(\d*)\s*日\s*'
            r'(\d*)\s*年\s*(\d*)\s*月\s*(\d*)\s*日\s*',
            re.DOTALL
        )
        rep_match = rep_pattern.search(text)
        if rep_match:
            # 法定代表人
            results["甲方法定代表人"] = {"value": rep_match.group(1).strip() or "未填写", "confidence": 0.90}
            results["乙方法定代表人"] = {"value": rep_match.group(2).strip() or "未填写", "confidence": 0.90}
            # 委托代理人
            results["甲方委托代理人"] = {"value": rep_match.group(3).strip() or "未填写", "confidence": 0.90}
            results["乙方委托代理人"] = {"value": rep_match.group(4).strip() or "未填写", "confidence": 0.90}
            # 签署日期
            if rep_match.group(5) or rep_match.group(6) or rep_match.group(7):
                results["甲方签署日期"] = {
                    "value": f"{rep_match.group(5)}年{rep_match.group(6)}月{rep_match.group(7)}日" if (rep_match.group(5) or rep_match.group(6) or rep_match.group(7)) else "未填写",
                    "confidence": 0.95
                }
            if rep_match.group(8) or rep_match.group(9) or rep_match.group(10):
                results["乙方签署日期"] = {
                    "value": f"{rep_match.group(8)}年{rep_match.group(9)}月{rep_match.group(10)}日" if (rep_match.group(8) or rep_match.group(9) or rep_match.group(10)) else "未填写",
                    "confidence": 0.95
                }

        return results

    def process_contract_file(self, file_path):
        text = self.read_contract_file(file_path)
        if not text:
            return None
        return self.extract_entities(text)


# 使用示例
if __name__ == "__main__":
    extractor = AdvancedContractExtractor()
    file_path = "111.docx"
    print(f"正在处理文件: {file_path}")
    result = extractor.process_contract_file(file_path)

    print("\n===== 合同关键要素提取结果 =====")
    for key, info in result.items():
        # 格式化显示
        if key == "履行期限":
            value = f"自 {info['value']['start_date']} 至 {info['value']['end_date']}"
        else:
            value = info['value']
        print(f"{key}：{value}")
        print(f"   置信度：{info['confidence']:.2f}\n")