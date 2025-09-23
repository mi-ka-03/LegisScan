import fitz  # PyMuPDF用于处理PDF
import docx  # python-docx用于处理Word
import difflib
import os
from datetime import datetime
import re


class EnhancedContractComparator:
    def __init__(self):
        """初始化增强版合同比对器"""
        # 自定义HTML模板，修复了CSS格式问题
        self.html_template = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>合同比对结果</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        .meta-info {{ margin: 20px 0; color: #666; }}
        .comparison-container {{ display: flex; gap: 20px; margin-top: 20px; }}
        .file-section {{ flex: 1; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .file-title {{ background-color: #f8f9fa; padding: 10px; margin: -15px -15px 15px; border-bottom: 1px solid #ddd; font-weight: bold; }}
        .diff-added {{ background-color: #d4edda; padding: 2px 4px; border-radius: 3px; }}
        .diff-removed {{ background-color: #f8d7da; text-decoration: line-through; padding: 2px 4px; border-radius: 3px; }}
        .diff-changed {{ background-color: #fff3cd; padding: 2px 4px; border-radius: 3px; }}
        .section-header {{ background-color: #e9f5ff; padding: 8px; margin: 10px 0; font-weight: bold; }}
        .no-change {{ color: #666; font-style: italic; }}
        .change-summary {{ background-color: #f1f8e9; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .summary-item {{ margin: 5px 0; }}
    </style>
</head>
<body>
    <h1>合同比对报告</h1>
    <div class="meta-info">
        <p>比对时间: {timestamp}</p>
        <p>原始文件: {file1_name}</p>
        <p>新文件: {file2_name}</p>
    </div>

    <div class="change-summary">
        <h2>修改摘要</h2>
        <div class="summary-item">• 新增内容: {additions_count} 处</div>
        <div class="summary-item">• 删除内容: {deletions_count} 处</div>
        <div class="summary-item">• 修改内容: {changes_count} 处</div>
    </div>

    <h2>详细比对</h2>
    <div class="comparison-container">
        <div class="file-section">
            <div class="file-title">原始文件内容: {file1_name}</div>
            <div class="content">{original_content}</div>
        </div>
        <div class="file-section">
            <div class="file-title">新文件内容: {file2_name}</div>
            <div class="content">{new_content}</div>
        </div>
    </div>
</body>
</html>
        """

    def extract_text_from_pdf(self, pdf_path):
        """从PDF文件中提取文本"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            return self.clean_text(text)
        except Exception as e:
            print(f"提取PDF文本时出错: {str(e)}")
            return None

    def extract_text_from_word(self, word_path):
        """从Word文件中提取文本"""
        try:
            doc = docx.Document(word_path)
            full_text = []
            for para in doc.paragraphs:
                full_text.append(para.text)
            return self.clean_text('\n'.join(full_text))
        except Exception as e:
            print(f"提取Word文本时出错: {str(e)}")
            return None

    def clean_text(self, text):
        """清理文本，去除多余空白和特殊字符"""
        # 去除多余的空行
        text = re.sub(r'\n\s*\n', '\n\n', text.strip())
        # 替换多个空格为一个空格
        text = re.sub(r' +', ' ', text)
        return text

    def extract_text(self, file_path):
        """根据文件类型提取文本"""
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            return None

        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif file_ext in ['.docx']:  # 注意不支持旧版.doc格式
            return self.extract_text_from_word(file_path)
        else:
            print(f"不支持的文件格式: {file_ext}")
            return None

    def compare_texts(self, text1, text2, file1_name, file2_name):
        """比对两个文本并生成增强版HTML格式的比对结果"""
        # 将文本按行分割
        text1_lines = text1.splitlines()
        text2_lines = text2.splitlines()

        # 使用SequenceMatcher进行更精确的比对
        matcher = difflib.SequenceMatcher(None, text1_lines, text2_lines)

        # 构建两边的内容，标记差异
        original_content = []
        new_content = []

        # 统计修改类型数量
        additions_count = 0
        deletions_count = 0
        changes_count = 0

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            # 处理原始文件内容
            if tag in ['replace', 'delete', 'equal']:
                for line in text1_lines[i1:i2]:
                    if tag == 'replace':
                        original_content.append(f'<span class="diff-removed">{line}</span>')
                        changes_count += 1
                    elif tag == 'delete':
                        original_content.append(f'<span class="diff-removed">{line}</span>')
                        deletions_count += 1
                    else:  # equal
                        original_content.append(line)

            # 处理新文件内容
            if tag in ['replace', 'insert', 'equal']:
                for line in text2_lines[j1:j2]:
                    if tag == 'replace':
                        new_content.append(f'<span class="diff-added">{line}</span>')
                        # 已经在上面统计过change_count
                    elif tag == 'insert':
                        new_content.append(f'<span class="diff-added">{line}</span>')
                        additions_count += 1
                    else:  # equal
                        new_content.append(line)

        # 如果没有差异
        if len(original_content) == 0:
            original_content.append('<span class="no-change">无内容</span>')
        if len(new_content) == 0:
            new_content.append('<span class="no-change">无内容</span>')

        # 加入换行符
        original_html = '<br>'.join(original_content)
        new_html = '<br>'.join(new_content)

        # 获取当前时间戳
        timestamp = datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")

        # 填充HTML模板
        html_result = self.html_template.format(
            timestamp=timestamp,
            file1_name=file1_name,
            file2_name=file2_name,
            original_content=original_html,
            new_content=new_html,
            additions_count=additions_count,
            deletions_count=deletions_count,
            changes_count=changes_count
        )

        return html_result

    def save_comparison_result(self, html_content, output_dir="comparison_results"):
        """保存比对结果到HTML文件"""
        # 创建输出目录（如果不存在）
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 生成带时间戳的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"contract_comparison_{timestamp}.html")

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return output_file

    def compare_files(self, file1_path, file2_path):
        """比对两个合同文件（PDF或Word）"""
        # 提取两个文件的文本
        text1 = self.extract_text(file1_path)
        text2 = self.extract_text(file2_path)

        if not text1 or not text2:
            print("无法提取文件文本，比对失败")
            return None

        # 获取文件名（不含路径）
        file1_name = os.path.basename(file1_path)
        file2_name = os.path.basename(file2_path)

        # 比对文本
        html_result = self.compare_texts(text1, text2, file1_name, file2_name)

        # 保存结果
        output_file = self.save_comparison_result(html_result)

        print(f"比对完成，结果已保存至: {output_file}")
        return output_file


def main():
    """主函数，演示增强版合同比对功能"""
    print("===== 增强版合同比对工具 =====")

    # 示例文件路径 - 实际使用时请修改为你的文件路径
    old_contract = "old_contract.docx"  # 旧合同
    new_contract = "new_contract.docx"  # 新合同

    # 创建比对器实例
    comparator = EnhancedContractComparator()

    # 执行比对
    result_file = comparator.compare_files(old_contract, new_contract)

    if result_file:
        print(f"请用浏览器打开 {result_file} 查看比对结果")
        print("提示:")
        print("  - 红色删除线: 旧文件有而新文件没有的内容")
        print("  - 绿色背景: 新文件有而旧文件没有的内容")
        print("  - 黄色背景: 修改过的内容")


if __name__ == "__main__":
    main()
