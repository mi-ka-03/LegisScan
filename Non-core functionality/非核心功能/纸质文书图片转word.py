from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from paddleocr import PaddleOCR
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
from docx import Document
import os
import uuid
import logging
from datetime import datetime
import shutil
import tempfile
from pathlib import Path

# ======================== 配置与初始化 ========================
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ocr_service.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 初始化Flask应用
app = Flask(__name__)
CORS(app)


# 配置参数
class Config:
    # 临时文件存储配置
    TEMP_DOCS_DIR = Path("temp_docs")
    TEMP_FILE_RETENTION_HOURS = 24  # 临时文件保留时间
    # OCR配置
    OCR_LANGUAGE = 'ch'
    USE_ANGLE_CORRECTION = True
    LINE_HEIGHT_THRESHOLD = 20  # 行高阈值，用于文本块分组
    # 服务配置
    HOST = '0.0.0.0'
    PORT = 5001
    DEBUG = False  # 生产环境建议设为False


app.config.from_object(Config)

# 确保临时目录存在
app.config['TEMP_DOCS_DIR'].mkdir(parents=True, exist_ok=True)

# 初始化PaddleOCR
try:
    logger.info("正在初始化PaddleOCR模型...")
    ocr = PaddleOCR(
        use_angle_cls=app.config['USE_ANGLE_CORRECTION'],
        lang=app.config['OCR_LANGUAGE'],
        show_log=False  # 关闭PaddleOCR内部日志
    )
    logger.info("OCR模型初始化成功")
except Exception as e:
    logger.error(f"OCR模型初始化失败: {str(e)}", exc_info=True)
    raise


# ======================== 工具函数 ========================
def clean_temp_files():
    """清理过期的临时文件"""
    try:
        now = datetime.now().timestamp()
        retention_seconds = app.config['TEMP_FILE_RETENTION_HOURS'] * 3600

        for file_path in app.config['TEMP_DOCS_DIR'].glob("*"):
            if file_path.is_file():
                file_mtime = os.path.getmtime(file_path)
                if now - file_mtime > retention_seconds:
                    os.remove(file_path)
                    logger.info(f"已清理过期临时文件: {file_path}")
    except Exception as e:
        logger.warning(f"清理临时文件时发生错误: {str(e)}")


def validate_image_file(file):
    """验证上传的文件是否为有效的图片"""
    allowed_extensions = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}
    if '.' not in file.filename:
        return False, "文件名格式不正确"

    ext = file.filename.rsplit('.', 1)[1].lower()
    if ext not in allowed_extensions:
        return False, f"不支持的文件格式，支持的格式: {', '.join(allowed_extensions)}"

    # 简单验证文件头
    try:
        # 读取文件开头几个字节验证是否为图片
        file.seek(0)
        header = file.read(10)
        file.seek(0)  # 重置文件指针
        return True, "文件验证通过"
    except Exception as e:
        return False, f"文件读取错误: {str(e)}"


# ======================== 核心功能函数 ========================
def create_word_document_from_ocr(ocr_result):
    """
    根据OCR结果创建Word文档
    :param ocr_result: PaddleOCR返回的识别结果
    :return: 生成的Word文档的临时文件路径
    """
    try:
        doc = Document()

        # 添加标题和说明
        doc.add_heading('OCR文本识别结果', 0)
        doc.add_paragraph('注意：此文档由AI自动识别生成，可能存在误差，请仔细核对。')
        doc.add_paragraph('识别时间：' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        doc.add_paragraph('--- 识别内容开始 ---')

        if not ocr_result or not ocr_result[0]:
            doc.add_paragraph("未识别到任何文字内容。")
            temp_file_name = f"empty_{uuid.uuid4()}.docx"
            temp_path = app.config['TEMP_DOCS_DIR'] / temp_file_name
            doc.save(temp_path)
            return str(temp_path)

        # 提取文本块和其位置信息
        text_blocks = []
        for line in ocr_result[0]:
            poly, (text, score) = line
            x1, y1 = poly[0]
            x2, y2 = poly[2]
            text_blocks.append((text, int(x1), int(y1), int(x2), int(y2), score))

        # 布局还原核心逻辑
        # 1. 根据y坐标对文本块进行分组，模拟行
        lines = []
        current_line = []

        # 按y坐标排序
        text_blocks.sort(key=lambda b: b[2])

        for block in text_blocks:
            if not current_line:
                current_line.append(block)
            else:
                # 如果当前块的y坐标与行首块的y坐标差小于阈值，则认为在同一行
                if block[2] - current_line[0][2] < app.config['LINE_HEIGHT_THRESHOLD']:
                    current_line.append(block)
                else:
                    lines.append(current_line)
                    current_line = [block]
        if current_line:
            lines.append(current_line)

        # 2. 逐行处理，写入Word文档
        for line in lines:
            # 同一行内，按x坐标从左到右排序
            line.sort(key=lambda b: b[1])

            # 将该行所有文本块合并成一个段落
            paragraph_text = " ".join([block[0] for block in line])

            # 添加段落
            doc.add_paragraph(paragraph_text)

        # 保存到临时文件
        temp_file_name = f"ocr_result_{uuid.uuid4()}.docx"
        temp_path = app.config['TEMP_DOCS_DIR'] / temp_file_name
        doc.save(temp_path)

        logger.info(f"已生成Word文档: {temp_path}")
        return str(temp_path)

    except Exception as e:
        logger.error(f"创建Word文档失败: {str(e)}", exc_info=True)
        raise


def process_contract_image(image_bytes):
    """处理合同图片，返回Word文档路径"""
    try:
        # 将字节流转换为图片
        image = Image.open(BytesIO(image_bytes))
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # 执行OCR识别
        logger.info("开始OCR识别...")
        ocr_result = ocr.ocr(img, cls=app.config['USE_ANGLE_CORRECTION'])
        logger.info("OCR识别完成")

        # 根据OCR结果创建Word文档
        word_file_path = create_word_document_from_ocr(ocr_result)

        return word_file_path, None

    except Exception as e:
        error_msg = f"处理图片失败：{str(e)}"
        logger.error(error_msg, exc_info=True)
        return None, error_msg


# ======================== API接口 ========================
@app.route('/convert_contract', methods=['POST'])
def convert_contract_api():
    """转换合同图片为Word文档的API接口"""
    try:
        # 定期清理临时文件
        clean_temp_files()

        if 'image' not in request.files:
            return jsonify({"success": False, "error": "请上传图片文件"}), 400

        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({"success": False, "error": "未选择文件"}), 400

        # 验证图片文件
        is_valid, msg = validate_image_file(image_file)
        if not is_valid:
            return jsonify({"success": False, "error": msg}), 400

        # 处理图片
        image_bytes = image_file.read()
        word_path, error = process_contract_image(image_bytes)

        if error:
            return jsonify({"success": False, "error": error}), 400

        # 发送生成的Word文件
        return send_file(
            word_path,
            as_attachment=True,
            download_name=f'ocr_result_{datetime.now().strftime("%Y%m%d_%H%M%S")}.docx',
            mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        )

    except Exception as e:
        error_msg = f"服务器错误：{str(e)}"
        logger.error(error_msg, exc_info=True)
        return jsonify({"success": False, "error": error_msg}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """服务健康检查接口"""
    return jsonify({
        "status": "healthy",
        "service": "ocr-converter",
        "timestamp": datetime.now().isoformat()
    })


# ======================== 启动服务 ========================
def main():
    logger.info("启动OCR转换服务...")
    try:
        app.run(
            host=app.config['HOST'],
            port=app.config['PORT'],
            debug=app.config['DEBUG']
        )
    except Exception as e:
        logger.critical(f"服务启动失败: {str(e)}", exc_info=True)


if __name__ == '__main__':
    main()