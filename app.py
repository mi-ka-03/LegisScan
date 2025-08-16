from flask import Flask, request, jsonify, send_from_directory  # 补充导入send_from_directory
from flask_cors import CORS
import torch
from predictor import LegalSpellingCorrector
import os

app = Flask(__name__)
CORS(app)

# 新增：根路径路由，返回前端页面
@app.route('/')
def serve_frontend():
    return send_from_directory('static', 'index.html')  # 从static文件夹返回index.html

# 以下是原有的模型加载和/correct接口代码（保持不变）
model = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    # ...（原代码）
    """加载训练好的模型"""
    global model
    # 替换为你的模型加载逻辑
    model_path = "models/best_model"  # 模型保存路径
    if not os.path.exists(model_path):
        raise Exception(f"模型文件不存在：{model_path}")
    
    # 初始化纠错器（根据你的实际类实现调整）
    model = LegalSpellingCorrector(
        model_path=model_path,
        device=device
    )
    print(f"模型加载成功，使用设备：{device}")

# 启动时加载模型
load_model()


@app.route('/correct', methods=['POST'])
def correct_text():
    try:
        data = request.get_json()
        # 先验证传参
        if not data or 'text' not in data:
            return jsonify({"error": "前端没传 text！"}), 400
        text = data['text']

        # 调用模型前，手动打印 text 看是否正常
        print(f"收到文本：{text}")

        # 模型预测环节
        corrected_text = model.correct_text(text)  # 调用正确的方法名
        # 手动构造 errors（如果需要返回错误差异，调用 compare_text）
        comparison = model.compare_text(text, corrected_text)
        errors = comparison['differences']  # 提取差异信息
        return jsonify({
            "original_text": text,
            "corrected_text": corrected_text,
            "errors": errors
        })

    # 关键：把异常详细信息抛给前端/终端
    except Exception as e:
        # 终端打印完整报错栈
        import traceback
        traceback.print_exc()
        # 给前端返回具体错误
        return jsonify({"error": f"后端炸了：{str(e)}"}), 500

if __name__ == '__main__':
    # 启动服务（debug=True仅用于开发，生产环境需关闭）
    app.run(host='0.0.0.0', port=5000, debug=False)
