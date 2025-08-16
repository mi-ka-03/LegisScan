from flask import Flask, request, jsonify
from flask_cors import CORS  # 解决跨域问题
import torch
from predictor import LegalSpellingCorrector  # 导入你的模型预测类
import os

# 初始化Flask应用
app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 全局变量存储模型（避免重复加载）
model = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
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
    """处理文本纠错请求"""
    try:
        # 获取前端发送的文本
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "请提供文本参数"}), 400
        
        text = data['text']
        if len(text) == 0:
            return jsonify({"error": "文本不能为空"}), 400
        
        # 调用模型进行纠错
        corrected_text, errors = model.correct(text)  # 假设模型有correct方法
        
        # 返回结果
        return jsonify({
            "original_text": text,
            "corrected_text": corrected_text,
            "errors": errors  # 错误列表，如["错误1->正确1", "错误2->正确2"]
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # 启动服务（debug=True仅用于开发，生产环境需关闭）
    app.run(host='0.0.0.0', port=5000, debug=True)
