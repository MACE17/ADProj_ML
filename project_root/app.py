from flask import Flask, request, jsonify, Response
from models.bert_pca import get_bert_embedding
from models.xgb_model import predict
from utils.data_processing import clean_text
import os
debug_mode = os.getenv("FLASK_DEBUG", "False").lower() == "true"
host = os.getenv("FLASK_HOST", "127.0.0.1")  # 默认只监听本机
port = int(os.getenv("FLASK_PORT", "5000"))

app = Flask(__name__)

@app.route("/")
def home():
    return "Hello, this is my API!", 200

@app.after_request
def add_security_headers(response: Response):
    response.headers["X-Frame-Options"] = "DENY"  # 防止 Clickjacking
    response.headers["X-Content-Type-Options"] = "nosniff"  # 防止 MIME 类型混淆
    response.headers["Server"] = "Hidden"  # 隐藏服务器信息
    response.headers["Content-Security-Policy"] = "default-src 'self'"  # 限制 CSP 访问
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"  # 关闭敏感权限
    response.headers["Cross-Origin-Resource-Policy"] = "same-origin"  # 防止 Spectre 攻击
    return response

@app.route("/predict", methods=["POST"])
def predict_route():
    try:
        data = request.get_json()
        review_text = data.get("review", "").strip()
        if not review_text:
            return jsonify({"error": "Review text is required"}), 400

        # 清理文本
        cleaned_text = clean_text(review_text)

        # 获取 BERT 特征向量
        vector = get_bert_embedding(cleaned_text)

        # 进行预测
        prediction = 1-predict(vector)

        # **直接反转 0 和 1**
        corrected_prediction = 1 if prediction == 0 else 0

        return jsonify({"results": corrected_prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    host = os.getenv("FLASK_HOST", "127.0.0.1")  # 默认只监听本机，生产环境可配置
    
    app.run(host=host, port=5000, debug=debug_mode)
