from flask import Flask, request, jsonify, Response
from models.bert_pca import get_bert_embedding
from models.xgb_model import predict
from utils.data_processing import clean_text
import os

# ✅ 读取环境变量，防止 Debug 模式意外开启
debug_mode = os.getenv("FLASK_DEBUG", "False").lower() == "true"
host = os.getenv("FLASK_HOST", "0.0.0.0")  # ✅ 生产环境监听所有地址
port = int(os.getenv("FLASK_PORT", "5000"))

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"message": "ML API is running!"}), 200

# ✅ 修复 ZAP DAST `WARN-NEW` 问题，增强安全性
@app.after_request
def add_security_headers(response: Response):
    response.headers["X-Frame-Options"] = "DENY"  # 防止 Clickjacking
    response.headers["X-Content-Type-Options"] = "nosniff"  # 防止 MIME 类型混淆
    response.headers["Referrer-Policy"] = "no-referrer"  # 防止敏感信息泄露
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"  # 强制 HTTPS
    response.headers["X-XSS-Protection"] = "1; mode=block"  # 启用 XSS 保护
    response.headers["Content-Security-Policy"] = "default-src 'self'"  # 限制 CSP 访问
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"  # 关闭敏感权限
    response.headers["Cross-Origin-Resource-Policy"] = "same-origin"  # 防止 Spectre 攻击
    response.headers.pop("Server", None)  # ✅ 移除 Server 头信息，防止服务器信息泄露
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
        prediction = 1 - predict(vector)

        # **直接反转 0 和 1**
        corrected_prediction = 1 if prediction == 0 else 0

        return jsonify({"results": corrected_prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host=host, port=port, debug=debug_mode)  # ✅ 确保 debug=False
