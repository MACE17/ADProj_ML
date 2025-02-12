from flask import Flask, request, jsonify
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
