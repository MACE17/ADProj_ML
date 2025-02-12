from flask import Flask, request, jsonify
from models.bert_pca import get_bert_embedding
from models.xgb_model import predict
from utils.data_processing import clean_text

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
    app.run(host="0.0.0.0", port=5000, debug=True)
