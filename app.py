import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# โหลดโมเดลที่ฝึกไว้
model = joblib.load("trained_voting_classifierV3.pkl")  # โหลดโมเดลตรงๆ

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return jsonify({"message": "Flask API for clothing size prediction is running."})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        print(f"🔹 Received Data: {data}")

        # ตรวจสอบค่าที่ได้รับ
        features = [
            float(data["weight"]),
            float(data["age"]),
            float(data["height"])
        ]
        print(f"Features: {features}")

        features_array = np.array([features]).reshape(1, -1)
        prediction = model.predict(features_array)[0]  # ดึงค่าผลลัพธ์ออกมาโดยตรง

        print(f"Prediction: {prediction}")
        return jsonify({"prediction": prediction})

    except Exception as e:
        print(f"Server Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
