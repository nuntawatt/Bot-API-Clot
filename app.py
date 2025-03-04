import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# โหลดโมเดลที่ฝึกไว้
final_model = joblib.load("trained_voting_classifierV3.pkl")
model = final_model["model"]
label_encoders = final_model["label_encoders"]

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

        print(f"🔍 LabelEncoder for size: {label_encoders['size'].classes_}")

        # ตรวจสอบค่าที่ได้รับ
        if "size" in data and data["size"] not in label_encoders["size"].classes_:
            print(f"Invalid size value received: {data['size']}")
            return jsonify({
                "error": f"Invalid size value: {data['size']}, must be one of {label_encoders['size'].classes_}"
            }), 400

        # แปลงค่าของ size ถ้ามีใน input
        encoded_size = label_encoders["size"].transform([data["size"]])[0] if "size" in data else None
        print(f"Encoded size: {encoded_size}")

        features = [
            float(data["weight"]),
            float(data["age"]),
            float(data["height"])
        ]
        print(f"Features: {features}")

        features_array = np.array([features]).reshape(1, -1)
        prediction = model.predict(features_array)
        size_predicted = label_encoders["size"].inverse_transform([prediction[0]])[0]

        print(f"Prediction: {size_predicted}")
        return jsonify({"prediction": size_predicted})

    except Exception as e:
        print(f"Server Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
