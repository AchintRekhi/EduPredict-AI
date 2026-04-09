# app.py — Flask Backend for Student GPA & Persistence Prediction
# Loads all pre-trained models and exposes 3 prediction endpoints.

import os
import sys
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify

# Resolve paths relative to project root (one level up from webapp/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")

app = Flask(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Load all models & preprocessors at startup
# ──────────────────────────────────────────────────────────────────────────────

# 1. Persistence — Neural Network
from tensorflow.keras.models import load_model

persistence_nn = load_model(os.path.join(MODEL_DIR, "trained_model.h5"))
persistence_preprocessor = joblib.load(os.path.join(MODEL_DIR, "preprocessor.pkl"))

# 2. Persistence — Random Forest
persistence_rf = joblib.load(os.path.join(MODEL_DIR, "rf_model.pkl"))
persistence_rf_preprocessor = joblib.load(os.path.join(MODEL_DIR, "rf_preprocessor.pkl"))

# 3. GPA Regression — Neural Network
gpa_nn = load_model(os.path.join(MODEL_DIR, "gpa_regression_model.h5"))
gpa_scaler = joblib.load(os.path.join(MODEL_DIR, "gpa_scaler.pkl"))

# 4. GPA Regression — Random Forest
gpa_rf = joblib.load(os.path.join(MODEL_DIR, "rf_gpa_model.pkl"))

# 5. GPA Improvement — Neural Network
gpa_improvement_nn = load_model(os.path.join(MODEL_DIR, "gpa_improvement_model.h5"))
gpa_improvement_scaler = joblib.load(os.path.join(MODEL_DIR, "gpa_improvement_scaler.pkl"))


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main single-page application."""
    return render_template("index.html")


@app.route("/predict/persistence", methods=["POST"])
def predict_persistence():
    """Predict first-year persistence using both NN and RF models.

    Expected JSON payload keys:
        First_Term_Gpa, Second_Term_Gpa, First_Language, Funding,
        FastTrack, Coop, Residency, Gender, Previous_Education,
        Age_Group, High_School_Average_Mark, Math_Score, English_Grade
    """
    try:
        data = request.get_json()

        # Build input DataFrame (column order must match training)
        input_dict = {
            "First_Term_Gpa": float(data["First_Term_Gpa"]),
            "Second_Term_Gpa": float(data["Second_Term_Gpa"]),
            "First_Language": float(data["First_Language"]),
            "Funding": int(data["Funding"]),
            "School": 1,  # Placeholder — not used by model but required by preprocessor column order
            "FastTrack": int(data["FastTrack"]),
            "Coop": int(data["Coop"]),
            "Residency": int(data["Residency"]),
            "Gender": int(data["Gender"]),
            "Previous_Education": float(data["Previous_Education"]),
            "Age_Group": float(data["Age_Group"]),
            "High_School_Average_Mark": float(data["High_School_Average_Mark"]),
            "Math_Score": float(data["Math_Score"]),
            "English_Grade": float(data["English_Grade"]),
        }
        input_df = pd.DataFrame([input_dict])

        # ── Neural Network Prediction ──
        X_nn = persistence_preprocessor.transform(input_df)
        nn_prob = float(persistence_nn.predict(X_nn)[0][0])
        nn_prediction = int(nn_prob > 0.43)  # Tuned threshold

        # ── Random Forest Prediction ──
        X_rf = persistence_rf_preprocessor.transform(input_df)
        rf_prob = float(persistence_rf.predict_proba(X_rf)[0][1])
        rf_prediction = int(rf_prob > 0.43)

        return jsonify({
            "success": True,
            "nn": {
                "prediction": nn_prediction,
                "probability": round(nn_prob, 4),
                "label": "Will Persist" if nn_prediction == 1 else "At Risk"
            },
            "rf": {
                "prediction": rf_prediction,
                "probability": round(rf_prob, 4),
                "label": "Will Persist" if rf_prediction == 1 else "At Risk"
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/predict/gpa", methods=["POST"])
def predict_gpa():
    """Predict second-term GPA using both NN and RF models.

    Expected JSON payload keys:
        First_Term_Gpa, High_School_Average_Mark
    """
    try:
        data = request.get_json()

        first_gpa = float(data["First_Term_Gpa"])
        hs_mark = float(data["High_School_Average_Mark"])

        input_df = pd.DataFrame([{
            "First_Term_Gpa": first_gpa,
            "High_School_Average_Mark": hs_mark
        }])

        # ── Neural Network Prediction ──
        X_scaled = gpa_scaler.transform(input_df)
        nn_pred = float(gpa_nn.predict(X_scaled)[0][0])

        # ── Random Forest Prediction ──
        rf_pred = float(gpa_rf.predict(input_df.values)[0])

        return jsonify({
            "success": True,
            "nn": {"predicted_gpa": round(nn_pred, 2)},
            "rf": {"predicted_gpa": round(rf_pred, 2)}
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/predict/improvement", methods=["POST"])
def predict_improvement():
    """Predict GPA improvement (2nd term − 1st term) using NN model.

    Expected JSON payload keys:
        High_School_Average_Mark, First_Term_Gpa, Second_Term_Gpa
    """
    try:
        data = request.get_json()

        input_df = pd.DataFrame([{
            "High_School_Average_Mark": float(data["High_School_Average_Mark"]),
            "First_Term_Gpa": float(data["First_Term_Gpa"]),
            "Second_Term_Gpa": float(data["Second_Term_Gpa"]),
        }])

        X_scaled = gpa_improvement_scaler.transform(input_df)
        prediction = float(gpa_improvement_nn.predict(X_scaled)[0][0])

        return jsonify({
            "success": True,
            "predicted_improvement": round(prediction, 2)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)
