import os
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash
from joblib import load

app = Flask(__name__)
app.secret_key = "change_this_for_prod"

BASE_DIR = os.path.dirname(__file__)
ARTIFACTS = os.path.join(BASE_DIR, 'models', 'artifacts')

def load_metrics():
    path = os.path.join(ARTIFACTS, 'metrics.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}

def load_model_scaler(task):
    model_path = os.path.join(ARTIFACTS, f"{task}_model.pkl")
    scaler_path = os.path.join(ARTIFACTS, f"{task}_scaler.pkl")
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
    return load(model_path), load(scaler_path)

@app.route("/")
def index():
    metrics = load_metrics()
    return render_template("index.html", metrics=metrics)

@app.route("/compare")
def compare():
    metrics = load_metrics()
    return render_template("compare.html", metrics=metrics)

@app.route("/predict/heart", methods=["GET", "POST"])
def predict_heart():
    # Define expected feature order (must match training dataset columns excluding 'target')
    feature_names = [
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
    ]

    if request.method == "POST":
        try:
            values = []
            for name in feature_names:
                v = request.form.get(name)
                if v is None or v == "":
                    raise ValueError(f"Missing value for {name}")
                values.append(float(v))
            X = np.array(values).reshape(1, -1)

            model, scaler = load_model_scaler('heart')
            if model is None:
                flash("Model not trained yet. Please run training.", "danger")
                return redirect(url_for('index'))

            X_scaled = scaler.transform(X)
            proba = None
            try:
                proba = float(model.predict_proba(X_scaled)[0, 1])
            except Exception:
                # fallback to decision function or predicted label
                proba = None
            pred = int(model.predict(X_scaled)[0])

            return render_template(
                "result.html",
                disease_name="Heart Disease",
                prediction=pred,
                probability=proba,
                interpretation="High risk" if pred == 1 else "Low risk"
            )
        except Exception as e:
            flash(f"Error: {e}", "danger")
            return redirect(url_for('predict_heart'))

    return render_template("predict_heart.html")

if __name__ == "__main__":
    app.run(debug=True)
