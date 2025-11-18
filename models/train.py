import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import dump
from registry import get_model_candidates
from evaluators import evaluate_model, cross_validate_model
from preprocessors import preprocess_heart, preprocess_diabetes

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), 'artifacts')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def train_for_task(task_name: str, df: pd.DataFrame, preprocess_fn):
    print(f"=== Training for {task_name} ===")
    prep = preprocess_fn(df)
    X_train, X_test, y_train, y_test = train_test_split(
        prep.X, prep.y, test_size=0.2, random_state=42, stratify=prep.y
    )
    models = get_model_candidates()
    results = {}
    best_model_name = None
    best_f1 = -1.0

    for name, model in models.items():
        print(f"Training {name}...")
        metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
        cv_res = cross_validate_model(model, prep.X, prep.y, cv_splits=5)
        metrics.update(cv_res)
        results[name] = metrics
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_model_name = name
            best_model = model

    # Persist best model and scaler
    dump(best_model, os.path.join(ARTIFACT_DIR, f"{task_name}_model.pkl"))
    dump(prep.scaler, os.path.join(ARTIFACT_DIR, f"{task_name}_scaler.pkl"))
    print(f"Best model for {task_name}: {best_model_name} (F1={best_f1:.3f})")

    return results, best_model_name

def main():
    metrics_all = {}

    # Heart
    heart_path = os.path.join(DATA_DIR, 'heart.csv')
    heart_df = pd.read_csv(heart_path)
    heart_metrics, heart_best = train_for_task('heart', heart_df, preprocess_heart)
    metrics_all['heart'] = {"best": heart_best, "models": heart_metrics}

    # Diabetes (optional; will train if dataset exists)
    diabetes_path = os.path.join(DATA_DIR, 'diabetes.csv')
    if os.path.exists(diabetes_path):
        diabetes_df = pd.read_csv(diabetes_path)
        diabetes_metrics, diabetes_best = train_for_task('diabetes', diabetes_df, preprocess_diabetes)
        metrics_all['diabetes'] = {"best": diabetes_best, "models": diabetes_metrics}

    # Save metrics
    with open(os.path.join(ARTIFACT_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics_all, f, indent=2)

if __name__ == "__main__":
    main()
