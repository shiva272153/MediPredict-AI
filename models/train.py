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
    
    # We now strictly use the single model defined in registry
    models = get_model_candidates()
    # Assuming only one model is left, or we pick the first one
    name, model = list(models.items())[0]
    
    print(f"Training {name}...")
    # Train on full provided dataset (since we are finalizing the model)
    # But for consistency with previous logic, we can still split or just fit on all.
    # To keep it robust, let's fit on the whole dataset for the final artifact
    model.fit(prep.X, prep.y)
    
    # Evaluate (Optional, just to print score)
    # X_train, X_test, y_train, y_test = train_test_split(prep.X, prep.y, test_size=0.2, random_state=42)
    # model.fit(X_train, y_train)
    # score = model.score(X_test, y_test)
    # print(f"Validation Accuracy: {score:.3f}")

    # Persist model and scaler
    dump(model, os.path.join(ARTIFACT_DIR, f"{task_name}_model.pkl"))
    dump(prep.scaler, os.path.join(ARTIFACT_DIR, f"{task_name}_scaler.pkl"))
    print(f"Saved {name} for {task_name}")

    return {}, name

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
