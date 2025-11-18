# Disease Predictor (Heart-first)

A Flask web app that trains and compares multiple ML models for disease prediction and provides a clean UI for making predictions.

## Features
- Multi-model evaluation (LR, DT, RF, SVM, KNN, GB)
- Cross-validation metrics
- Heart disease form and prediction
- Bootstrap UI + charts

## Quickstart
1. Create venv and install `requirements.txt`
2. Put `data/heart.csv` (must include `target` column) and optional `data/diabetes.csv` (with `Outcome`)
3. Run `python models/train.py`
4. Start app with `python app.py`

## Notes
- Feature order in the heart form must match training columns.
- You can add more diseases by:
  - Adding a `preprocess_<disease>` in `preprocessors.py`
  - Duplicating the training call in `train.py`
  - Creating a new route and form template.
