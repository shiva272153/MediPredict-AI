# Disease Predictor

A Flask-based web application for predicting the likelihood of **Heart Disease** and **Diabetes** using Machine Learning models. This project demonstrates an end-to-end ML pipeline, from model training to a user-friendly web interface.

## Features

- **Multi-Disease Prediction**: 
  - **Heart Disease**: Predicts risk based on age, sex, chest pain type, blood pressure, etc.
  - **Diabetes**: Predicts likelihood based on glucose, BMI, insulin levels, age, etc.
- **Machine Learning**: Uses `scikit-learn` to train and evaluate models (Logistic Regression by default).
- **Web Interface**: Clean and responsive UI built with **Flask**, **Bootstrap**, and **Jinja2**.
- **Prediction Logging**: Automatically saves prediction results to CSV files (`data/logged_predictions.csv` and `data/logged_predictions_diabetes.csv`) for record-keeping.
- **Privacy & Terms**: Dedicated pages for Privacy Policy and Terms of Service.

## Technologies Used

- **Backend**: Python, Flask
- **ML/Data**: Scikit-learn, Pandas, NumPy, Joblib
- **Frontend**: HTML5, CSS3, Bootstrap 5, JavaScript (Chart.js)
- **Containerization**: (Optional - if applicable, otherwise omit)

## Project Structure

```
disease-predictor/
├── app.py                 # Main Flask application entry point
├── models/                # ML model training scripts and artifacts
│   ├── train.py           # Script to train models
│   ├── preprocessors.py   # Data preprocessing logic
│   └── artifacts/         # Saved .pkl models and scalers
├── templates/             # HTML templates (Jinja2)
├── static/                # Static assets (CSS, JS, images)
├── data/                  # Dataset storage and prediction logs
└── requirements.txt       # Python dependencies
```

## Quickstart

### Prerequisites
- Python 3.8+ installed
- Git (optional, for cloning)

### Installation

1. **Clone the repository** (or download source):
   ```bash
   git clone <repository_url>
   cd disease-predictor
   ```

2. **Create a Virtual Environment**:
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate

   # Mac/Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare Data**:
   - Ensure your training datasets (`heart.csv` and `diabetes.csv`) are placed in the `data/` directory.
   - *Note: `heart.csv` must have a `target` column, and `diabetes.csv` must have an `Outcome` column.*

5. **Train Models**:
   Run the training script to generate the model artifacts (`.pkl` files).
   ```bash
   python models/train.py
   ```

6. **Run the Application**:
   ```bash
   python app.py
   ```
   The app will start at `http://127.0.0.1:5000/`.

## Usage

1. Open your browser and navigate to `http://127.0.0.1:5000/`.
2. Select **Heart Disease** or **Diabetes** from the dashboard.
3. Fill in the patient's medical details in the form.
4. Click **Predict** to see the result (Risk/No Risk) and the probability score.
5. All predictions are logged to the `data/` directory for future reference.

## License

This project is open-source and available under the [MIT License](LICENSE).
