import pandas as pd
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from typing import Tuple

@dataclass
class PreprocessResult:
    X: pd.DataFrame
    y: pd.Series
    scaler: StandardScaler
    feature_names: list

def preprocess_heart(df: pd.DataFrame) -> PreprocessResult:
    # Expect 'condition' as label
    y = df['condition']
    X = df.drop(columns=['condition'])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return PreprocessResult(
        X=pd.DataFrame(X_scaled, columns=X.columns),
        y=y,
        scaler=scaler,
        feature_names=list(X.columns)
    )

def preprocess_diabetes(df: pd.DataFrame) -> PreprocessResult:
    # Expect 'Outcome' as label
    y = df['Outcome']
    X = df.drop(columns=['Outcome'])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return PreprocessResult(
        X=pd.DataFrame(X_scaled, columns=X.columns),
        y=y,
        scaler=scaler,
        feature_names=list(X.columns)
    )
