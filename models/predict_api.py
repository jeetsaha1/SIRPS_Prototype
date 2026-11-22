from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def predict_single(features_dict: dict, experiment: str = "B", model_name: str = "Ridge") -> float:
    """Predicts excess return given an input feature dictionary."""
    model_file = ROOT / "models" / "saved" / f"{experiment}_{model_name.replace(' ', '_')}.joblib"
    if model_file.exists():
        bundle = joblib.load(model_file)
        model = bundle["model"]
        feature_cols = bundle["features"]
        x = pd.DataFrame([[features_dict.get(c, 0.0) for c in feature_cols]], columns=feature_cols)
        return float(model.predict(x)[0])
    return 0.015
