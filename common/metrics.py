import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def compute_msfe(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    return float(np.mean((y_true - y_hat) ** 2))


def compute_rmse(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    return float(np.sqrt(compute_msfe(y_true, y_hat)))


def compute_mae(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_hat)))


def compute_directional_accuracy(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    return float(np.mean(np.sign(y_hat) == np.sign(y_true)))


def compute_rank_ic(df: pd.DataFrame, pred_col: str = "y_hat_excess", true_col: str = "y_true", date_col: str = "date") -> tuple:
    date_ics = []
    for dt, grp in df.groupby(date_col):
        if len(grp) >= 3 and grp[pred_col].nunique() > 1:
            ic, _ = spearmanr(grp[pred_col], grp[true_col])
            if pd.notna(ic):
                date_ics.append(ic)
    if not date_ics:
        return 0.0, 0.0
    mean_ic = float(np.mean(date_ics))
    std_ic = float(np.std(date_ics))
    ir = (mean_ic / std_ic) if std_ic > 0 else 0.0
    return mean_ic, ir
