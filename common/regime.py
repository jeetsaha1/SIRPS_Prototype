import numpy as np
import pandas as pd


def classify(close: pd.Series, bear_dd: float = -0.15, confirm_days: int = 10) -> pd.DataFrame:
    """Causal regime classifier using NIFTY 50 trailing close.
    
    Rules (from Blueprint v2.0):
    - Bear: Drawdown <= bear_dd (e.g. -15%) OR (Close < MA200 AND MA50 < MA200 AND Drawdown <= -0.10)
    - Bull: Close > MA200 AND MA50 > MA200 AND Drawdown > -0.10
    - Neutral: Neither rule met
    
    10-day confirmation: switches only after new raw regime has held 10 consecutive trading days.
    Never back-dated. Initial state Neutral.
    """
    n = len(close)
    ma50 = close.rolling(50, min_periods=1).mean()
    ma200 = close.rolling(200, min_periods=1).mean()
    running_max = close.rolling(252, min_periods=1).max()
    drawdown_252 = (close - running_max) / running_max

    raw_regimes = []
    for i in range(n):
        c = close.iloc[i]
        m50 = ma50.iloc[i]
        m200 = ma200.iloc[i]
        dd = drawdown_252.iloc[i]

        is_bear = (dd <= bear_dd) or (c < m200 and m50 < m200 and dd <= -0.10)
        if is_bear:
            raw_regimes.append("Bear")
        else:
            is_bull = (c > m200 and m50 > m200 and dd > -0.10)
            if is_bull:
                raw_regimes.append("Bull")
            else:
                raw_regimes.append("Neutral")

    confirmed_regimes = []
    current_confirmed = "Neutral"
    streak_regime = None
    streak_count = 0

    for i in range(n):
        raw = raw_regimes[i]
        if raw == streak_regime:
            streak_count += 1
        else:
            streak_regime = raw
            streak_count = 1

        if streak_count >= confirm_days and streak_regime != current_confirmed:
            current_confirmed = streak_regime

        confirmed_regimes.append(current_confirmed)

    return pd.DataFrame({
        "close": close.values,
        "ma50": ma50.values,
        "ma200": ma200.values,
        "drawdown_252": drawdown_252.values,
        "raw_regime": raw_regimes,
        "confirmed_regime": confirmed_regimes,
    })
