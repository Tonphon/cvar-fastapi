import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

def normalize_weights(vals: np.ndarray) -> np.ndarray:
    vals = np.array(vals, dtype=float)
    if not np.isfinite(vals).all():
        raise ValueError("Portfolio contains non-numeric values.")
    if np.allclose(vals, 0):
        raise ValueError("All portfolio values are zero.")
    s = float(np.sum(vals))
    if abs(s - 1.0) < 1e-3:
        w = vals
    else:
        w = np.maximum(vals, 0.0)
        w = w / float(np.sum(w))
    return w

def load_portfolio_csv(path: str) -> Tuple[Optional[str], List[str], np.ndarray]:
    df = pd.read_csv(path)
    cols_lower = [c.lower() for c in df.columns]

    # long: asset, weight, (optional) date
    if "asset" in cols_lower and "weight" in cols_lower:
        df.columns = cols_lower
        date_str = None
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])
            latest = df["date"].max()
            df = df[df["date"] == latest].copy()
            date_str = str(latest.date())
        assets = df["asset"].astype(str).tolist()
        w_prev = normalize_weights(df["weight"].astype(float).to_numpy())
        return date_str, assets, w_prev

    # wide 1-row: date, TICKER1, TICKER2...
    if df.shape[0] < 1 or df.shape[1] < 2:
        raise ValueError("portfolio.csv must be 1 row: date + at least 1 asset column.")

    first = df.columns[0]
    date_str = None
    try:
        date_str = str(pd.to_datetime(df.loc[0, first]).date())
    except Exception:
        pass

    assets = list(df.columns[1:])
    vals = pd.to_numeric(df.loc[0, assets], errors="coerce").to_numpy(dtype=float)
    if np.any(~np.isfinite(vals)):
        bad = [a for a, v in zip(assets, vals) if not np.isfinite(v)]
        raise ValueError(f"Non-numeric values for assets: {bad}")

    return date_str, assets, normalize_weights(vals)
