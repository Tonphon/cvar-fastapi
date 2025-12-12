import pandas as pd

def worst_days_table(returns_csv_path: str, split: str = "test", k: int = 10):
    df = pd.read_csv(returns_csv_path)
    df = df[df["split"] == split].copy()
    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values("ret_prev").head(k)

    out = df[["date", "ret_prev", "ret_opt"]].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out.to_dict(orient="records")
