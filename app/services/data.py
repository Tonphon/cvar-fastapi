from typing import Dict, List, Optional
import pandas as pd
import requests

def stooq_url(symbol: str) -> str:
    return f"https://stooq.com/q/d/l/?s={symbol}&i=d"

def fetch_close(symbol: str, session: requests.Session) -> pd.Series:
    r = session.get(stooq_url(symbol), timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    from io import StringIO
    df = pd.read_csv(StringIO(r.text))
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    close = pd.to_numeric(df["close"], errors="coerce")
    s = pd.Series(close.to_numpy(dtype=float), index=df["date"]).dropna()
    if s.empty:
        raise ValueError(f"{symbol}: empty series")
    return s

def build_returns(assets: List[str], lookback_days: int, end_date: Optional[pd.Timestamp]=None) -> pd.DataFrame:
    sess = requests.Session()
    closes: Dict[str, pd.Series] = {}
    failures: Dict[str, str] = {}

    for sym in assets:
        try:
            closes[sym] = fetch_close(sym, sess)
        except Exception as e:
            failures[sym] = str(e)

    if failures:
        lines = "\n".join([f"- {k}: {v}" for k, v in failures.items()])
        raise RuntimeError("Some symbols failed to download:\n" + lines)

    df = pd.concat(closes, axis=1, join="inner")
    df.columns = assets
    df = df.sort_index()

    if end_date is not None:
        end_date = pd.to_datetime(end_date).normalize()
        df = df.loc[df.index <= end_date]
        if df.empty:
            raise ValueError(f"No price data on or before end_date={end_date.date()} for chosen assets.")

    df = df.tail(lookback_days + 1)
    if len(df) < 80:
        raise ValueError("Not enough overlapping history across all assets for selected end_date/lookback_days.")

    rets = df.pct_change().dropna(how="any")
    return rets.tail(lookback_days)
