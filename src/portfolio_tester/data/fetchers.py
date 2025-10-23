import pandas as pd
import numpy as np
from .cache import key_path

def _cache_read(path):
    if path.exists():
        return pd.read_parquet(path)
    return None

def _cache_write(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)

def fetch_prices_monthly(tickers, start=None, end=None):
    """Fetch daily Adj Close from yfinance, resample to month-end.
    Returns a (monthly) price DataFrame with columns = tickers.
    """
    import yfinance as yf
    key = "|".join(tickers) + f"|{start}|{end}"
    path = key_path("prices_yf", key)
    df = _cache_read(path)
    if df is None:
        data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False, interval="1d", group_by="column")
        # yfinance returns wide format; adjust to a simple DataFrame of Adj Close
        if isinstance(tickers, (list, tuple)) and len(tickers) > 1:
            px = data["Adj Close"].copy()
        else:
            # Single ticker returns a Series; normalize to DataFrame
            px = data["Adj Close"].to_frame(tickers[0])
        df = px.resample("M").last().dropna(how="all")
        _cache_write(df, path)
    return df

def fetch_fred_series(series_id, start=None, end=None):
    """Fetch a FRED series monthly. Tries fredapi (if FRED_API_KEY) then falls back to pandas-datareader."""
    key = f"{series_id}|{start}|{end}"
    path = key_path("fred", key)
    df = _cache_read(path)
    if df is None:
        try:
            from fredapi import Fred
            import os
            fred = Fred(api_key=os.getenv("FRED_API_KEY", ""))
            s = fred.get_series(series_id, observation_start=start, observation_end=end)
            df = s.to_frame(series_id)
        except Exception:
            from pandas_datareader import data as pdr
            df = pdr.DataReader(series_id, "fred", start=start, end=end)
        df.index = pd.to_datetime(df.index)
        df = df.resample("M").last()
        _cache_write(df, path)
    return df

def prep_returns_and_macro(prices_m, start=None, end=None):
    """
    - Trim to dates
    - Force common overlap
    - Compute monthly simple returns
    - Build monthly inflation (CPIAUCSL) and monthly risk-free (TB3MS converted)
    Returns: (returns_df, inflation_series, riskfree_series)
    """
    prices_m = prices_m.sort_index()
    if start: prices_m = prices_m[prices_m.index >= pd.to_datetime(start)]
    if end:   prices_m = prices_m[prices_m.index <= pd.to_datetime(end)]
    prices_m = prices_m.dropna(axis=1, how="all").dropna(how="any")
    rets_m = prices_m.pct_change().dropna()

    cpi = fetch_fred_series("CPIAUCSL", start=rets_m.index.min(), end=rets_m.index.max())
    tb3 = fetch_fred_series("TB3MS", start=rets_m.index.min(), end=rets_m.index.max())

    cpi = cpi.reindex(rets_m.index).ffill()
    infl_m = cpi["CPIAUCSL"].pct_change().reindex(rets_m.index).fillna(0.0)

    rf_m = ((1.0 + (tb3.reindex(rets_m.index)["TB3MS"] / 100.0)) ** (1/12.0) - 1.0)
    rf_m = rf_m.fillna(method="ffill").fillna(0.0)

    return rets_m, infl_m.rename("inflation_m"), rf_m.rename("rf_m")
