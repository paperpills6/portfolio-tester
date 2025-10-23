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
    """
    Fetch daily (auto-adjusted) prices from yfinance and resample to monthly (M) last.
    Returns a monthly price DataFrame with columns = tickers.
    """
    import pandas as pd
    import yfinance as yf
    key = "|".join(tickers) + f"|{start}|{end}"
    path = key_path("prices_yf", key)

    df = _cache_read(path)
    if df is not None:
        return df

    # auto_adjust=True already includes dividends/splits -> use 'Close'
    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        interval="1d",
        group_by="column"
    )

    # Normalize to a simple DataFrame with 1 column per ticker of closing prices
    def _extract_close_frame(data, tickers):
        # Multi-ticker usually returns a MultiIndex DataFrame
        if isinstance(data, pd.DataFrame) and isinstance(data.columns, pd.MultiIndex):
            lvl0 = data.columns.get_level_values(0)
            # Try field-first layout: ('Close'/'Adj Close', ticker)
            for fld in ("Close", "Adj Close"):
                if fld in set(lvl0):
                    return data[fld].copy()
            # Try ticker-first layout: (ticker, 'Close'/'Adj Close')
            for fld in ("Close", "Adj Close"):
                try:
                    return data.xs(fld, level=1, axis=1).copy()
                except KeyError:
                    pass

        # Single-ticker may return a regular DataFrame
        if isinstance(data, pd.DataFrame) and not isinstance(data.columns, pd.MultiIndex):
            for fld in ("Close", "Adj Close"):
                if fld in data.columns:
                    # ensure the column is named as the ticker
                    return data[[fld]].rename(columns={fld: tickers[0]}).copy()

        # Single-ticker may return a Series
        if hasattr(data, "name") and data.name in ("Close", "Adj Close"):
            return data.to_frame(tickers[0]).copy()

        raise ValueError("Could not find 'Close' or 'Adj Close' in yfinance output.")

    px_daily = _extract_close_frame(data, tickers)

    # Make sure columns are exactly in the input order (for consistency)
    # Missing tickers (if any) will be dropped automatically by reindexing columns present
    present = [t for t in tickers if t in px_daily.columns]
    px_daily = px_daily[present]
 
    # Resample to month-end and cache
    df = px_daily.resample("ME").last().dropna(how="all")
    _cache_write(df, path)
    return df

def fetch_fred_series(series_id, start=None, end=None):
    """
    Fetch a FRED series via CSV endpoints (no API key), bypassing system proxies.
    Tries 'downloaddata' then 'fredgraph' CSV. Caches monthly, month-end data.
    """
    import io
    import pandas as pd
    import requests, certifi

    key = f"{series_id}|{start}|{end}"
    path = key_path("fred", key)
    cached = _cache_read(path)
    if cached is not None:
        return cached

    urls = [
        f"https://fred.stlouisfed.org/series/{series_id}/downloaddata/{series_id}.csv",
        f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}",
    ]

    sess = requests.Session()
    # CRITICAL: ignore proxy environment variables that may be misconfigured
    sess.trust_env = False
    headers = {"User-Agent": "portfolio-tester/0.1"}

    last_exc = None
    for url in urls:
        try:
            r = sess.get(
                url,
                timeout=30,
                verify=certifi.where(),
                headers=headers,
                allow_redirects=True,
                proxies={"http": None, "https": None},
            )
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))

            # Normalize CSV: expect a observation_date column + value column
            if "observation_date" not in df.columns:
                raise ValueError("CSV missing observation_date column")
            df["observation_date"] = pd.to_datetime(df["observation_date"])

            # Pick the value column
            if series_id in df.columns:
                value_col = series_id
            else:
                # fredgraph.csv returns the series as a non-observation_date column
                value_cols = [c for c in df.columns if c != "observation_date"]
                if not value_cols:
                    raise ValueError("CSV missing value column")
                value_col = value_cols[0]
                df = df.rename(columns={value_col: series_id})

            df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
            df = df.dropna(subset=[series_id]).set_index("observation_date")[[series_id]]

            if start is not None:
                df = df[df.index >= pd.to_datetime(start)]
            if end is not None:
                df = df[df.index <= pd.to_datetime(end)]

            df = df.resample("ME").last()
            _cache_write(df, path)
            return df
        except Exception as e:
            last_exc = e
            continue

    raise RuntimeError(f"Failed to fetch FRED series {series_id}: {last_exc}")

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
    # Fix first warning by explicitly setting fill_method=None
    infl_m = cpi["CPIAUCSL"].pct_change(fill_method=None).reindex(rets_m.index).fillna(0.0)

    rf_m = ((1.0 + (tb3.reindex(rets_m.index)["TB3MS"] / 100.0)) ** (1/12.0) - 1.0)
    # Fix second warning by using ffill() method instead of fillna(method="ffill")
    rf_m = rf_m.ffill().fillna(0.0)

    return rets_m, infl_m.rename("inflation_m"), rf_m.rename("rf_m")
