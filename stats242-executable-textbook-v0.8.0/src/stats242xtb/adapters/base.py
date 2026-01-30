from __future__ import annotations
import re
from typing import Any, Dict, Optional
import pandas as pd
import numpy as np

def _find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower()
        if key in cols_lower:
            return cols_lower[key]
    for cand in candidates:
        if cand.startswith("re:"):
            pat = re.compile(cand[3:], flags=re.IGNORECASE)
            for c in df.columns:
                if pat.match(c):
                    return c
    return None

def parse_timestamp(df: pd.DataFrame, mapping: Dict[str, Any] | None = None) -> pd.Series:
    m = mapping or {}
    if "ts" in m:
        col = m["ts"]
        return pd.to_datetime(df[col], utc=True, errors="coerce")

    col = _find_col(df, ["ts", "timestamp", "datetime", "date_time", "time_stamp"])
    if col is not None:
        return pd.to_datetime(df[col], utc=True, errors="coerce")

    date_col = m.get("date") or _find_col(df, ["date", "DATE", "TradeDate", "Quotedate"])
    time_col = m.get("time") or _find_col(df, ["time", "TIME", "TradeTime", "QuoteTime"])
    if date_col is not None and time_col is not None:
        s = df[date_col].astype(str).str.strip() + " " + df[time_col].astype(str).str.strip()
        return pd.to_datetime(s, utc=True, errors="coerce")

    epoch_col = m.get("epoch") or _find_col(df, ["epoch", "EPOCH", "ts_ns", "timestamp_ns", "timestamp_ms", "timestamp_us", "timestamp_s"])
    if epoch_col is not None:
        unit = m.get("epoch_unit")
        if unit is None:
            n = epoch_col.lower()
            if "ns" in n: unit = "ns"
            elif "us" in n: unit = "us"
            elif "ms" in n: unit = "ms"
            else: unit = "s"
        return pd.to_datetime(df[epoch_col], unit=unit, utc=True, errors="coerce")

    raise ValueError("Could not infer timestamp. Provide --map with ts/date+time/epoch mapping.")

def canonicalize_trades(df: pd.DataFrame, schema: str = "auto", mapping: Dict[str, Any] | None = None) -> pd.DataFrame:
    m = mapping or {}
    ts = parse_timestamp(df, m)
    pcol = m.get("price") or _find_col(df, ["price", "PRICE", "TradePrice", "re:^TRDPRC\d+$"])
    if pcol is None:
        raise ValueError("Could not infer trade price column. Provide mapping.price.")
    size_col = m.get("size") or _find_col(df, ["size", "SIZE", "TradeSize", "volume", "VOL", "re:^TRDVOL\d+$"])
    side_col = m.get("side") or _find_col(df, ["side", "SIDE", "BuySell", "BS", "direction"])

    out = pd.DataFrame({"ts": ts, "price": pd.to_numeric(df[pcol], errors="coerce")})
    if size_col is not None:
        out["size"] = pd.to_numeric(df[size_col], errors="coerce")
    if side_col is not None:
        s = df[side_col].astype(str).str.upper().str.strip()
        s = s.replace({"BUY": "B", "SELL": "S", "1": "B", "-1": "S"})
        out["side"] = s.where(s.isin(["B", "S"]), np.nan)
    out = out.dropna(subset=["ts", "price"]).sort_values("ts").reset_index(drop=True)
    return out

def canonicalize_quotes(df: pd.DataFrame, schema: str = "auto", mapping: Dict[str, Any] | None = None) -> pd.DataFrame:
    m = mapping or {}
    ts = parse_timestamp(df, m)
    bid_col = m.get("bid") or _find_col(df, ["bid", "BID", "BestBid", "BID_PRICE", "re:^BID\d+$"])
    ask_col = m.get("ask") or _find_col(df, ["ask", "ASK", "BestAsk", "ASK_PRICE", "re:^ASK\d+$"])
    if bid_col is None or ask_col is None:
        raise ValueError("Could not infer bid/ask columns. Provide mapping.bid/mapping.ask.")
    bid_sz_col = m.get("bid_size") or _find_col(df, ["bid_size", "BIDSIZ", "BID_SIZE", "BestBidSize", "re:^BIDSIZ\d+$"])
    ask_sz_col = m.get("ask_size") or _find_col(df, ["ask_size", "ASKSIZ", "ASK_SIZE", "BestAskSize", "re:^ASKSIZ\d+$"])
    out = pd.DataFrame({"ts": ts, "bid": pd.to_numeric(df[bid_col], errors="coerce"), "ask": pd.to_numeric(df[ask_col], errors="coerce")})
    if bid_sz_col is not None: out["bid_size"] = pd.to_numeric(df[bid_sz_col], errors="coerce")
    if ask_sz_col is not None: out["ask_size"] = pd.to_numeric(df[ask_sz_col], errors="coerce")
    out = out.dropna(subset=["ts", "bid", "ask"]).sort_values("ts").reset_index(drop=True)
    return out

def canonicalize_prices(df: pd.DataFrame, schema: str = "auto", mapping: Dict[str, Any] | None = None) -> pd.DataFrame:
    m = mapping or {}
    ts = parse_timestamp(df, m)
    close_col = m.get("close") or _find_col(df, ["close", "CLOSE", "Close", "adj_close", "AdjClose", "price", "PRICE", "last"])
    if close_col is None:
        raise ValueError("Could not infer close/price column. Provide mapping.close.")
    out = pd.DataFrame({"ts": ts, "close": pd.to_numeric(df[close_col], errors="coerce")})
    out = out.dropna().sort_values("ts").reset_index(drop=True)
    return out

def canonicalize_events(df: pd.DataFrame, schema: str = "auto", mapping: Dict[str, Any] | None = None) -> pd.DataFrame:
    ts = parse_timestamp(df, mapping or {})
    out = pd.DataFrame({"ts": ts})
    out = out.dropna().sort_values("ts").reset_index(drop=True)
    return out

def canonicalize_lob(df: pd.DataFrame, schema: str = "auto", mapping: Dict[str, Any] | None = None) -> pd.DataFrame:
    m = mapping or {}
    ts = parse_timestamp(df, m)
    mid_col = m.get("mid") or _find_col(df, ["mid", "MID", "midprice", "MIDPRICE", "re:^MID\d+$"])
    bid_sz_col = m.get("bid_size") or _find_col(df, ["bid_size", "BIDSIZ", "BID_SIZE", "BestBidSize"])
    ask_sz_col = m.get("ask_size") or _find_col(df, ["ask_size", "ASKSIZ", "ASK_SIZE", "BestAskSize"])
    if mid_col is None:
        raise ValueError("Could not infer mid column. Provide mapping.mid.")
    if bid_sz_col is None or ask_sz_col is None:
        raise ValueError("Could not infer bid/ask size columns. Provide mapping.bid_size/mapping.ask_size.")
    out = pd.DataFrame({"ts": ts, "mid": pd.to_numeric(df[mid_col], errors="coerce"),
                        "bid_size": pd.to_numeric(df[bid_sz_col], errors="coerce"),
                        "ask_size": pd.to_numeric(df[ask_sz_col], errors="coerce")})
    out = out.dropna().sort_values("ts").reset_index(drop=True)
    return out
