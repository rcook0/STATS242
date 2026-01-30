from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import pandas as pd

@dataclass(frozen=True)
class Contract:
    kind: str
    required_cols: Tuple[str, ...]
    optional_cols: Tuple[str, ...]
    ts_col: str = "ts"
    description: str = ""

def _c(kind: str, req: Tuple[str, ...], opt: Tuple[str, ...], desc: str) -> Contract:
    return Contract(kind=kind, required_cols=req, optional_cols=opt, description=desc)

CONTRACTS: Dict[str, Contract] = {
    "prices": _c("prices", ("ts","close"), (), "Single-asset price series."),
    "prices_multi": _c("prices_multi", ("ts",), ("asset_*",), "Multi-asset price panel; columns after ts are asset prices."),
    "returns_capm": _c("returns_capm", ("ts","strategy","market"), ("rf",), "Return series for CAPM regressions."),
    "trades": _c("trades", ("ts","price"), ("size","side"), "Trade prints."),
    "quotes": _c("quotes", ("ts","bid","ask"), ("bid_size","ask_size"), "Top-of-book quotes."),
    "events": _c("events", ("ts",), (), "Event timestamps (durations derived)."),
    "lob": _c("lob", ("ts","mid","bid_size","ask_size"), (), "Level-1 LOB snapshots."),
}

def describe_contract(kind: str) -> str:
    if kind not in CONTRACTS:
        raise KeyError(f"Unknown contract kind: {kind}")
    c = CONTRACTS[kind]
    opt = ", ".join(c.optional_cols) if c.optional_cols else "(none)"
    return (
        f"Kind: {c.kind}\n"
        f"Required columns: {', '.join(c.required_cols)}\n"
        f"Optional columns: {opt}\n"
        f"Description: {c.description}"
    )

def validate(kind: str, df: pd.DataFrame, *, strict: bool = False) -> Dict:
    if kind not in CONTRACTS:
        raise KeyError(f"Unknown contract kind: {kind}")
    c = CONTRACTS[kind]
    missing = [col for col in c.required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for {kind}: {missing}")

    out: Dict = {"kind": kind, "n_rows": int(len(df)), "columns": list(df.columns)}
    ts = pd.to_datetime(df[c.ts_col], utc=True, errors="coerce")
    out["ts_parse_failures"] = int(ts.isna().sum())
    if strict and out["ts_parse_failures"] > 0:
        raise ValueError(f"Unparseable timestamps: {out['ts_parse_failures']}")

    ts_ok = ts.dropna()
    out["ts_monotone_non_decreasing"] = bool(ts_ok.is_monotonic_increasing)
    out["ts_duplicates"] = int(ts_ok.duplicated().sum())
    if strict and not out["ts_monotone_non_decreasing"]:
        raise ValueError("Timestamps not monotone non-decreasing")

    def _num(col: str):
        return pd.to_numeric(df[col], errors="coerce")

    if kind == "prices":
        close = _num("close")
        out["close_nan"] = int(close.isna().sum())
        out["close_nonpositive"] = int((close <= 0).sum(skipna=True))
        if strict and out["close_nonpositive"] > 0:
            raise ValueError("Found nonpositive close prices")

    if kind == "quotes":
        bid = _num("bid"); ask = _num("ask")
        out["bid_gt_ask"] = int((bid > ask).sum(skipna=True))
        if strict and out["bid_gt_ask"] > 0:
            raise ValueError("Found rows with bid > ask")

    if kind == "trades":
        p = _num("price")
        out["price_nonpositive"] = int((p <= 0).sum(skipna=True))
        if strict and out["price_nonpositive"] > 0:
            raise ValueError("Found nonpositive trade prices")
        if "size" in df.columns:
            sz = _num("size")
            out["size_negative"] = int((sz < 0).sum(skipna=True))

    if kind == "events":
        dt = pd.to_datetime(df["ts"], utc=True, errors="coerce").diff().dt.total_seconds().dropna()
        out["duration_nonpositive"] = int((dt <= 0).sum())
        out["duration_p50"] = float(dt.quantile(0.5)) if len(dt) else None
        out["duration_p90"] = float(dt.quantile(0.9)) if len(dt) else None

    if kind == "lob":
        mid = _num("mid")
        out["mid_nonpositive"] = int((mid <= 0).sum(skipna=True))
        bs = _num("bid_size"); aS = _num("ask_size")
        out["bid_size_negative"] = int((bs < 0).sum(skipna=True))
        out["ask_size_negative"] = int((aS < 0).sum(skipna=True))

    return out
