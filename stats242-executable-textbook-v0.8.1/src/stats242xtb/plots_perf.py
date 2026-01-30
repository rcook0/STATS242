from __future__ import annotations
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .plots import save_fig
from .util import ensure_dir

def plot_equity(equity: pd.Series, title: str, out_path: str) -> str:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(equity.index, equity.to_numpy())
    ax.set_title(title)
    ax.set_xlabel("time")
    ax.set_ylabel("equity")
    ax.grid(True, alpha=0.3)
    return save_fig(fig, out_path)

def plot_equity_multi(equities: Dict[str, pd.Series], title: str, out_path: str) -> str:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for k, eq in equities.items():
        ax.plot(eq.index, eq.to_numpy(), label=k)
    ax.set_title(title)
    ax.set_xlabel("time")
    ax.set_ylabel("equity")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    return save_fig(fig, out_path)

def plot_drawdown(dd: pd.Series, title: str, out_path: str) -> str:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(dd.index, dd.to_numpy())
    ax.set_title(title)
    ax.set_xlabel("time")
    ax.set_ylabel("drawdown")
    ax.grid(True, alpha=0.3)
    return save_fig(fig, out_path)

def plot_turnover(turnover: pd.Series, title: str, out_path: str) -> str:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(turnover.index, turnover.to_numpy())
    ax.set_title(title)
    ax.set_xlabel("time")
    ax.set_ylabel("turnover")
    ax.grid(True, alpha=0.3)
    return save_fig(fig, out_path)

def plot_cost_curve(cost_to_metric: Dict[float, float], title: str, out_path: str, ylabel: str = "metric") -> str:
    xs = np.array(sorted(cost_to_metric.keys()), dtype=float)
    ys = np.array([cost_to_metric[x] for x in xs], dtype=float)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xs, ys, marker="o")
    ax.set_title(title)
    ax.set_xlabel("cost (bps)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    return save_fig(fig, out_path)

def plot_weights_heatmap(W: pd.DataFrame, title: str, out_path: str, max_rows: int = 250) -> str:
    # downsample for readability if necessary
    X = W.copy()
    if len(X) > max_rows:
        idx = np.linspace(0, len(X) - 1, max_rows).astype(int)
        X = X.iloc[idx]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(X.to_numpy(), aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("asset")
    ax.set_ylabel("time (downsampled)")
    ax.set_xticks(range(len(X.columns)))
    ax.set_xticklabels(list(X.columns), rotation=90, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return save_fig(fig, out_path)
