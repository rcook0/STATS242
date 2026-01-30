from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from .util import ensure_dir

def save_fig(fig, out_path: str) -> str:
    ensure_dir(os.path.dirname(out_path))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path

def plot_series(y: pd.Series, title: str, out_path: str, xlabel: str = "", ylabel: str = "") -> str:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(y.to_numpy())
    ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    return save_fig(fig, out_path)

def plot_hist(y: pd.Series, title: str, out_path: str, bins: int = 60) -> str:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    arr = pd.to_numeric(y, errors="coerce").dropna().to_numpy()
    ax.hist(arr, bins=bins, density=True, alpha=0.9)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return save_fig(fig, out_path)

def plot_qq(y: pd.Series, title: str, out_path: str) -> str:
    arr = pd.to_numeric(y, errors="coerce").dropna().to_numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    stats.probplot(arr, dist="norm", plot=ax)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return save_fig(fig, out_path)

def plot_acf_bar(acf: pd.Series, title: str, out_path: str) -> str:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(acf.index.to_numpy(), acf.to_numpy())
    ax.set_title(title)
    ax.set_xlabel("Lag")
    ax.grid(True, alpha=0.3)
    return save_fig(fig, out_path)

def plot_scatter_fit(x: pd.Series, y: pd.Series, title: str, out_path: str) -> str:
    xv = pd.to_numeric(x, errors="coerce").dropna()
    yv = pd.to_numeric(y, errors="coerce").dropna()
    idx = xv.index.intersection(yv.index)
    xv = xv.loc[idx].to_numpy()
    yv = yv.loc[idx].to_numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xv, yv, s=8, alpha=0.7)
    if len(xv) >= 2:
        m, b = np.polyfit(xv, yv, 1)
        xs = np.linspace(xv.min(), xv.max(), 50)
        ax.plot(xs, m*xs + b)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return save_fig(fig, out_path)

def plot_bar(weights: dict, title: str, out_path: str) -> str:
    keys = list(weights.keys())
    vals = [weights[k] for k in keys]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(keys, vals)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    return save_fig(fig, out_path)

def plot_cum_pnl(pnl: pd.Series, title: str, out_path: str) -> str:
    arr = pd.to_numeric(pnl, errors="coerce").fillna(0.0)
    cum = arr.cumsum()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(cum.to_numpy())
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return save_fig(fig, out_path)
