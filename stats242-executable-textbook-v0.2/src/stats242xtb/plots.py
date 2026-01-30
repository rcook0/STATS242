from __future__ import annotations
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from .util import ensure_dir

def save_fig(fig, out_path: str) -> str:
    ensure_dir(os.path.dirname(out_path))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path

def plot_series(y: pd.Series, title: str, out_path: str) -> str:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(y.to_numpy())
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return save_fig(fig, out_path)

def plot_hist(y: pd.Series, title: str, out_path: str, bins: int = 60) -> str:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(y.to_numpy(), bins=bins, density=True, alpha=0.9)
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
