import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .optimizer import compute_cvar

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def compute_var_threshold(port_returns: np.ndarray, alpha: float) -> float:
    return float(np.quantile(port_returns, alpha))

def save_line(x, y, title, xlabel, ylabel, outpath):
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_return_hist(ret: np.ndarray, alpha: float, title: str, outpath: str) -> None:
    var_thr = compute_var_threshold(ret, alpha)
    cvar = compute_cvar(ret, alpha)

    plt.figure()
    plt.hist(ret, bins=40)
    plt.axvline(var_thr, linewidth=2, label=f"VaR(q{alpha:.2f}) = {var_thr:.4f}")
    plt.title(title)
    plt.xlabel("Daily return")
    plt.ylabel("Frequency")
    plt.legend()
    plt.text(0.02, 0.95, f"CVaR({alpha:.2f}) = {cvar:.4f}", transform=plt.gca().transAxes, va="top")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_equity_curve(dates, ret_prev, ret_new, title, outpath):
    eq_prev = np.cumprod(1.0 + ret_prev)
    eq_new = np.cumprod(1.0 + ret_new)
    plt.figure()
    plt.plot(dates, eq_prev, label="Prev")
    plt.plot(dates, eq_new, label="Optimized")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_weights_before_after(assets, w_prev, w_new, outpath, top_n=12):
    order = np.argsort(-w_new)[: min(top_n, len(w_new))]
    a = [assets[i] for i in order]
    prev = w_prev[order]
    new = w_new[order]
    x = np.arange(len(a)); width = 0.4
    plt.figure()
    plt.bar(x - width/2, prev, width, label="Prev")
    plt.bar(x + width/2, new, width, label="New")
    plt.title("Weights: before vs after")
    plt.xticks(x, a, rotation=45, ha="right")
    plt.ylabel("Weight")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
