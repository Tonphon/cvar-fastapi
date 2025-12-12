import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Dict

@dataclass
class OptCfg:
    alpha: float
    objective: str
    lam: float
    long_only: bool
    w_max: float
    turnover_max: float
    tc_bps: float
    seed: int
    iters: int
    step_size: float
    init_temp: float
    final_temp: float
    pen_turnover: float
    pen_invalid: float

def compute_cvar(port_returns: np.ndarray, alpha: float) -> float:
    losses = -port_returns
    losses_sorted = np.sort(losses)
    k = max(1, int(math.ceil(alpha * len(losses_sorted))))
    return float(np.mean(losses_sorted[-k:]))

def project_long_only(w: np.ndarray, w_max: float) -> np.ndarray:
    w = np.clip(w, 0.0, w_max)
    s = float(w.sum())
    if s <= 1e-12:
        n = len(w)
        w[:] = min(w_max, 1.0 / n)
        s = float(w.sum())
    w = w / s

    for _ in range(2000):
        over = w > w_max + 1e-15
        if not np.any(over):
            break
        excess = float(np.sum(w[over] - w_max))
        w[over] = w_max
        under = w < w_max - 1e-15
        slack = (w_max - w[under])
        slack_sum = float(np.sum(slack))
        if slack_sum <= 1e-12:
            break
        w[under] = w[under] + excess * (slack / slack_sum)
        w = w / float(w.sum())

    w = np.clip(w, 0.0, w_max)
    return w / float(w.sum())

def make_feasible(w: np.ndarray, cfg: OptCfg) -> np.ndarray:
    if not cfg.long_only:
        raise NotImplementedError("MVP supports long_only=true only.")
    return project_long_only(w, cfg.w_max)

def turnover(w_new: np.ndarray, w_prev: np.ndarray) -> float:
    return float(np.sum(np.abs(w_new - w_prev)))

def tc_cost(w_new: np.ndarray, w_prev: np.ndarray, bps: float) -> float:
    return turnover(w_new, w_prev) * (bps / 10000.0)

def objective_value(w: np.ndarray, R_train: np.ndarray, cfg: OptCfg, w_prev: np.ndarray) -> Tuple[float, float, float]:
    if not np.isfinite(w).all():
        return cfg.pen_invalid, float("inf"), float("-inf")

    port = R_train @ w
    cvar = compute_cvar(port, cfg.alpha)
    mu = float(np.mean(port))

    cost = tc_cost(w, w_prev, cfg.tc_bps)
    pen = 0.0
    to = turnover(w, w_prev)
    if to > cfg.turnover_max:
        pen += cfg.pen_turnover * (to - cfg.turnover_max)

    if cfg.objective == "min_cvar":
        obj = cvar + cost + pen
    else:
        obj = (-mu) + cfg.lam * cvar + cost + pen

    return float(obj), float(cvar), float(mu)

def temp_at(k: int, cfg: OptCfg) -> float:
    frac = k / max(1, (cfg.iters - 1))
    return cfg.init_temp * ((cfg.final_temp / cfg.init_temp) ** frac)

def propose(w: np.ndarray, cfg: OptCfg, rng: np.random.Generator) -> np.ndarray:
    n = len(w)
    i = int(rng.integers(0, n))
    j = int(rng.integers(0, n - 1))
    if j >= i:
        j += 1
    delta = float(rng.uniform(-cfg.step_size, cfg.step_size))
    w2 = w.copy()
    w2[i] -= delta
    w2[j] += delta
    return make_feasible(w2, cfg)

def anneal_with_history(R_train: np.ndarray, cfg: OptCfg, w_prev: np.ndarray) -> Tuple[np.ndarray, float, pd.DataFrame]:
    rng = np.random.default_rng(cfg.seed)

    w_cur = w_prev.copy()
    obj_cur, cvar_cur, mu_cur = objective_value(w_cur, R_train, cfg, w_prev)

    w_best = w_cur.copy()
    obj_best, cvar_best, mu_best = obj_cur, cvar_cur, mu_cur

    rows = []
    for k in range(cfg.iters):
        T = temp_at(k, cfg)
        w_new = propose(w_cur, cfg, rng)
        obj_new, cvar_new, mu_new = objective_value(w_new, R_train, cfg, w_prev)

        d = obj_new - obj_cur
        accept = (d <= 0) or (T > 0 and rng.random() < math.exp(-d / T))

        if accept:
            w_cur, obj_cur, cvar_cur, mu_cur = w_new, obj_new, cvar_new, mu_new
            if obj_cur < obj_best:
                w_best = w_cur.copy()
                obj_best, cvar_best, mu_best = obj_cur, cvar_cur, mu_cur

        rows.append({
            "iter": k,
            "temp": T,
            "accepted": int(accept),
            "obj_best": obj_best,
            "cvar_best": cvar_best,
            "mean_best": mu_best,
        })

    return w_best, float(obj_best), pd.DataFrame(rows)

def metrics(R: np.ndarray, w: np.ndarray, alpha: float) -> Dict[str, float]:
    port = R @ w
    return {
        "mean": float(np.mean(port)),
        "stdev": float(np.std(port, ddof=1)),
        "cvar": float(compute_cvar(port, alpha)),
        "min": float(np.min(port)),
        "max": float(np.max(port)),
    }
