import os, json
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

from .portfolio import load_portfolio_csv
from .data import build_returns
from .optimizer import OptCfg, make_feasible, anneal_with_history, turnover, tc_cost, metrics
from .plotting import ensure_dir, save_line, plot_return_hist, plot_equity_curve, plot_weights_before_after
from .report import worst_days_table
from .llm import generate_memo
from ..storage import run_dir, write_json, set_status

def run_pipeline(run_id: str, portfolio_path: str, cfg_dict: Dict[str, Any], do_memo: bool, memo_model: Optional[str]) -> None:
    out = run_dir(run_id)
    figdir = os.path.join(out, "figures")
    ensure_dir(figdir)

    try:
        set_status(run_id, "running")

        as_of, assets, w_prev_raw = load_portfolio_csv(portfolio_path)

        # feasibility check
        w_max = float(cfg_dict["w_max"])
        if cfg_dict["long_only"] and w_max * len(assets) < 1.0 - 1e-12:
            raise ValueError(f"Infeasible: long_only=true requires w_max >= 1/N. N={len(assets)} so w_max >= {1/len(assets):.4f}")

        optcfg = OptCfg(
            alpha=float(cfg_dict["alpha"]),
            objective=str(cfg_dict["objective"]),
            lam=float(cfg_dict.get("lambda", cfg_dict.get("lambda_", 0.5)) or 0.5),
            long_only=bool(cfg_dict["long_only"]),
            w_max=float(cfg_dict["w_max"]),
            turnover_max=float(cfg_dict["turnover_max"]),
            tc_bps=float(cfg_dict["transaction_cost_bps"]),
            seed=int(cfg_dict["seed"]),
            iters=int(cfg_dict["iters"]),
            step_size=float(cfg_dict["step_size"]),
            init_temp=float(cfg_dict["init_temp"]),
            final_temp=float(cfg_dict["final_temp"]),
            pen_turnover=float(cfg_dict["penalty_turnover"]),
            pen_invalid=float(cfg_dict["penalty_invalid"]),
        )

        w_prev = make_feasible(w_prev_raw, optcfg)

        end_date = pd.to_datetime(as_of) if as_of else None
        df_ret = build_returns(assets, int(cfg_dict["lookback_days"]), end_date=end_date)

        n = len(df_ret)
        cut = int(n * float(cfg_dict["train_ratio"]))
        if cut < 60 or (n - cut) < 60:
            raise ValueError("Not enough data for train/test. Increase lookback_days or adjust train_ratio.")

        R_train = df_ret.iloc[:cut].to_numpy(dtype=float)
        R_test  = df_ret.iloc[cut:].to_numpy(dtype=float)
        R_full  = df_ret.to_numpy(dtype=float)

        w_opt, best_obj, hist = anneal_with_history(R_train, optcfg, w_prev)

        # save core files
        pd.DataFrame({"asset": assets, "weight": w_opt}).to_csv(os.path.join(out, "weights_opt.csv"), index=False)
        d = w_opt - w_prev
        pd.DataFrame({"asset": assets, "w_prev": w_prev, "w_new": w_opt, "delta": d}).to_csv(os.path.join(out, "trades.csv"), index=False)

        hist.to_csv(os.path.join(out, "objective_history.csv"), index=False)

        dates = df_ret.index
        ret_prev = R_full @ w_prev
        ret_opt  = R_full @ w_opt
        split = np.array(["train"] * cut + ["test"] * (n - cut), dtype=object)
        returns_csv = os.path.join(out, "portfolio_returns_full.csv")
        pd.DataFrame({
            "date": dates.astype(str),
            "split": split,
            "ret_prev": ret_prev,
            "ret_opt": ret_opt,
        }).to_csv(returns_csv, index=False)

        summary = {
            "as_of": as_of,
            "objective": optcfg.objective,
            "alpha": optcfg.alpha,
            "lambda": (optcfg.lam if optcfg.objective == "mean_minus_lambda_cvar" else None),
            "best_objective_value": float(best_obj),
            "turnover": float(turnover(w_opt, w_prev)),
            "transaction_cost_est": float(tc_cost(w_opt, w_prev, optcfg.tc_bps)),
            "train": metrics(R_train, w_opt, optcfg.alpha),
            "test": metrics(R_test, w_opt, optcfg.alpha),
            "sanity": {
                "sum_weights": float(np.sum(w_opt)),
                "max_weight": float(np.max(w_opt)),
                "min_weight": float(np.min(w_opt)),
            }
        }
        write_json(os.path.join(out, "summary.json"), summary)

        # plots
        save_line(hist["iter"], hist["obj_best"], "Best objective vs iteration", "Iteration", "Best objective", os.path.join(figdir, "objective_vs_iteration.png"))
        save_line(hist["iter"], hist["cvar_best"], "Best train CVaR vs iteration", "Iteration", f"Best CVaR({optcfg.alpha:.2f})", os.path.join(figdir, "cvar_vs_iteration.png"))

        plot_return_hist(ret_opt[:cut], optcfg.alpha, "Train return distribution (optimized)", os.path.join(figdir, "return_hist_train_opt.png"))
        plot_return_hist(ret_opt[cut:], optcfg.alpha, "Test return distribution (optimized)", os.path.join(figdir, "return_hist_test_opt.png"))

        plot_equity_curve(dates, ret_prev, ret_opt, "Equity curve (full): Prev vs Optimized", os.path.join(figdir, "equity_curve_full.png"))
        plot_equity_curve(dates[cut:], ret_prev[cut:], ret_opt[cut:], "Equity curve (test only): Prev vs Optimized", os.path.join(figdir, "equity_curve_test_only.png"))

        plot_weights_before_after(assets, w_prev, w_opt, os.path.join(figdir, "weights_before_after.png"), top_n=12)

        # optional memo (compact payload)
        if do_memo:
            payload = {
                "summary": summary,
                "top_trades": pd.read_csv(os.path.join(out, "trades.csv")).assign(abs_delta=lambda x: x["delta"].abs())
                              .sort_values("abs_delta", ascending=False).head(6)[["asset","w_prev","w_new","delta"]].to_dict(orient="records"),
                "worst_test_days": worst_days_table(returns_csv, split="test", k=8),
                "plots": [
                    "figures/equity_curve_test_only.png",
                    "figures/return_hist_test_opt.png",
                    "figures/weights_before_after.png",
                ],
            }
            memo = generate_memo(payload, model=memo_model)
            write_json(os.path.join(out, "llm_memo.json"), {"model": memo_model, "memo": memo})

        set_status(run_id, "done")

    except Exception as e:
        set_status(run_id, "error", error=str(e))
