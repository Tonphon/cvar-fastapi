# CVaR Portfolio Optimizer API (FastAPI)

## Overview

This repository contains the FastAPI backend for a CVaR-based portfolio optimizer using Simulated Annealing (SA). The API accepts a portfolio CSV and a JSON config, downloads historical prices for the listed tickers, builds daily returns, optimizes portfolio weights under constraints, generates plots, and serves all run artifacts via HTTP.

**Frontend UI (recommended):** https://github.com/Tonphon/cvar-frontend

## What CVaR Means

- **VaR(α)**: A loss cutoff such that only the worst α fraction of days are worse than it.
- **CVaR(α)**: The average loss within the worst α fraction of days (tail-risk severity).

## Requirements

- Python 3.10+ recommended
- Internet access (for price download)
- Optional: OpenAI API key for memo generation

## Setup

### Clone and Install Dependencies

```bash
git clone https://github.com/Tonphon/cvar-fastapi.git
cd cvar-fastapi
python -m venv .venv
```

**Activate virtual environment:**

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

## Run (Local)

```bash
uvicorn app.main:app --reload
```

**Swagger UI:** http://127.0.0.1:8000/docs

## Environment Variables

### Optional (recommended on servers):
- `MPLBACKEND=Agg` (headless matplotlib backend)

### Optional (only if memo enabled):
- `OPENAI_API_KEY` (OpenAI key used for memo generation)

**PowerShell example:**
```powershell
$env:MPLBACKEND="Agg"
$env:OPENAI_API_KEY="sk-..."
uvicorn app.main:app --reload
```

## Input CSV Format

Single-row portfolio CSV:

```csv
date,AAPL.US,MSFT.US,SPY.US,GLD.US
2025-12-12,0.25,0.25,0.40,0.10
```

## Config Fields (Common)

- **lookback_days**: Number of past trading days to download
- **train_ratio**: Fraction used for optimization (e.g., 0.7)
- **alpha**: CVaR tail level (0.05 = worst 5% days)
- **objective**: `min_cvar` or `mean_minus_lambda_cvar`
- **lambda**: Risk-aversion weight (only used for `mean_minus_lambda_cvar`)
- **long_only**: Disallow negative weights (no shorting)
- **w_max**: Max weight per asset
- **turnover_max**: Max total absolute weight change from current weights
- **transaction_cost_bps**: Transaction cost per unit turnover (bps)
- **SA parameters**: `iters`, `step_size`, `init_temp`, `final_temp`, `seed`, `penalty_turnover`

## API Usage (curl)

### Start a run:

```bash
curl -X POST "http://127.0.0.1:8000/api/runs" ^
  -F "portfolio=@portfolio.csv" ^
  -F "config_json={\"lookback_days\":504,\"train_ratio\":0.7,\"alpha\":0.05,\"objective\":\"min_cvar\",\"lambda\":0.5,\"long_only\":true,\"w_max\":0.6,\"turnover_max\":0.5,\"transaction_cost_bps\":10,\"iters\":2000,\"step_size\":0.05,\"init_temp\":1.0,\"final_temp\":0.001,\"seed\":42,\"penalty_turnover\":50}" ^
  -F "do_memo=false"
```

### Poll status:
```bash
curl "http://127.0.0.1:8000/api/runs/<RUN_ID>"
```

### Get summary:
```bash
curl "http://127.0.0.1:8000/api/runs/<RUN_ID>/summary"
```

### Download a plot:
```bash
curl -O "http://127.0.0.1:8000/runs/<RUN_ID>/figures/equity_curve_test_only.png"
```

## Outputs

Each run produces `runs/<run_id>/` containing:

- `summary.json`, `weights_opt.csv`, `trades.csv`, `objective_history.csv`
- `portfolio_returns_full.csv`
- `figures/*.png` plots
- Optional memo output (if enabled)

## Build / Production

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
