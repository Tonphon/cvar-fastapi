import os, json
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from typing import Optional

from .settings import settings
from .schemas import RunConfig, RunCreateResponse, RunStatus, MemoResponse
from .storage import new_run_id, ensure_run_dirs, run_dir, get_status, list_files, read_json
from .services.runner import run_pipeline

app = FastAPI(title="CVaR SA Optimizer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve run artifacts
os.makedirs(settings.RUNS_DIR, exist_ok=True)
app.mount("/runs", StaticFiles(directory=settings.RUNS_DIR), name="runs")

@app.post("/api/runs", response_model=RunCreateResponse)
async def create_run(
    background: BackgroundTasks,
    portfolio: UploadFile = File(...),
    config_json: str = Form(...),
    do_memo: bool = Form(False),
    memo_model: Optional[str] = Form(None),
):
    run_id = new_run_id()
    ensure_run_dirs(run_id)

    # save portfolio.csv
    portfolio_path = os.path.join(run_dir(run_id), "portfolio.csv")
    with open(portfolio_path, "wb") as f:
        f.write(await portfolio.read())

    # parse config
    try:
        cfg_dict = json.loads(config_json)
        cfg = RunConfig.model_validate(cfg_dict)  # validates types/ranges
        cfg_dict = cfg.model_dump(by_alias=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid config_json: {e}")

    # basic bounds to avoid abuse
    if cfg_dict["iters"] > 20000:
        raise HTTPException(status_code=400, detail="iters too high for MVP (max 20000).")

    background.add_task(run_pipeline, run_id, portfolio_path, cfg_dict, do_memo, memo_model)
    return RunCreateResponse(run_id=run_id, status="queued", message="Run started")

@app.get("/api/runs/{run_id}", response_model=RunStatus)
def get_run(run_id: str):
    st = get_status(run_id)
    if not os.path.exists(run_dir(run_id)):
        raise HTTPException(status_code=404, detail="run_id not found")
    files = list_files(run_id) if st["status"] in ("done", "error") else []
    return RunStatus(run_id=run_id, status=st["status"], error=st.get("error"), files=files)

@app.get("/api/runs/{run_id}/summary")
def get_summary(run_id: str):
    path = os.path.join(run_dir(run_id), "summary.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="summary not found (run not done yet?)")
    return read_json(path)

@app.get("/api/runs/{run_id}/memo", response_model=MemoResponse)
def get_memo(run_id: str):
    path = os.path.join(run_dir(run_id), "llm_memo.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="memo not found")
    data = read_json(path)
    return MemoResponse(run_id=run_id, model=data.get("model") or settings.DEFAULT_MODEL, memo=data["memo"])
