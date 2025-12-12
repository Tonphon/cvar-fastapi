from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict, Any, List

class RunConfig(BaseModel):
    # data
    lookback_days: int = 504
    train_ratio: float = 0.70

    # risk/objective
    alpha: float = 0.05
    objective: Literal["min_cvar", "mean_minus_lambda_cvar"] = "min_cvar"
    lambda_: Optional[float] = Field(default=0.5, alias="lambda")

    # constraints
    long_only: bool = True
    w_max: float = 0.50
    turnover_max: float = 0.40

    # costs
    transaction_cost_bps: float = 10.0

    # SA
    seed: int = 42
    iters: int = 8000
    step_size: float = 0.05
    init_temp: float = 1.0
    final_temp: float = 0.001

    # penalties
    penalty_turnover: float = 50.0
    penalty_invalid: float = 1_000_000.0

class RunCreateResponse(BaseModel):
    run_id: str
    status: Literal["queued", "running", "done", "error"]
    message: str

class RunStatus(BaseModel):
    run_id: str
    status: Literal["queued", "running", "done", "error"]
    error: Optional[str] = None
    files: List[str] = []

class MemoResponse(BaseModel):
    run_id: str
    model: str
    memo: Dict[str, Any]
