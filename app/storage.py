import os, json, uuid
from typing import Dict, Any, List, Optional
from .settings import settings

def new_run_id() -> str:
    return uuid.uuid4().hex[:12]

def run_dir(run_id: str) -> str:
    return os.path.join(settings.RUNS_DIR, run_id)

def ensure_run_dirs(run_id: str) -> None:
    os.makedirs(run_dir(run_id), exist_ok=True)
    os.makedirs(os.path.join(run_dir(run_id), "figures"), exist_ok=True)

def write_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def list_files(run_id: str) -> List[str]:
    base = run_dir(run_id)
    out = []
    for root, _, files in os.walk(base):
        for fn in files:
            rel = os.path.relpath(os.path.join(root, fn), base)
            out.append(rel.replace("\\", "/"))
    return sorted(out)

def status_path(run_id: str) -> str:
    return os.path.join(run_dir(run_id), "status.json")

def set_status(run_id: str, status: str, error: Optional[str]=None) -> None:
    write_json(status_path(run_id), {"run_id": run_id, "status": status, "error": error})

def get_status(run_id: str) -> Dict[str, Any]:
    p = status_path(run_id)
    if not os.path.exists(p):
        return {"run_id": run_id, "status": "queued", "error": None}
    return read_json(p)
