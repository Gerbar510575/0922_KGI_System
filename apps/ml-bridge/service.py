from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Any, Dict, List
import subprocess, json, os, yaml, logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("ml-bridge")

app = FastAPI(title="ML Bridge Service")

CONFIG_PATH = os.getenv("MLBRIDGE_CONFIG", "/app/config/models.yaml")
TIMEOUT_SEC = int(os.getenv("SUBPROC_TIMEOUT", "15"))
VISUAL_DIR = os.getenv("VISUAL_DIR", "/app/visualizations")

# 僅允許這些可執行程式，防止 YAML config 被竄改後執行任意命令
ALLOWED_EXECUTABLES: frozenset[str] = frozenset({
    "Rscript", "java", "python", "python3", "./predict",
})
# workdir 必須在此目錄之下（容器內）
ALLOWED_WORKDIR_PREFIX = "/app"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    MODELS = yaml.safe_load(f).get("models", {})


class PredictIn(BaseModel):
    model_id: str
    input: Dict[str, Any]


def _validate_model_config(model_id: str, cmd: List[str], workdir: str) -> None:
    """驗證 YAML config 的 cmd 與 workdir，防止任意命令執行。"""
    if not cmd:
        raise HTTPException(status_code=500, detail=f"model {model_id} cmd not configured")
    executable = cmd[0]
    if executable not in ALLOWED_EXECUTABLES:
        logger.error(f"model {model_id} 使用了不在白名單的可執行程式：{executable}")
        raise HTTPException(status_code=500, detail=f"model {model_id} uses disallowed executable")
    resolved = os.path.realpath(workdir)
    if not resolved.startswith(ALLOWED_WORKDIR_PREFIX):
        logger.error(f"model {model_id} workdir 超出允許範圍：{workdir} → {resolved}")
        raise HTTPException(status_code=500, detail=f"model {model_id} workdir outside allowed prefix")


def _run_model(model_id: str, input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """執行單個模型，回傳 JSON 結果"""
    if model_id not in MODELS:
        raise HTTPException(status_code=404, detail=f"unknown model_id: {model_id}")

    cfg = MODELS[model_id]
    cmd: List[str] = cfg.get("cmd", [])
    workdir: str = cfg.get("workdir", "/app")

    _validate_model_config(model_id, cmd, workdir)

    logger.info(f"執行模型 {model_id}：cmd={cmd}, workdir={workdir}")
    try:
        proc = subprocess.run(
            cmd,
            input=json.dumps(input_dict),
            text=True,
            capture_output=True,
            cwd=workdir,
            timeout=TIMEOUT_SEC,
            check=False,
        )
    except subprocess.TimeoutExpired:
        logger.warning(f"模型 {model_id} 執行逾時（{TIMEOUT_SEC}s）")
        return {"status": "error", "error": "timeout"}
    except Exception as e:
        logger.exception(f"模型 {model_id} 啟動失敗")
        return {"status": "error", "error": f"spawn error: {e}"}

    if proc.returncode != 0:
        logger.error(f"模型 {model_id} 回傳非零退出碼：{proc.stderr.strip()[:200]}")
        return {"status": "error", "error": proc.stderr.strip()}

    out = (proc.stdout or "").strip()
    try:
        payload = json.loads(out)
    except json.JSONDecodeError:
        logger.warning(f"模型 {model_id} 輸出無法解析為 JSON")
        payload = {"raw": out}

    logger.info(f"模型 {model_id} 執行成功")
    return {"status": "ok", "output": payload}


@app.get("/health")
def health():
    return {"status": "ok", "models": list(MODELS.keys())}


@app.post("/predict")
def predict(req: PredictIn):
    result = _run_model(req.model_id, req.input)
    return {"model_id": req.model_id, **result}


@app.post("/predict_all")
def predict_all(req: Dict[str, Any]):
    if "input" not in req:
        raise HTTPException(status_code=400, detail="missing input")

    results = {}
    for model_id in MODELS.keys():
        results[model_id] = _run_model(model_id, req["input"])
    return {"results": results}


@app.get("/visualizations/{filename}")
def get_visualization(filename: str):
    """提供訓練過程的視覺化圖片"""
    file_path = os.path.join(VISUAL_DIR, filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(file_path)

