import requests, os

MLB = os.getenv("ML_BRIDGE_URL", "http://ml-bridge:7000")

def infer_risk_via_bridge(kyc: dict) -> dict:
    """
    呼叫 ML Bridge 的 R 模型 (risk_r) 推斷風險。
    回傳格式: {"type": "保守/穩健/積極", "p": 機率值}
    """
    try:
        resp = requests.post(f"{MLB}/predict",
                             json={"model_id": "risk_r", "input": kyc},
                             timeout=20)
        resp.raise_for_status()
        out = resp.json().get("output", {})

        prob = out.get("prob")
        if prob is not None:
            p = float(prob[0]) if isinstance(prob, list) else float(prob)
            if p > 0.7:
                return {"type": "積極", "p": p}
            if p > 0.4:
                return {"type": "穩健", "p": p}
            return {"type": "保守", "p": p}

        preds = out.get("predictions")
        if preds:
            v = str(preds[0])
            if v in ["1","Y","Yes","Positive","TRUE"]:
                return {"type": "積極", "p": 0.8}
            else:
                return {"type": "保守", "p": 0.2}

    except Exception:
        pass

    # fallback
    return {"type": "穩健", "p": 0.5}
