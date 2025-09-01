from fastapi import FastAPI
from jinja2 import Environment, FileSystemLoader
import markdown, io, base64, matplotlib.pyplot as plt

app = FastAPI(title="Report Service")
env = Environment(loader=FileSystemLoader("templates"), autoescape=True)
tpl = env.get_template("report_zh.md.j2")

def plot_pie(weights: dict):
    fig, ax = plt.subplots(figsize=(3.8,3.8))
    ax.pie(list(weights.values()), labels=list(weights.keys()), autopct="%1.0f%%")
    ax.set_title("建議資產配置")
    buf = io.BytesIO(); plt.tight_layout(); fig.savefig(buf, format="png", dpi=160); plt.close(fig)
    return "data:image/png;base64,"+base64.b64encode(buf.getvalue()).decode()

def plot_price(dates, prices, title="價格走勢"):
    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(dates, prices)
    ax.set_title(title); ax.grid(True)
    buf = io.BytesIO(); plt.tight_layout(); fig.savefig(buf, format="png", dpi=160); plt.close(fig)
    return "data:image/png;base64,"+base64.b64encode(buf.getvalue()).decode()

@app.post("/generate")
def generate(payload: dict):
    # payload: {client:{name,risk_level,goal,horizon_years}, advice:{...}, refs:{hits:[{source,chunk}]}, quotes:{...} 可選}
    advice = payload["advice"]
    charts = {
        "alloc_pie": plot_pie(advice["target_allocation"]),
    }
    # 可選：若有單一代表性商品之歷史價，畫一張
    if "series" in payload:
        charts["price"] = plot_price(payload["series"]["dates"], payload["series"]["prices"], title=payload["series"].get("title","走勢圖"))

    md = tpl.render(**payload, charts=charts)
    html = markdown.markdown(md, extensions=["tables"])
    return {"markdown": md, "html": html}
