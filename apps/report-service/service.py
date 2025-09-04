from fastapi import FastAPI, HTTPException
from jinja2 import Environment, FileSystemLoader
import markdown, io, base64, matplotlib.pyplot as plt

app = FastAPI(title="Report Service")
env = Environment(loader=FileSystemLoader("templates"), autoescape=True)
tpl = env.get_template("report_zh.md.j2")

def plot_pie(weights: dict):
    if not weights:
        return ""
    fig, ax = plt.subplots(figsize=(3.8,3.8))
    ax.pie(list(weights.values()), labels=list(weights.keys()), autopct="%1.0f%%")
    ax.set_title("建議基金配置")
    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png", dpi=160)
    plt.close(fig)
    return "data:image/png;base64,"+base64.b64encode(buf.getvalue()).decode()

def plot_price(dates, prices, title="基金價格走勢"):
    if not dates or not prices:
        return ""
    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(dates, prices)
    ax.set_title(title)
    ax.grid(True)
    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png", dpi=160)
    plt.close(fig)
    return "data:image/png;base64,"+base64.b64encode(buf.getvalue()).decode()

@app.post("/report")
def generate(payload: dict):
    try:
        advice = payload.get("advice", {})
        charts = {}
        if "target_allocation" in advice:
            charts["alloc_pie"] = plot_pie(advice["target_allocation"])
        if "series" in payload:
            charts["price"] = plot_price(
                payload["series"].get("dates"),
                payload["series"].get("prices"),
                title=payload["series"].get("title","基金走勢")
            )

        md = tpl.render(**payload, charts=charts)
        html = markdown.markdown(md, extensions=["tables"])
        return {"markdown": md, "html": html}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"report generation failed: {e}")

