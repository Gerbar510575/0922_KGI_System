from fastapi import FastAPI, HTTPException
from jinja2 import Environment, FileSystemLoader
import markdown, io, base64, matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

app = FastAPI(title="Report Service")
env = Environment(loader=FileSystemLoader("templates"), autoescape=True)
tpl = env.get_template("report_zh.md.j2")

def plot_price(dates, prices, title="基金價格走勢"):
    if not dates or not prices:
        return ""
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(dates, prices)
    ax.set_title(title)
    ax.grid(True)
    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png", dpi=160)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def md_to_pdf(md_text: str) -> bytes:
    """將 Markdown 轉成 PDF（簡易版，不支援圖片 base64 渲染）。"""
    styles = getSampleStyleSheet()
    story = []
    for line in md_text.splitlines():
        if not line.strip():
            story.append(Spacer(1, 12))
        else:
            story.append(Paragraph(line, styles["Normal"]))
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    doc.build(story)
    return buf.getvalue()

@app.post("/report")
def generate(payload: dict):
    try:
        charts = {}

        # 如果 payload 有價格序列，就生成走勢圖
        if "series" in payload:
            charts["price"] = plot_price(
                payload["series"].get("dates"),
                payload["series"].get("prices"),
                title=payload["series"].get("title", "基金走勢")
            )

        # 渲染 Markdown 報告
        md = tpl.render(**payload, charts=charts)
        html = markdown.markdown(md, extensions=["tables"])
        pdf_bytes = md_to_pdf(md)

        return {
            "markdown": md,
            "html": html,
            "pdf": base64.b64encode(pdf_bytes).decode()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"report generation failed: {e}")



