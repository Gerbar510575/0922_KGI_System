from fastapi import FastAPI, HTTPException
from jinja2 import Environment, FileSystemLoader
import markdown, io, base64, matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

app = FastAPI(title="Report Service")
env = Environment(loader=FileSystemLoader("templates"), autoescape=True)
tpl = env.get_template("report_zh.md.j2")


# === 畫圖 ===
def plot_price(dates, prices, title="基金價格走勢"):
    """回傳 base64 圖片給前端 HTML/Markdown 用"""
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


def plot_price_raw(dates, prices, title="基金價格走勢"):
    """回傳 BytesIO 給 PDF 內嵌用"""
    if not dates or not prices:
        return None
    buf = io.BytesIO()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(dates, prices)
    ax.set_title(title)
    ax.grid(True)
    plt.tight_layout()
    fig.savefig(buf, format="png", dpi=160)
    plt.close(fig)
    buf.seek(0)
    return buf


# === Markdown 轉 PDF ===
def md_to_pdf(md_text: str, charts: dict) -> bytes:
    styles = getSampleStyleSheet()
    story = []
    for line in md_text.splitlines():
        if not line.strip():
            story.append(Spacer(1, 12))
        else:
            # 簡單判斷 markdown 標題
            if line.startswith("## "):
                story.append(Paragraph(f"<b>{line[3:]}</b>", styles["Heading2"]))
            elif line.startswith("# "):
                story.append(Paragraph(f"<b>{line[2:]}</b>", styles["Heading1"]))
            else:
                story.append(Paragraph(line, styles["Normal"]))
    # 插入圖表（如果有）
    if charts.get("price_raw"):
        story.append(Spacer(1, 12))
        story.append(Image(charts["price_raw"], width=400, height=200))
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    doc.build(story)
    return buf.getvalue()


# === API ===
@app.post("/report")
def generate(payload: dict):
    try:
        charts = {}

        # 如果 payload 有價格序列，就生成圖表
        if "series" in payload:
            charts["price"] = plot_price(
                payload["series"].get("dates"),
                payload["series"].get("prices"),
                title=payload["series"].get("title", "基金走勢")
            )
            charts["price_raw"] = plot_price_raw(
                payload["series"].get("dates"),
                payload["series"].get("prices"),
                title=payload["series"].get("title", "基金走勢")
            )

        # 渲染 Markdown 報告
        md = tpl.render(**payload, charts=charts)
        html = markdown.markdown(md, extensions=["tables"])
        pdf_bytes = md_to_pdf(md, charts)

        return {
            "markdown": md,
            "html": html,
            "pdf": base64.b64encode(pdf_bytes).decode()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"report generation failed: {e}")




