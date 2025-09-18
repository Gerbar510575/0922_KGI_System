from fastapi import FastAPI, HTTPException
from jinja2 import Environment, FileSystemLoader
import markdown, io, base64
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
import logging, traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("report-service")

app = FastAPI(title="Report Service")

# 載入 Jinja2 模板
env = Environment(loader=FileSystemLoader("templates"), autoescape=True)
tpl = env.get_template("report_zh.md.j2")

def md_to_pdf(md_text: str) -> bytes:
    """將 Markdown 文字轉成 PDF（純文字，不再插入圖表）"""
    styles = getSampleStyleSheet()
    story = []

    for line in md_text.splitlines():
        if not line.strip():
            story.append(Spacer(1, 12))
        elif line.startswith("## "):
            story.append(Paragraph(f"<b>{line[3:]}</b>", styles["Heading2"]))
        elif line.startswith("# "):
            story.append(Paragraph(f"<b>{line[2:]}</b>", styles["Heading1"]))
        else:
            story.append(Paragraph(line, styles["Normal"]))

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    doc.build(story)
    return buf.getvalue()

@app.post("/report")
def generate(payload: dict):
    logger.info(f"收到 /report 請求, keys={list(payload.keys())}")
    try:
        md = tpl.render(**payload)
        html = markdown.markdown(md, extensions=["tables"])
        pdf_bytes = md_to_pdf(md)

        return {
            "markdown": md,
            "html": html,
            "pdf": base64.b64encode(pdf_bytes).decode(),
        }
    except Exception as e:
        logger.error("報告生成失敗")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"report generation failed: {e}")









