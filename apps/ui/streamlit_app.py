import os
import io
import base64
import requests
import pandas as pd
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import json

# ---------------- 頁面設定 (必須最先呼叫一次) ----------------
st.set_page_config(page_title="KGI 個人化 AI 投資建議系統", layout="wide")

# ---------------- 自動尋找中文字型 ----------------
try:
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    if not os.path.exists(font_path):
        font_path = "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"
    if os.path.exists(font_path):
        zh_font = fm.FontProperties(fname=font_path)
        matplotlib.rcParams["font.family"] = zh_font.get_name()
except Exception:
    pass

# ---------------- Streamlit Theme Style ----------------
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-family: 'Noto Sans TC', 'Microsoft JhengHei', sans-serif;
    }
    .big-title {
        color: #004B97;
        font-size: 32px;
        font-weight: bold;
    }
    .section-header {
        color: #004B97;
        font-size: 24px;
        font-weight: 600;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .sub-header {
        color: #0070C0;
        font-size: 20px;
        font-weight: 500;
        margin-top: 15px;
        margin-bottom: 5px;
    }
    thead tr th {
        background-color: #004B97;
        color: white !important;
    }
    .q-card {
        background-color: #F0F0F0;
        border-radius: 10px;
        padding: 12px;
        margin-top: 10px;
        margin-bottom: 5px;
        color: #000000;
        font-weight: 500;
    }
    .rag-card {
        background-color: #F8FBFF;
        border: 2px solid #004B97;
        border-radius: 10px;
        padding: 15px;
        margin-top: 5px;
        margin-bottom: 20px;
        color: #000000;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- 工具函式 ----------------
def save_fig_to_session(fig, key, close=True):
    if fig is None:
        return
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.getvalue()).decode()
    st.session_state.setdefault("charts", {})[key] = b64
    if close:
        plt.close(fig)

def color_perf(val):
    try:
        v = float(str(val).replace("%",""))
        if v > 0: return "color: green"
        if v < 0: return "color: red"
    except:
        return ""
    return ""

# ---------------- 基本設定 ----------------
GATE = os.getenv("GATEWAY_URL", "http://localhost:8000")
CSV_PATH = os.getenv("TRAIN_CSV_PATH", "train.csv")
JSON_PATH = os.getenv("TRAIN_JSON_PATH", "/app/data/metadata_all_chunked_para.json")

if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)

# ---------------- Normalize 工具 ----------------
def normalize_perf_keys(perf_dict: dict) -> dict:
    if not perf_dict:
        return {}
    mapping = {
        "新台幣A": "新台幣A", "新台幣A級": "新台幣A", "新臺幣A": "新台幣A", "新臺幣A級": "新台幣A",
        "美元A": "美元A", "美元A級": "美元A",
        "人民幣A": "人民幣A", "人民幣A級": "人民幣A",
        "南非幣A": "南非幣A", "南非幣A級": "南非幣A"
    }
    normalized = {}
    for k, v in perf_dict.items():
        key = mapping.get(k, k)
        normalized[key] = v
    return normalized

# ---------------- 載入基金績效 JSON ----------------
fund_perf_data = {}
if os.path.exists(JSON_PATH):
    try:
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            json_raw = json.load(f)
        for entry in json_raw:
            meta = entry.get("metadata", {})
            if meta.get("doc_type") == "月報":
                code = (meta.get("fund_code") or "").strip()
                perf = meta.get("performance") or entry.get("performance")
                if code and perf:
                    fund_perf_data[code] = normalize_perf_keys(perf)
    except Exception as e:
        st.warning(f"⚠️ 無法讀取績效 JSON: {e}")

# ---------------- 下拉選單選項 ----------------
gender_opts = df["Gender"].dropna().unique().tolist()
married_opts = df["Married"].dropna().unique().tolist()
dependents_opts = df["Dependents"].dropna().unique().tolist()
education_opts = df["Education"].dropna().unique().tolist()
self_emp_opts = df["Self_Employed"].dropna().unique().tolist()
property_opts = df["Property_Area"].dropna().unique().tolist()
product_opts = ["美國股票型基金", "非投資等級債券型基金", "海外債券型基金", "保險商品"]

fund_name_map_full = {
    "G006": "凱基雲端趨勢基金",
    "G011": "凱基醫院及長照產業基金",
    "G012": "凱基環球趨勢基金",
    "G013": "凱基未來移動基金",
}
fund_name_map_short = {
    "G006": "雲端趨勢",
    "G011": "醫院長照",
    "G012": "環球趨勢",
    "G013": "未來移動",
}

# ---------------- UI ----------------
st.markdown('<div class="big-title">💡 KGI 個人化 AI 投資建議系統</div>', unsafe_allow_html=True)

# ===== (1) 客戶資料 =====
st.markdown('<div class="section-header">👤 客戶風險承受度分群</div>', unsafe_allow_html=True)
with st.form("kyc_form"):
    name = st.text_input("姓名", "江宏繹")
    gender = st.selectbox("性別", gender_opts)
    married = st.selectbox("婚姻狀態", married_opts)
    dependents = st.selectbox("扶養人數", dependents_opts)
    education = st.selectbox("教育程度", education_opts)
    self_emp = st.selectbox("是否自僱", self_emp_opts)
    property_area = st.selectbox("房地產區域", property_opts)
    applicant_income = st.number_input("您的收入", min_value=0, value=2000)
    coapplicant_income = st.number_input("您配偶的收入", min_value=0, value=1000)
    #product_pref = st.selectbox("偏好金融商品", product_opts)

    st.markdown('<div class="section-header">🎯 偏好金融商品</div>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:22px; color:#0070C0; font-weight:bold;">請選擇您有興趣的金融商品：</p>', unsafe_allow_html=True)
    st.selectbox("偏好金融商品", product_opts, key="product_pref_big")

    submit = st.form_submit_button("🚀 取得個人化 AI 建議基金")

if submit:
    client = {
        "name": name,
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_emp,
        "Property_Area": property_area,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        #"ProductPref": product_pref,
    }
    st.session_state["client"] = client
    payload = {"kyc": client}
    try:
        st.session_state["advice"] = requests.post(
            f"{GATE}/advise", json=payload, timeout=90
        ).json()
        st.session_state["charts"] = {}
        #st.success("✅ 已生成建議，請往下查看結果")
    except Exception as e:
        st.warning(f"無法取得建議：{e}")

if "advice" in st.session_state and isinstance(st.session_state["advice"], dict):
    advice = st.session_state["advice"]
    debug = advice.get("debug_info", {})
    risk_type = debug.get("risk_type", "未知")
    risk_score = debug.get("risk_score", 0.5)
    st.metric("客戶風險屬性", risk_type, f"{risk_score:.2f}")

st.divider()

# ===== (2) 基金推薦流程 =====


st.markdown('<div class="section-header">📊 個人化 AI 建議基金篩選過程說明</div>', unsafe_allow_html=True)

if "advice" in st.session_state and isinstance(st.session_state["advice"], dict):
    advice = st.session_state["advice"]
    fund = advice.get("selected_fund", {}) or {}
    debug = advice.get("debug_info", {})

    # (1) 初始基金池 (Beta 計算)
    st.markdown("##### ① 初始基金池 (Beta 計算)")
    all_betas = debug.get("all_fund_betas", {})
    if all_betas:
        rows = []
        for fc, b in all_betas.items():
            name = fund_name_map_full.get(fc, fc)
            beta_str = f"{b:.4f}"
            if b > 1:
                beta_str = f"<span style='color:red'>{beta_str} </span>"
            else:
                beta_str = f"<span style='color:green'>{beta_str}</span>"
            rows.append(f"| {name} | {beta_str} |")

        md_all = "| 基金名稱 | Beta |\n|---|---|\n" + "\n".join(rows)
        st.markdown(md_all, unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(3,1))
        labels = [fund_name_map_short.get(fc, fc) for fc in all_betas.keys()]
        ax.bar(labels, list(all_betas.values()))
        ax.axhline(1.0, color="red", linestyle="--", label="基準 Beta=1")
        ax.set_ylabel("Beta", fontsize=5)
        ax.tick_params(axis="x", labelsize=5)
        ax.tick_params(axis="y", labelsize=5)
        ax.legend(fontsize=5, loc="upper right")
        st.pyplot(fig)

    # (2) 市場熱度
    st.markdown("##### ② 市場熱度篩選")
    st.latex(r"熱度分數 = \frac{成交量_{當下} - 成交量平均值_{過去 30 日}}{成交量標準差_{過去 30 日}}")
    heat_data = debug.get("fund_heat_data", {})
    if heat_data:
        rows = []
        for fc, d in heat_data.items():
            name = fund_name_map_full.get(fc, fc)
            score = d.get("rel_volume_score", 0.0)
            score_str = f"{score:.4f}"
            if score > 0:
                score_str = f"<span style='color:orange'>{score_str} </span>"
            else:
                score_str = f"<span style='color:blue'>{score_str}</span>"
            rows.append(f"| {name} | {score_str} |")

        md_heat = "| 基金名稱 | 熱度分數 |\n|---|---|\n" + "\n".join(rows)
        st.markdown(md_heat, unsafe_allow_html=True)

    # (3) 最終推薦基金
    st.markdown("##### ③ 最終推薦基金")
    if fund:
        text_info = {
            "基金名稱": fund.get("name", "未知"),
            "成立日期": fund.get("inception", "未知"),
            "基金經理人": fund.get("manager", "未知"),
            "基金規模 (百萬臺幣)": fund.get("aum (NTD million)", "未知"), 
            "幣別": fund.get("currency", "未知"),
            "風險等級": fund.get("risk_level", "未知"),
        }

        # 格式化 Beta
        try:
            beta_val = float(text_info["Beta"])
            beta_str = f"{beta_val:.4f}"
            if beta_val > 1:
                text_info["Beta"] = f"<span style='color:red'>{beta_str} ⚡</span>"
            else:
                text_info["Beta"] = f"<span style='color:green'>{beta_str}</span>"
        except:
            pass

        rows = [f"| {k} | {v} |" for k, v in text_info.items()]
        md_final = "| 欄位 | 值 |\n|---|---|\n" + "\n".join(rows)
        st.markdown(md_final, unsafe_allow_html=True)


    # (4) 投資報告（含 RAG 問答）
    st.markdown("##### ④ 投資建議報告")

    q = st.text_input("請輸入關於系統推薦基金的問題（例如：說服我投資...）")

    if st.button("📑 產生投資報告"):
        if q.strip():
            try:
                # Step 1: 呼叫 RAG 拿回答
                selected_fund_code = fund.get("code") or fund.get("fund_code")
                payload_q = {"query": q}
                if selected_fund_code:
                    payload_q["fund_code"] = selected_fund_code

                rag_resp = requests.post(
                    f"{GATE}/query", json=payload_q, timeout=100
                ).json()
                st.session_state["rag"] = rag_resp
                st.session_state["last_question"] = q

                # Step 1.5: 整理過往績效，轉成模板可用的 list of dict
                code = fund.get("code") or fund.get("fund_code", "")
                base_code = code.split("_")[0] if code else ""
                perf = fund_perf_data.get(base_code, {})

                performance_rows = []
                if perf:
                    col_order = ["3個月", "1年", "3年", "5年", "成立以來"]
                    for currency, vals in perf.items():
                        row = {"currency": currency}
                        for c in col_order:
                            val = vals.get(c, None)
                            if val is not None:
                                try:
                                    row[c] = f"{float(val):.2f}"
                                except:
                                    row[c] = val
                            else:
                                row[c] = "-"
                        performance_rows.append(row)

                # Step 1.6: 把過往績效與預測圖加到 advice
                advice_data = st.session_state.get("advice", {})
                advice_data["performance"] = performance_rows

                fc = advice_data.get("fund_forecast", {})
                if fc and fc.get("forecast_plot"):
                    # 確保模板能拿到 base64 預測圖
                    advice_data.setdefault("fund_forecast", {})["forecast_plot"] = fc["forecast_plot"]

                # Step 2: 呼叫 Report API
                payload_r = {
                    "client": st.session_state.get("client", {}),
                    "advice": advice_data,
                    "refs": {
                        "query": q,       # 加上使用者輸入的問題
                        **rag_resp        # 再合併 RAG 回答 (answer 等欄位)
                    },
                }
                resp = requests.post(f"{GATE}/report", json=payload_r, timeout=180).json()
                st.session_state["final_report"] = resp

                # Step 3: 直接把報告內容顯示在前端
                st.markdown("### 📑 個人化投資建議分析報告")
                st.markdown("---")
                # 先把 Markdown 轉成 HTML，再加樣式
                html_report = resp.get("html")  # 後端已用 markdown 套件轉好的 HTML
                if html_report:
                    styled_html = f"""
                    <div style='background-color:#f9f9f9; color:#333; padding:20px; border-radius:10px; line-height:1.6;'>
                        <style>
                            h1 {{
                                color: #1f77b4;
                                font-size: 22px;
                                margin-top: 0.2rem;
                                margin-bottom: 0.6rem;
                            }}
                            h2 {{
                                color: #2ca02c;
                                font-size: 18px;
                                margin-top: 0.2rem;
                                margin-bottom: 0.6rem;
                            }}
                            table {{
                                border-collapse: collapse;
                                width: 100%;
                                margin: 8px 0 12px 0;
                            }}
                            th, td {{
                                border: 1px solid #ddd;
                                padding: 6px 10px;
                                text-align: center;
                            }}
                            th {{
                                background-color: #f0f0f0;
                            }}
                            p {{
                                margin: 0.2rem 0;
                            }}
                        </style>
                        {html_report}
                    </div>
                    """
                    st.components.v1.html(styled_html, height=700, scrolling=True)
                else:
                    # 後援：若後端沒回 html，就直接顯示 markdown（無自訂樣式）
                    st.markdown(resp.get("markdown", "（無報告內容）"))
            except Exception as e:
                st.error(f"❌ 報告生成失敗: {e}")
        else:
            st.warning("請輸入您的問題！")





















