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
st.set_page_config(page_title="凱基客製化 AI 投資建議系統", layout="wide", initial_sidebar_state="collapsed")

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
        height: 100%;
        margin: 0;
        padding: 0;
    }
    .main .block-container {
        max-width: 100%;
        padding-top: 0rem;
        padding-right: 0rem;
        padding-left: 0rem;
        padding-bottom: 0rem;
    }      
    .big-title {
        color: #004B97;
        font-size: 32px;
        font-weight: bold;
    }
    .section-header {
        color: #0070C0;
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
    /* Steps bar styling */
    .steps-wrap{
        display:flex;
        justify-content:center;
        align-items:center;
        gap:10px;
        margin:8px 0 14px 0;
        flex-wrap:wrap;
    }
    .step-item{
        display:flex;
        align-items:center;
        gap:8px;
        padding:8px 12px;
        border-radius:999px;
        border:1px solid #d9e6fb;
        background:#f5f9ff;
        color:#034694;
        font-weight:600;
        font-size:14px;
    }
    .step-item.active{
        background:#003366;   /* 深藍底 */
        border-color:#001a33; /* 深藍邊框 */
        color:#ffffff;        /* 白字 */
        font-weight:700;      /* 更粗體 */
        box-shadow:0 0 6px rgba(0,0,0,0.3);
    }
    .step-arrow{
        font-size:16px;
        color:#7aa7ff;
    }
    </style>
""", unsafe_allow_html=True)


# ---- 小工具：渲染三步驟示意圖 (active ∈ {"risk","filter","rag"}) ----
def render_steps(active: str = "risk"):
    classes = {
        "risk": "step-item active" if active == "risk" else "step-item",
        "filter": "step-item active" if active == "filter" else "step-item",
        "rag": "step-item active" if active == "rag" else "step-item",
    }
    html = f"""
    <div class="steps-wrap">
        <div class="{classes['risk']}">客戶風險承受度分群</div>
        <div class="step-arrow">➜</div>
        <div class="{classes['filter']}">基金推薦篩選</div>
        <div class="step-arrow">➜</div>
        <div class="{classes['rag']}">RAG 問答</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

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

# ---------------- 專案目錄樹顯示 ----------------
TREE_PATH = "/app/data/project_tree.txt"  # 放在專案根目錄的 /data 下
tree_text = None
if os.path.exists(TREE_PATH):
    with open(TREE_PATH, "r", encoding="utf-8") as f:
        tree_text = f.read()

# ---------------- 專案摘要 (列點方式) ----------------
project_summary = [
    "📂 **apps/** → 核心微服務 (advisor, gateway, market, ml-bridge, rag, report, ui)",
    "⚙️ **configs/** → 環境變數與 RAG 設定檔",
    "📊 **data/** → 專案資料 (基金 JSON、metadata)",
    "🛠️ **infra/** → 基礎建設 (docker-compose, Dockerfile)",
    "🚀 **scripts/** → 啟動與初始化腳本 (bootstrap_demo.sh, run_all.sh)",
    "💾 **chroma_db/**、**hf_cache/** → 執行快取，可重建",
    "📄 **.env / README.md / Makefile** → 專案配置與說明"
]

# ---------------- 載入訓練資料 ----------------
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
st.markdown('<div class="big-title">凱基客製化 AI 投資建議系統</div>', unsafe_allow_html=True)

# 顯示專案摘要
#st.markdown('<div class="section-header">📂 專案目錄摘要</div>', unsafe_allow_html=True)
#summary_text = "\n".join([f"- {item}" for item in project_summary])
#st.info(summary_text)

def render_mermaid(mermaid_code: str, height: int = 600):
    html = f"""
    <div>
      <div class="mermaid">
{mermaid_code}
      </div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <script>
      mermaid.initialize({{ startOnLoad: true, theme: "neutral" }});
    </script>
    """
    st.components.v1.html(html, height=height, scrolling=True)

mermaid_code = r"""
flowchart TB
  advisor["advisor"]
  gateway["gateway"]
  market["market"]
  ml_bridge["ml-bridge"]
  rag["rag"]
  report["report"]
  ui["ui"]

  advisor -->|"http://market:8005"| market
  advisor -->|"http://ml-bridge:7000"| ml_bridge

  gateway -->|"http://market:8005"| market
  gateway -->|"http://rag:8002"| rag
  gateway -->|"http://advisor:8003"| advisor
  gateway -->|"http://report:8004"| report

  ui -->|"http://localhost:8000"| gateway

  %% 樣式定義
  classDef microservice fill:#e6f2ff,stroke:#004b97,stroke-width:2px,font-weight:bold;

  %% 套用樣式
  class advisor,gateway,market,ml_bridge,rag,report,ui microservice;
"""

st.markdown("### 凱基客製化 AI 投資建議系統架構圖")
render_mermaid(mermaid_code, height=600)

# === 客戶風險承受度分群區塊 ===
st.markdown(
    """
    <style>
    @media (prefers-color-scheme: dark) {
        .custom-card {
            background-color: #1E2A38 !important; /* 深藍灰底 */
            border: 2px solid #3399FF !important; /* 亮藍邊框 */
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
        }
    }
    @media (prefers-color-scheme: light) {
        .custom-card {
            background-color: #F8FBFF !important; /* 淺藍底 */
            border: 2px solid #004B97 !important; /* 深藍邊框 */
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
        }
    }
    </style>
    <div class="custom-card">
    """,
    unsafe_allow_html=True,
)


# === 在專案摘要與「模擬客戶資料」之間插入步驟圖：強調【客戶風險承受度分群】 ===
render_steps(active="risk")

# 顯示完整目錄樹 (若檔案存在)
# if tree_text:
#     st.markdown('<div class="section-header">📂 專案目錄</div>', unsafe_allow_html=True)
#     st.code(tree_text, language="text")

# ===== (1) 客戶資料 =====
st.markdown('<div class="section-header">🧑‍💼 Kaggle 平台上銀行借貸資料集</div>', unsafe_allow_html=True)

with st.form("kyc_form"):
    name = st.text_input("姓名", "江宏繹")
    gender = st.selectbox("性別", gender_opts)
    married = st.selectbox("婚姻狀態", married_opts)
    dependents = st.selectbox("扶養人數", dependents_opts)
    education = st.selectbox("教育程度", education_opts)
    self_emp = st.selectbox("是否自僱", self_emp_opts)
    property_area = st.selectbox("房地產區域", property_opts)

    st.markdown("<br>", unsafe_allow_html=True)  # 增加間距
    applicant_income = st.number_input("您的收入", min_value=0, value=2000)
    coapplicant_income = st.number_input("您配偶的收入", min_value=0, value=1000)
    loan_amount = st.number_input("貸款金額", min_value=0, value=100)
    loan_amount_term = st.number_input("貸款期限 (天)", min_value=0, value=360)
    credit_history = st.number_input("信用紀錄 (0=無, 1=有)", min_value=0, max_value=1, value=1)                                   
    #product_pref = st.selectbox("偏好金融商品", product_opts)

    st.markdown('<div class="section-header">📊 分類模型結果</div>', unsafe_allow_html=True)

    # 顯示圖片
    st.image("/app/data/model_compare.png", caption="三種模型效能比較", use_column_width=True)

    st.markdown('<div class="section-header">🎯 偏好金融商品 </div>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:22px; color:#0070C0; font-weight:bold;">請選擇您有偏好的金融商品：</p>', unsafe_allow_html=True)
    st.selectbox("金融商品", product_opts, key="product_pref_big")
    #st.selectbox(product_opts, key="product_pref_big")

    submit = st.form_submit_button("🚀 取得系統推薦的一檔基金")

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
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_amount_term,
        "Credit_History": credit_history,
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
    risk_score = debug.get("risk_score", 0.8)
    #st.json(debug)   # 直接輸出 debug_info 看真實內容
    st.metric("客戶風險屬性", risk_type, f"{risk_score:.2f}")
    #st.metric("客戶風險屬性", risk_type)

# (4) 關閉卡片容器
st.markdown("</div>", unsafe_allow_html=True)

#st.divider()

st.markdown(
    """
    <style>
    @media (prefers-color-scheme: dark) {
        .custom-card {
            background-color: #1E2A38 !important; /* 深藍灰底 */
            border: 2px solid #3399FF !important; /* 亮藍邊框 */
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
        }
    }
    @media (prefers-color-scheme: light) {
        .custom-card {
            background-color: #F8FBFF !important; /* 淺藍底 */
            border: 2px solid #004B97 !important; /* 深藍邊框 */
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
        }
    }
    </style>
    <div class="custom-card">
    """,
    unsafe_allow_html=True,
)

# === 在 divider 之下插入步驟圖：強調【基金推薦篩選】 ===
render_steps(active="filter")

# ===== (2) 基金推薦流程 =====
if "advice" in st.session_state and isinstance(st.session_state["advice"], dict):
    advice = st.session_state["advice"]
    fund = advice.get("selected_fund", {}) or {}
    debug = advice.get("debug_info", {})

    # ===== CSS 美化 =====
    st.markdown(
        """
        <style>
        table {
            margin-left: auto;
            margin-right: auto;
            border-collapse: collapse;
            font-size: 14px;  /* 加大字體 */
        }
        th {
            text-align: center !important;
            font-weight: bold;
            padding: 8px 14px;  /* 增加間距 */
            border-bottom: 1px solid #ddd;
        }
        td {
            text-align: center !important;
            padding: 8px 14px;  /* 增加間距 */
            border-bottom: 1px solid #f0f0f0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    # (1) 初始基金池 (Beta 計算)
    st.markdown("##### 篩選邏輯 ① : 基金 Beta 值計算 (簡報用，不於客戶端呈現)")
    st.markdown("<br>", unsafe_allow_html=True)

    all_betas = debug.get("all_fund_betas", {})
    if all_betas:
        rows = []
        for fc, b in all_betas.items():
            name = fund_name_map_full.get(fc, fc)
            beta_str = f"{b:.4f}"
            if b > 1:
                beta_str = f"<span style='color:red; font-weight:bold'>{beta_str}</span>"
            else:
                beta_str = f"<span style='color:green; font-weight:bold'>{beta_str}</span>"
            rows.append(f"| {name} | {beta_str} |")

        md_all = "| 基金名稱 | Beta |\n|---|---|\n" + "\n".join(rows)
        st.markdown(md_all, unsafe_allow_html=True)

        # 公式（依 advisor-service 的 CAPM 定義）
        #st.latex(r"\beta_{\text{基金}} = \sum_{i \in 持股} w_i \cdot \beta_i \quad , \quad \beta_i = \frac{\text{Cov}(R_i, R_m)}{\text{Var}(R_m)}")

        # 說明
        st.info("""
        **基金 Beta 值計算方式**  
        - 先抓取基金的前十大持股標的 (股票) 近 120 個交易日的收盤價。
        - 結合 **CAPM** 模型計算各股票的 Beta 值。  
        - 再依基金持股比例，將股票 Beta 值加權平均，得到基金層級的 Beta 值。  
        - **Beta 值 > 1** ：該檔基金波動大於基金池，屬於高風險，適合推薦給高風險承受度客戶。  
        - **Beta 值 < 1** ：該檔基金波動低於基金池，屬於低風險，適合推薦給低風險承受度客戶。
        """)

        st.markdown("<br><br>", unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(1.5, 0.5))  # 縮小圖表
        labels = [fund_name_map_short.get(fc, fc) for fc in all_betas.keys()]
        ax.bar(labels, list(all_betas.values()))
        ax.axhline(1.0, color="red", linestyle="--", label="基準 Beta=1")
        ax.set_ylabel("Beta", fontsize=5)
        ax.tick_params(axis="x", labelsize=5)
        ax.tick_params(axis="y", labelsize=5)
        #ax.legend(fontsize=4, loc="upper right")
        st.pyplot(fig)

        st.markdown("<br><br><hr><br>", unsafe_allow_html=True)

    # (2) 市場熱度
    st.markdown("##### 篩選邏輯 ② : 基金熱度分數計算 (簡報用，不於客戶端呈現)")
    st.markdown("<br>", unsafe_allow_html=True)

    # 公式（依 market-service 定義）
    st.latex(r"\text{基金熱度} = \sum_{i \in 持股} w_i \cdot \frac{成交量_{i,t} - \text{均量}_{i,30日}}{\text{標準差}_{i,30日}}")

    # 說明
    st.info("""
    **基金熱度分數計算方式**  
    - 先計算個股最近一日成交量相對於 30 日均量與波動的偏離程度。  
    - 再依基金持股比例對各股票的分數加權平均，得到基金的整體熱度。  
    - 分數越高代表近期市場交易活絡，顯示市場關注度較高。  
    """)

    st.markdown("<br><br>", unsafe_allow_html=True)

    heat_data = debug.get("fund_heat_data", {})
    if heat_data:
        max_score = max(d.get("rel_volume_score", 0.0) for d in heat_data.values())
        rows = []
        for fc, d in heat_data.items():
            name = fund_name_map_full.get(fc, fc)
            score = d.get("rel_volume_score", 0.0)
            score_str = f"{score:.4f}"

            if score == max_score:
                score_str = f"<span style='color:purple; font-weight:bold'>{score_str}</span>"
            elif score > 0:
                score_str = f"<span style='color:orange'>{score_str}</span>"
            else:
                score_str = f"<span style='color:blue'>{score_str}</span>"

            rows.append(f"| {name} | {score_str} |")

        md_heat = "| 基金名稱 | 熱度分數 |\n|---|---|\n" + "\n".join(rows)
        st.markdown(md_heat, unsafe_allow_html=True)
        st.markdown("<br><br><hr><br>", unsafe_allow_html=True)

    # (3) 最終推薦基金
    st.markdown('<div class="section-header">🎯 系統推薦基金 </div>', unsafe_allow_html=True)
    if fund:
        text_info = {
            "基金名稱": fund.get("name", "未知"),
            "成立日期": fund.get("inception", "未知"),
            "基金經理人": fund.get("manager", "未知"),
            "基金規模 (百萬臺幣)": fund.get("aum (NTD million)", "未知"), 
            "幣別": fund.get("currency", "未知"),
            "風險等級": fund.get("risk_level", "未知"),
        }

        # 格式化 Beta（如果後續加入 Beta，可在此處處理）
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

# 關閉卡片容器
st.markdown("</div>", unsafe_allow_html=True)
    
st.markdown(
    """
    <style>
    @media (prefers-color-scheme: dark) {
        .custom-card {
            background-color: #1E2A38 !important; /* 深藍灰底 */
            border: 2px solid #3399FF !important; /* 亮藍邊框 */
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
        }
    }
    @media (prefers-color-scheme: light) {
        .custom-card {
            background-color: #F8FBFF !important; /* 淺藍底 */
            border: 2px solid #004B97 !important; /* 深藍邊框 */
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
        }
    }
    </style>
    <div class="custom-card">
    """,
    unsafe_allow_html=True,
)


render_steps(active="rag")

# (4) 投資報告（含 RAG 問答）
st.markdown('<div class="section-header">🎯 RAG 問答 </div>', unsafe_allow_html=True)

q = st.text_input("請輸入關於系統推薦基金的問題（例如：基金前十大持股?）")

if st.button("📑 產生投資建議報告"):
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
            st.markdown("### 📑 客製化投資建議分析報告")
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

# 關閉卡片容器
st.markdown("</div>", unsafe_allow_html=True)























