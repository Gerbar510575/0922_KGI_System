import os
import io
import base64
import requests
import pandas as pd
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ---------------- 自動尋找中文字型 ----------------
try:
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    if not os.path.exists(font_path):
        font_path = "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"
    if os.path.exists(font_path):
        zh_font = fm.FontProperties(fname=font_path)
        matplotlib.rcParams["font.family"] = zh_font.get_name()
        print(f"✅ 使用中文字型: {zh_font.get_name()}")
    else:
        print("⚠️ 找不到 NotoSansCJK-Regular.ttc，仍使用預設字型")
except Exception as e:
    print(f"⚠️ 設定中文字型失敗: {e}")

# ---------------- 工具函式 ----------------
def save_fig_to_session(fig, key, close=True):
    if fig is None:
        return
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.getvalue()).decode()
    st.session_state.setdefault("charts", {})[key] = b64
    if close:
        plt.close(fig)

def save_external_image_to_session(data_url_or_b64, key):
    if not data_url_or_b64:
        return
    if isinstance(data_url_or_b64, str) and data_url_or_b64.startswith("data:image"):
        try:
            data_url_or_b64 = data_url_or_b64.split(",", 1)[1]
        except Exception:
            pass
    st.session_state.setdefault("charts", {})[key] = data_url_or_b64

# ---------------- 基本設定 ----------------
GATE = os.getenv("GATEWAY_URL", "http://localhost:8000")
CSV_PATH = os.getenv("TRAIN_CSV_PATH", "train.csv")

if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
else:
    df = pd.DataFrame({
        "Gender": ["Male", "Female"],
        "Married": ["Yes", "No"],
        "Dependents": ["0", "1", "2", "3+"],
        "Education": ["Graduate", "Not Graduate"],
        "Self_Employed": ["Yes", "No"],
        "Property_Area": ["Urban", "Rural", "Semiurban"],
    })

gender_opts = df["Gender"].dropna().unique().tolist()
married_opts = df["Married"].dropna().unique().tolist()
dependents_opts = df["Dependents"].dropna().unique().tolist()
education_opts = df["Education"].dropna().unique().tolist()
self_emp_opts = df["Self_Employed"].dropna().unique().tolist()
property_opts = df["Property_Area"].dropna().unique().tolist()

st.set_page_config(page_title="KGI 個人化 AI 投資建議系統", layout="wide")
st.title("KGI 個人化 AI 投資建議系統")

tab1, tab2, tab3 = st.tabs(["請填寫您的個人資料", "個人化 AI 建議基金", "知識檢索與報告"])

# --- Tab1 ---
with tab1:
    with st.form("kyc_form"):
        name = st.text_input("姓名", "GB")
        gender = st.selectbox("性別", gender_opts)
        married = st.selectbox("婚姻狀態", married_opts)
        dependents = st.selectbox("扶養人數", dependents_opts)
        education = st.selectbox("教育程度", education_opts)
        self_emp = st.selectbox("是否自僱", self_emp_opts)
        property_area = st.selectbox("房地產區域", property_opts)
        applicant_income = st.number_input("您的收入", min_value=0)
        coapplicant_income = st.number_input("您配偶的收入", min_value=0)
        submit = st.form_submit_button("取得個人化 AI 建議基金")

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
        }
        st.session_state["client"] = client
        payload = {"kyc": client}

        try:
            st.session_state["advice"] = requests.post(
                f"{GATE}/advise", json=payload, timeout=90
            ).json()
            st.session_state["charts"] = {}  # reset charts
            st.success("✅ 已生成建議，請切換到『個人化 AI 建議基金』")
        except Exception as e:
            st.warning(f"無法取得建議：{e}")

# --- Tab2: 基金建議 + 視覺化（含預測，並同時儲存圖表） ---
with tab2:
    if "advice" in st.session_state and isinstance(st.session_state["advice"], dict):
        st.subheader("投資建議基金")

        fund = st.session_state["advice"].get("selected_fund", {}) or {}
        advice = st.session_state["advice"]

        if fund:
            # ========== 數值型欄位視覺化 ==========
            st.write("### 基金關鍵數據")
            col1, col2, col3 = st.columns(3)

            # Beta
            if "beta" in fund:
                with col1:
                    st.metric(label="Beta", value=f"{fund['beta']:.2f}")
                    fig, ax = plt.subplots(figsize=(2.5, 2))
                    ax.bar(["Beta"], [fund["beta"]])
                    ax.set_ylim(0, max(2.0, fund["beta"] * 1.2))
                    ax.set_ylabel("值")
                    st.pyplot(fig)
                    save_fig_to_session(fig, "fund_beta_bar")

            # 費用率
            if "fee" in fund:
                with col2:
                    try:
                        fee_val = float(fund["fee"]) * 100
                        st.metric(label="費用率 (%)", value=f"{fee_val:.2f}")
                        fig, ax = plt.subplots(figsize=(2.5, 2))
                        ax.bar(["Fee"], [fee_val])
                        ax.set_ylim(0, max(2.0, fee_val * 1.5))
                        ax.set_ylabel("%")
                        st.pyplot(fig)
                        save_fig_to_session(fig, "fund_fee_bar")
                    except:
                        st.metric(label="費用率", value=str(fund["fee"]))

            # 規模 (AUM)
            if "aum" in fund:
                with col3:
                    try:
                        aum_val = float(fund["aum"])
                        st.metric(label="基金規模 (NTD百萬)", value=f"{aum_val:,.0f}")
                        fig, ax = plt.subplots(figsize=(2.5, 2))
                        ax.bar(["AUM"], [aum_val])
                        ax.set_ylabel("NTD 百萬")
                        st.pyplot(fig)
                        save_fig_to_session(fig, "fund_aum_bar")
                    except:
                        st.metric(label="基金規模", value=str(fund["aum"]))

            # 風險等級
            if "risk_level" in fund:
                st.write("### 風險等級")
                risk_map = {"RR1": 1, "RR2": 2, "RR3": 3, "RR4": 4, "RR5": 5}
                r_val = risk_map.get(fund["risk_level"], None)
                if r_val:
                    fig, ax = plt.subplots(figsize=(4, 1.5))
                    ax.barh(["Risk"], [r_val])
                    ax.set_xlim(0, 5)
                    ax.set_xlabel("風險等級 (RR1~RR5)")
                    st.pyplot(fig)
                    save_fig_to_session(fig, "fund_risk_barh")
                else:
                    st.text(f"風險等級: {fund['risk_level']}")

            # ========== 分佈型欄位視覺化 ==========
            if "Top ten holdings" in fund:
                st.write("### 前十大持股")
                try:
                    holdings = {}
                    for item in fund["Top ten holdings"].split(","):
                        k, v = item.strip().rsplit(" ", 1)
                        holdings[k] = float(v.replace("%", ""))
                    h_df = pd.DataFrame(list(holdings.items()), columns=["Holding", "Weight"])
                    fig, ax = plt.subplots(figsize=(6, 3))
                    ax.barh(h_df["Holding"], h_df["Weight"])
                    ax.set_xlabel("比重 (%)")
                    st.pyplot(fig)
                    save_fig_to_session(fig, "holdings_top10_barh")
                except:
                    st.text(fund["Top ten holdings"])

            if "Industry allocation" in fund:
                st.write("### 產業配置")
                try:
                    inds = {}
                    for item in fund["Industry allocation"].split(","):
                        k, v = item.strip().rsplit(" ", 1)
                        inds[k] = float(v.replace("%", ""))
                    i_df = pd.DataFrame(list(inds.items()), columns=["Industry", "Weight"])
                    fig, ax = plt.subplots(figsize=(6, 3))
                    ax.pie(i_df["Weight"], labels=i_df["Industry"], autopct="%.1f%%")
                    st.pyplot(fig)
                    save_fig_to_session(fig, "industry_allocation_pie")
                except:
                    st.text(fund["Industry allocation"])

            if "Country allocation" in fund:
                st.write("### 國家配置")
                try:
                    countries = {}
                    for item in fund["Country allocation"].split(","):
                        k, v = item.strip().rsplit(" ", 1)
                        countries[k] = float(v.replace("%", ""))
                    c_df = pd.DataFrame(list(countries.items()), columns=["Country", "Weight"])
                    fig, ax = plt.subplots(figsize=(6, 3))
                    ax.bar(c_df["Country"], c_df["Weight"])
                    ax.set_ylabel("比重 (%)")
                    st.pyplot(fig)
                    save_fig_to_session(fig, "country_allocation_bar")
                except:
                    st.text(fund["Country allocation"])

            # ========== 預測（基金 & 個股）回到 Tab2，畫完圖同時存入 charts ==========
            # 基金價格預測 (用 P5/Median/P95 的 bar 圖)
            fc = advice.get("fund_forecast", {}) or {}
            if fc and "price_scenarios" in fc:
                st.write("### 基金價格預測 (10日)")
                vals = [
                    fc["price_scenarios"].get("P5_10d"),
                    fc["price_scenarios"].get("Median_10d"),
                    fc["price_scenarios"].get("P95_10d"),
                ]
                labels = ["P5", "Median", "P95"]
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.bar(labels, vals)
                ax.set_ylabel("Price")
                st.pyplot(fig)
                save_fig_to_session(fig, "fund_forecast_bar")

                # 若後端已有預測圖（base64 或 data URL），一併存起來
                if fc.get("forecast_plot"):
                    save_external_image_to_session(fc["forecast_plot"], "fund_forecast_plot")

            # 基金持股之個股價格預測（折線：每檔畫 P5/Median/P95 三點）
            stock_fcs = advice.get("stock_forecasts", {}) or {}
            if stock_fcs:
                st.write("### 基金持股之個股價格預測 (10日)")
                # 總覽圖
                fig, ax = plt.subplots(figsize=(6, 4))
                for t, sfc in stock_fcs.items():
                    points_x = ["P5", "Median", "P95"]
                    points_y = [sfc.get("P5_10d"), sfc.get("Median_10d"), sfc.get("P95_10d")]
                    ax.plot(points_x, points_y, marker="o", label=t)
                ax.set_ylabel("Price")
                ax.legend(fontsize=8, ncols=2)
                st.pyplot(fig)
                save_fig_to_session(fig, "stock_forecasts_overview")

                # 若各檔股票有後端提供的圖，也存起來
                for t, sfc in stock_fcs.items():
                    if sfc.get("forecast_plot"):
                        save_external_image_to_session(sfc["forecast_plot"], f"stock_forecast__{t}")

            # ========== 其他文字欄位 ==========
            st.write("### 基本文字資訊")
            text_info = {
                "名稱": fund.get("name", "未知"),
                "代碼": fund.get("code", "未知"),
                "成立日期": fund.get("inception", "未知"),
                "分類": fund.get("category", "未知"),
                "幣別": fund.get("currency", "未知"),
                "基金經理人": fund.get("manager", "未知"),
            }
            st.table(pd.DataFrame([text_info]))

        # 詢問基金相關問題（RAG：限制 10 秒）
        st.write("### 詢問基金相關問題")
        q = st.text_input("請輸入問題（例如：基金風險、投資標的...）")
        doc_type = st.selectbox("限定文件類型（可選）", ["不限", "月報", "公開說明書"], index=0)
        if st.button("🔎 發送問題"):
            if q.strip():
                selected_fund_code = fund.get("code") or fund.get("fund_code")
                payload = {"query": q}
                if selected_fund_code:
                    payload["fund_code"] = selected_fund_code
                if doc_type != "不限":
                    payload["doc_type"] = doc_type
                try:
                    rag_resp = requests.post(
                        f"{GATE}/query", json=payload, timeout=100  # 嚴控 10 秒
                    ).json()
                    st.session_state["rag"] = rag_resp
                    st.success("✅ 已取得知識檢索結果，請切換到『知識檢索與報告』")
                except Exception as e:
                    st.warning(f"RAG 逾時或失敗：{e}")
            else:
                st.warning("請輸入您的問題！")

# --- Tab3 ---
with tab3:
    if "client" in st.session_state and "advice" in st.session_state:
        payload = {
            "client": st.session_state["client"],
            "advice": st.session_state["advice"],
            "refs": st.session_state.get("rag", {}),
            # ❌ 不再把 charts 傳給 report，避免超時
        }
        try:
            resp = requests.post(f"{GATE}/report", json=payload, timeout=60).json()

            if resp.get("markdown"):
                st.subheader("📄 投資建議分析報告（Markdown）")
                st.markdown(resp["markdown"])
            if resp.get("html"):
                with st.expander("🔍 查看 HTML 版本"):
                    st.components.v1.html(resp["html"], height=600, scrolling=True)
            if resp.get("pdf"):
                pdf_bytes = base64.b64decode(resp["pdf"])
                st.download_button(
                    label="📥 下載 PDF 報告",
                    data=pdf_bytes,
                    file_name="report.pdf",
                    mime="application/pdf",
                )

            # 🚀 新增：直接顯示 Tab2 存下來的圖
            if "charts" in st.session_state:
                st.subheader("📊 圖片回顧")
                for key, b64 in st.session_state["charts"].items():
                    try:
                        st.image(base64.b64decode(b64), caption=key)
                    except Exception:
                        st.text(f"(無法顯示 {key})")

        except Exception as e:
            st.error(f"⚠️ 無法生成報告：{e}")










