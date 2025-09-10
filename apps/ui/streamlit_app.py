import streamlit as st, pandas as pd, requests, os, base64
import matplotlib.pyplot as plt
import seaborn as sns

GATE = os.getenv("GATEWAY_URL", "http://localhost:8000")
CSV_PATH = os.getenv("TRAIN_CSV_PATH", "train.csv")

df = pd.read_csv(CSV_PATH)

# 下拉選單選項
gender_opts = df["Gender"].dropna().unique().tolist()
married_opts = df["Married"].dropna().unique().tolist()
dependents_opts = df["Dependents"].dropna().unique().tolist()
education_opts = df["Education"].dropna().unique().tolist()
self_emp_opts = df["Self_Employed"].dropna().unique().tolist()
property_opts = df["Property_Area"].dropna().unique().tolist()

st.set_page_config(page_title="KGI 個人化投資建議", layout="wide")
st.title("AI 個人化投資建議系統")

tab1, tab2, tab3 = st.tabs(["請填寫您的資料", "建議基金", "知識檢索與報告"])

# --- Tab1: KYC 表單 ---
with tab1:
    with st.form("kyc_form"):
        name = st.text_input("姓名", "GB")
        gender = st.selectbox("性別", gender_opts)
        married = st.selectbox("婚姻狀態", married_opts)
        dependents = st.selectbox("扶養人數", dependents_opts)
        education = st.selectbox("教育", education_opts)
        self_emp = st.selectbox("是否自僱", self_emp_opts)
        property_area = st.selectbox("房地產區域", property_opts)
        applicant_income = st.number_input("您的收入", min_value=0)
        coapplicant_income = st.number_input("您配偶的收入", min_value=0)

        # === 新增 Beta 偏好輸入 ===
        #beta_min = st.number_input("當市場資產價格上漲/下跌 1 %，您希望您的資產價格「最低」上漲/下跌幅度(%)?", value=0.8, step=0.1)
        #beta_max = st.number_input("當市場資產價格上漲/下跌 1 %，您希望您的資產價格「最多」上漲/下跌幅度(%)?", value=1.2, step=0.1)

        submit = st.form_submit_button("取得投資建議")

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
            "beta_pref": [beta_min, beta_max],
        }
        payload = {"kyc": client}
        st.session_state["client"] = client
        st.session_state["advice"] = requests.post(f"{GATE}/advise", json=payload, timeout=90).json()
        st.success("✅ 已生成建議，請切換到『建議基金』")

# --- Tab2: 顯示基金建議 + 視覺化 ---
with tab2:
    if "advice" in st.session_state:
        st.subheader("投資建議基金")
        
        # 基金基本資訊
        fund = st.session_state["advice"]["selected_fund"]
        st.write("### 基金基本資訊")
        st.table(pd.DataFrame([fund]))
        
        # 基金 Beta 值
        st.write("### 基金 Beta 值")
        st.metric(label="基金 Beta", value=f"{fund['beta']:.2f}")

        # 個股 Beta 值
        if "stock_betas" in st.session_state["advice"]:
            stock_betas = st.session_state["advice"]["stock_betas"]
            st.write("### 基金持股之個股 Beta 值")
            sb_df = pd.DataFrame(stock_betas.items(), columns=["Ticker", "Beta"])
            st.bar_chart(sb_df.set_index("Ticker"))
        
        # 基金價格預測
        if "fund_forecast" in st.session_state["advice"]:
            fc = st.session_state["advice"]["fund_forecast"]
            st.write("### 基金價格預測 (CAPM+GARCH)")
            fig, ax = plt.subplots()
            ax.bar(
                ["P5", "Median", "P95"],
                [fc["price_scenarios"].get("p5"),
                 fc["price_scenarios"].get("median"),
                 fc["price_scenarios"].get("p95")]
            )
            ax.set_ylabel("Price")
            st.pyplot(fig)

        # 個股價格預測
        if "stock_forecasts" in st.session_state["advice"]:
            st.write("### 基金持股之個股價格預測")
            fig, ax = plt.subplots()
            for t, fc in st.session_state["advice"]["stock_forecasts"].items():
                ax.plot(
                    ["P5", "Median", "P95"], 
                    [fc["price_scenarios"].get("p5"),
                     fc["price_scenarios"].get("median"),
                     fc["price_scenarios"].get("p95")],
                    label=t
                )
            ax.set_ylabel("Price")
            ax.legend()
            st.pyplot(fig)

        # 基金市場熱度
        if "market_heat" in st.session_state["advice"]:
            heat = st.session_state["advice"]["market_heat"]
            st.write("### 基金市場熱度")
            st.metric(label="Relative Volume Score", value=f"{heat.get('rel_volume_score', 0.0):.2f}")


        # 新增：基金相關提問
        st.write("### 詢問基金相關問題")
        q = st.text_input("請輸入問題（例如：基金風險、投資標的...）")
        if st.button("🔎 發送問題"):
            if q.strip():
                rag_payload = {"query": q}
                try:
                    #rag_resp = requests.post(f"{GATE}/rag/auto", json=rag_payload, timeout=90).json()
                    rag_resp = requests.post(f"{GATE}/query", json=rag_payload, timeout=90).json()
                    st.session_state["rag"] = rag_resp
                    st.success("✅ 已取得知識檢索結果，請切換到『知識檢索與報告』")
                except Exception as e:
                    st.error(f"❌ RAG 查詢失敗: {e}")

# --- Tab3: 顯示 RAG 與報告 ---
with tab3:
    if "rag" in st.session_state:
        st.subheader("知識檢索回答 (RAG)")
        rag = st.session_state["rag"]

        # AI 回答
        if rag.get("answer"):
            st.markdown(f"**AI 回答：**\n\n{rag['answer']}")

        # 引用來源
        if rag.get("passages"):
            st.markdown("**引用來源：**")
            for p in rag["passages"]:
                st.markdown(f"""
                **Rank {p.get('rank')}**  
                - 相似度：{p.get('similarity', 0):.2f}  
                - 來源：{p['metadata'].get('source', '未知')}  
                - 頁碼：{p['metadata'].get('page', '未知')}  
                - 類型：{p['metadata'].get('doc_type', '未知')}  
                - 截止日期：{p['metadata'].get('asof_date', '未知')}  
                - 摘要：{p.get('snippet', '')}
                """)

    if "client" in st.session_state and "advice" in st.session_state:
        if st.button("📄 生成完整投資報告"):
            payload = {
                "client": st.session_state["client"],
                "advice": st.session_state["advice"],
                "refs": {
                    "answer": st.session_state.get("rag", {}).get("answer", ""),
                    "contexts": st.session_state.get("rag", {}).get("passages", [])
                },
            }
            rpt = requests.post(f"{GATE}/report", json=payload, timeout=120).json()

            if "markdown" in rpt:
                st.download_button("⬇️ 下載 Markdown 報告", rpt["markdown"], file_name="report.md")
            if "html" in rpt:
                st.download_button("⬇️ 下載 HTML 報告", rpt["html"], file_name="report.html")
            if "pdf" in rpt:
                pdf_bytes = base64.b64decode(rpt["pdf"])
                st.download_button("⬇️ 下載 PDF 報告", pdf_bytes, file_name="report.pdf", mime="application/pdf")






