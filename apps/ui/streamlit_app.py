import streamlit as st, pandas as pd, requests, os

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

tab1, tab2, tab3 = st.tabs(["填寫屬性", "建議基金", "知識檢索與報告"])

# --- Tab1: KYC 表單 ---
with tab1:
    with st.form("kyc_form"):
        name = st.text_input("姓名", "王小明")
        gender = st.selectbox("Gender", gender_opts)
        married = st.selectbox("Married", married_opts)
        dependents = st.selectbox("Dependents", dependents_opts)
        education = st.selectbox("Education", education_opts)
        self_emp = st.selectbox("Self Employed", self_emp_opts)
        property_area = st.selectbox("Property Area", property_opts)
        applicant_income = st.number_input("Applicant Income", min_value=0)
        coapplicant_income = st.number_input("Coapplicant Income", min_value=0)

        # === 新增 Beta 偏好輸入 ===
        beta_min = st.number_input("Beta 最小值", value=0.8, step=0.1)
        beta_max = st.number_input("Beta 最大值", value=1.2, step=0.1)

        submit = st.form_submit_button("取得建議")

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
            "beta_pref": [beta_min, beta_max],   # 傳到後端
        }
        payload = {"kyc": client}
        st.session_state["client"] = client
        st.session_state["advice"] = requests.post(f"{GATE}/advise", json=payload, timeout=90).json()
        st.success("✅ 已生成建議，請切換到『建議基金』")

# --- Tab2: 顯示基金建議 ---
with tab2:
    if "advice" in st.session_state:
        st.subheader("投資建議基金")
        fund = st.session_state["advice"]["selected_fund"]
        st.write("### 基金基本資訊")
        st.table(pd.DataFrame([fund]))
        st.write("### 推薦理由")
        for line in st.session_state["advice"]["explanation"]:
            st.write("- " + line)

        if st.session_state["advice"].get("suitability_flags"):
            st.warning("⚠️ 適合度提醒")
            for f in st.session_state["advice"]["suitability_flags"]:
                st.write(f"- {f['code']}：{f['issue']}")

# --- Tab3: 顯示 RAG 與報告 ---
with tab3:
    if "rag" in st.session_state:
        st.subheader("知識檢索回答 (RAG)")
        rag = st.session_state["rag"]
        if rag.get("answer"):
            st.markdown(f"**AI 回答：**\n\n{rag['answer']}")
        if rag.get("contexts"):
            st.markdown("**引用來源：**")
            for i, ctx in enumerate(rag["contexts"], 1):
                st.markdown(f"""
                **{i}. 來源：** {ctx.get('source')}  
                - 頁碼：{ctx.get('page', '未知')}  
                - 類型：{ctx.get('doc_type', '未知')}  
                - 截止日期：{ctx.get('asof_date', '未知')}  
                - 摘要：{ctx.get('chunk', '')[:300]}...
                """)

    if "client" in st.session_state and "advice" in st.session_state:
        if st.button("📄 生成完整投資報告"):
            payload = {
                "client": st.session_state["client"],
                "advice": st.session_state["advice"],
                "refs": {"contexts": st.session_state.get("rag", {}).get("contexts", [])},
            }
            rpt = requests.post(f"{GATE}/report", json=payload, timeout=120).json()
            st.download_button("⬇️ 下載 Markdown 報告", rpt["markdown"], file_name="report.md")
            st.download_button("⬇️ 下載 HTML 報告", rpt["html"], file_name="report.html")
            # PDF base64 decode
            pdf_bytes = base64.b64decode(rpt["pdf"])
            st.download_button("⬇️ 下載 PDF 報告", pdf_bytes, file_name="report.pdf", mime="application/pdf")



