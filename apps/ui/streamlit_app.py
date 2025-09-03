import streamlit as st, pandas as pd, requests, os

GATE = os.getenv("GATEWAY_URL","http://localhost:8000")
CSV_PATH = os.getenv("TRAIN_CSV_PATH", "train - 複製.csv")

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

tab1, tab2, tab3 = st.tabs(["填寫屬性", "建議與引用", "生成報告"])

# --- Tab1: KYC 表單 ---
with tab1:
    with st.form("kyc_form"):
        gender = st.selectbox("Gender", gender_opts)
        married = st.selectbox("Married", married_opts)
        dependents = st.selectbox("Dependents", dependents_opts)
        education = st.selectbox("Education", education_opts)
        self_emp = st.selectbox("Self Employed", self_emp_opts)
        property_area = st.selectbox("Property Area", property_opts)
        applicant_income = st.number_input("Applicant Income", min_value=0)
        coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
        prefs = st.multiselect("偏好", ["低費用","ESG","美元資產"], default=["低費用"])
        submit = st.form_submit_button("取得建議")

    if submit:
        payload = {"kyc": {
            "Gender": gender,
            "Married": married,
            "Dependents": dependents,
            "Education": education,
            "Self_Employed": self_emp,
            "Property_Area": property_area,
            "ApplicantIncome": applicant_income,
            "CoapplicantIncome": coapplicant_income,
            "preferences": prefs
        }}
        st.session_state["client"] = payload["kyc"]
        st.session_state["advice"] = requests.post(f"{GATE}/advise", json=payload, timeout=60).json()
        st.success("✅ 已生成建議，請切換到『建議與引用』")

# --- Tab2: 建議 + RAG 問答 ---
with tab2:
    if "advice" in st.session_state:
        st.subheader("投資建議")
        st.json(st.session_state["advice"])

        st.subheader("輸入您的問題")
        question = st.text_input("例如：基金的收益來源與風險揭露？")
        if st.button("取得引用"):
            query = "；".join([p.get("name") for p in st.session_state["advice"].get("picks", []) if "name" in p]) + " " + question
            st.session_state["refs"] = requests.post(f"{GATE}/rag/fusion", json={
                "query": query, "backend": "qdrant", "topk": 3, "topn_context": 6
            }, timeout=90).json()
            st.success("✅ 已取得引用，請切到『生成報告』")

# --- Tab3: 生成報告 ---
with tab3:
    if "advice" in st.session_state and "refs" in st.session_state:
        payload = {"client": st.session_state["client"], "advice": st.session_state["advice"], "refs": st.session_state["refs"]}
        r = requests.post(f"{GATE}/report", json=payload, timeout=90).json()
        st.download_button("下載 Markdown", data=r["markdown"], file_name="investment_report.md")
        st.components.v1.html(r["html"], height=700, scrolling=True)


