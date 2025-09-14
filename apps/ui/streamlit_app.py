import os
import base64
import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ---------------- 基本設定 ----------------
GATE = os.getenv("GATEWAY_URL", "http://localhost:8000")
CSV_PATH = os.getenv("TRAIN_CSV_PATH", "train.csv")

# 可選：避免初次啟動沒有 CSV 中斷
if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
else:
    df = pd.DataFrame({
        "Gender": ["Male","Female"],
        "Married": ["Yes","No"],
        "Dependents": ["0","1","2","3+"],
        "Education": ["Graduate","Not Graduate"],
        "Self_Employed": ["Yes","No"],
        "Property_Area": ["Urban","Semiurban","Rural"]
    })

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
        }
        payload = {"kyc": client}
        st.session_state["client"] = client

        try:
            st.session_state["advice"] = requests.post(
                f"{GATE}/advise", json=payload, timeout=90
            ).json()
            st.success("✅ 已生成建議，請切換到『建議基金』")
        except Exception as e:
            st.warning(f"無法取得建議：{e}")

# --- Tab2: 顯示基金建議 + 視覺化 ---
with tab2:
    if "advice" in st.session_state and isinstance(st.session_state["advice"], dict):
        st.subheader("投資建議基金")

        fund = st.session_state["advice"].get("selected_fund", {})
        if fund:
            st.write("### 基金基本資訊")
            st.table(pd.DataFrame([fund]))

            if "beta" in fund:
                st.write("### 基金 Beta 值")
                st.metric(label="基金 Beta", value=f"{fund['beta']:.2f}")

        if "stock_betas" in st.session_state["advice"]:
            sb_df = pd.DataFrame(
                st.session_state["advice"]["stock_betas"].items(),
                columns=["Ticker", "Beta"],
            )
            st.bar_chart(sb_df.set_index("Ticker"))

        if "fund_forecast" in st.session_state["advice"]:
            fc = st.session_state["advice"]["fund_forecast"]
            st.write("### 基金價格預測 (CAPM+i.i.d. 殘差模擬)")
            vals = [
                fc["price_scenarios"]["P5_10d"],
                fc["price_scenarios"]["Median_10d"],
                fc["price_scenarios"]["P95_10d"],
            ]
            fig, ax = plt.subplots()
            ax.bar(["P5", "Median", "P95"], vals)
            ax.set_ylabel("Price")
            st.pyplot(fig)
            if fc.get("forecast_plot"):
                st.image(fc["forecast_plot"], caption="基金預測圖")

        if "stock_forecasts" in st.session_state["advice"]:
            st.write("### 基金持股之個股價格預測")
            fig, ax = plt.subplots()
            for t, fc in st.session_state["advice"]["stock_forecasts"].items():
                ax.plot(["P5", "Median", "P95"],
                        [fc["P5_10d"], fc["Median_10d"], fc["P95_10d"]],
                        marker="o", label=t)
            ax.legend()
            st.pyplot(fig)
            for t, fc in st.session_state["advice"]["stock_forecasts"].items():
                if fc.get("forecast_plot"):
                    st.image(fc["forecast_plot"], caption=f"{t} 預測圖")

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
                rag_resp = requests.post(f"{GATE}/query", json=payload, timeout=90).json()
                st.session_state["rag"] = rag_resp
                st.success("✅ 已取得知識檢索結果，請切換到『知識檢索與報告』")
            else:
                st.warning("請輸入您的問題！")

# --- Tab3: 顯示 RAG 與報告 ---
with tab3:
    if "rag" in st.session_state and isinstance(st.session_state["rag"], dict):
        st.subheader("知識檢索回答 (RAG)")
        rag = st.session_state["rag"]

        if rag.get("answer"):
            st.markdown(f"**AI 回答：**\n\n{rag['answer']}")

        if rag.get("passages"):
            st.markdown("**引用來源（依 doc_type 呈現對應欄位）：**")
            for p in rag["passages"]:
                meta = p.get("metadata", {})
                doc_type = meta.get("doc_type", "未知")
                base_lines = [
                    f"- 相似度：{p.get('similarity', 0):.2f}",
                    f"- 基金代碼：{meta.get('fund_code', '未知')}",
                    f"- 基金名稱：{meta.get('fund_name', '未知')}",
                    f"- 類型：{doc_type}",
                    f"- 截止日期：{meta.get('asof_date', '未知')}",
                    f"- 風險等級：{meta.get('risk_level','未知')}",
                    f"- 幣別：{', '.join(meta.get('currency', []) or [])}",
                ]

                if doc_type == "月報":
                    if meta.get("fund_manager"): base_lines.append(f"- 基金經理人：{meta['fund_manager']}")
                    if meta.get("fund_size"): base_lines.append(f"- 基金規模：{meta['fund_size']}")
                    if meta.get("custodian"): base_lines.append(f"- 保管銀行：{meta['custodian']}")
                    if meta.get("mgmt_fee"): base_lines.append(f"- 經理費：{meta['mgmt_fee']}")
                    if meta.get("top_holdings"): base_lines.append(f"- 前十大持股：{meta['top_holdings']}")
                    if meta.get("industries"): base_lines.append(f"- 產業配置：{meta['industries']}")
                    if meta.get("regions"): base_lines.append(f"- 地區配置：{meta['regions']}")
                    if meta.get("performance"): base_lines.append(f"- 績效：{meta['performance']}")
                    if meta.get("strategy"): base_lines.append(f"- 策略：{meta['strategy']}")

                elif doc_type == "公開說明書":
                    if meta.get("establish_date"): base_lines.append(f"- 成立日期：{meta['establish_date']}")
                    if meta.get("management_company"): base_lines.append(f"- 經理公司：{meta['management_company']}")
                    if meta.get("custodian"): base_lines.append(f"- 保管銀行：{meta['custodian']}")
                    if meta.get("fund_type"): base_lines.append(f"- 基金型態：{meta['fund_type']}")
                    if meta.get("duration"): base_lines.append(f"- 存續期間：{meta['duration']}")
                    if meta.get("distribution"): base_lines.append(f"- 收益分配：{meta['distribution']}")
                    if meta.get("investment_scope"): base_lines.append(f"- 投資範圍：{meta['investment_scope']}")
                    if meta.get("features"): base_lines.append(f"- 特色：{meta['features']}")
                    if meta.get("fees"): base_lines.append(f"- 費用：{meta['fees']}")
                    if meta.get("nav_announcement"): base_lines.append(f"- 淨值公告：{meta['nav_announcement']}")
                    if meta.get("suitability"): base_lines.append(f"- 適合屬性：{meta['suitability']}")

                if p.get("snippet"):
                    base_lines.append(f"- 摘要：{p['snippet']}")

                st.markdown("\n".join(base_lines))

    if "client" in st.session_state and "advice" in st.session_state:
        if st.button("📄 生成完整投資報告"):
            payload = {
                "client": st.session_state["client"],
                "advice": st.session_state["advice"],
                "refs": {
                    "answer": st.session_state.get("rag", {}).get("answer", ""),
                    "contexts": st.session_state.get("rag", {}).get("passages", []),
                },
            }
            try:
                resp = requests.post(f"{GATE}/report", json=payload, timeout=120).json()
                if "markdown" in resp:
                    st.download_button("⬇️ 下載 Markdown 報告", resp["markdown"], file_name="report.md")
                if "html" in resp:
                    st.download_button("⬇️ 下載 HTML 報告", resp["html"], file_name="report.html")
                if "pdf" in resp:
                    pdf_bytes = base64.b64decode(resp["pdf"])
                    st.download_button("⬇️ 下載 PDF 報告", pdf_bytes, file_name="report.pdf", mime="application/pdf")
            except Exception as e:
                st.warning(f"無法生成報告：{e}")







