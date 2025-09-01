import streamlit as st, requests, os

GATE = os.getenv("GATEWAY_URL","http://localhost:8000")
st.set_page_config(page_title="KFH 個人化投資建議", layout="wide")
st.title("AI 生成個人化投資建議分析報告")

tab1, tab2, tab3 = st.tabs(["KYC 與宇宙", "建議與引用", "生成報告"])

with tab1:
    st.subheader("基本屬性")
    name = st.text_input("姓名", "王小明")
    risk = st.selectbox("風險等級", ["保守","穩健","積極"], index=1)
    goal = st.selectbox("投資目標", ["退休","教育","增長"], index=0)
    horizon = st.number_input("投資期限（年）", 1, 30, 5)
    prefs = st.multiselect("偏好", ["低費用","ESG","美元資產"], default=["低費用"])
    universe = st.text_area("候選商品（以逗號分隔）", "0050.TW, 006208.TW, VTI, BND")
    if st.button("取得建議"):
        payload = {"kyc":{"risk_level":risk,"horizon_years":horizon,"goal":goal,"preferences":prefs},
                   "universe":[t.strip() for t in universe.split(",") if t.strip()]}
        st.session_state["client"] = {"name":name,"risk_level":risk,"goal":goal,"horizon_years":horizon}
        st.session_state["advice"] = requests.post(f"{GATE}/advise", json=payload, timeout=30).json()
        st.success("完成建議，請切到『建議與引用』")

with tab2:
    if "advice" in st.session_state:
        st.subheader("建議與市場熱度")
        st.json(st.session_state["advice"])
        q = st.text_input("RAG 主題（可輸入：收益來源、風險揭露、費用說明…）", "收益來源")
        if st.button("取得引用"):
            tickers = [p["ticker"] for p in st.session_state["advice"]["picks"]]
            st.session_state["refs"] = requests.post(f"{GATE}/justify",
                                                     json={"query": "；".join(tickers) + " " + q}, timeout=30).json()
            st.success("已取得引用，請切到『生成報告』")
        
        
        # 0901加上去的版本
        if st.button("Multi-Query RAG（Qdrant）"):
            mq = requests.post(f"{GATE}/rag/multi_query", json={
                "query": q,           # 你在輸入框填的主題
                "backend": "qdrant",
                "topk": 3
            }, timeout=60).json()
            st.subheader("Multi-Query Answer")
            st.write(mq["answer"][:2000])
            with st.expander("Contexts"):
                st.json(mq["contexts"])

        # 若想即時抓網頁做臨時索引：
        # if st.button("Multi-Query RAG（Chroma+Web）"):
        #     mq = requests.post(f"{GATE}/rag/multi_query", json={
        #         "query": q,
        #         "backend": "chroma",
        #         "topk": 3,
        #         "urls": ["https://lilianweng.github.io/posts/2023-06-23-agent/"]
        #     }, timeout=90).json()
        #     ...
        
        # 0901加上去的版本
        if st.button("RAG-Fusion（Qdrant）"):
            rf = requests.post(f"{GATE}/rag/fusion", json={
                "query": q,
                "backend": "qdrant",
                "topk": 3,
                "topn_context": 6
            }, timeout=60).json()
            st.subheader("RRF Answer")
            st.write(rf["answer"][:2000])
            with st.expander("RRF Contexts"):
                st.json(rf["contexts"])

with tab3:
    if "advice" in st.session_state and "refs" in st.session_state:
        st.subheader("輸出報告")
        payload = {"client": st.session_state["client"], "advice": st.session_state["advice"], "refs": st.session_state["refs"]}
        r = requests.post(f"{GATE}/report", json=payload, timeout=60).json()
        st.download_button("下載 Markdown", data=r["markdown"], file_name="investment_report.md")
        st.components.v1.html(r["html"], height=700, scrolling=True)
