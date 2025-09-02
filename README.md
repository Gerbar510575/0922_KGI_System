# KFH Advisor 系統

本專案是一個基於 **FastAPI + LangChain + Qdrant + Docker** 的多服務 AI Agent 架構，包含：
- **RAG Service (apps/rag-service)**：知識檢索與回答
- **Gateway (apps/gateway)**：服務入口，路由到各子服務
- **Advisor (apps/advisor-service)**：投資建議
- **Report (apps/report-service)**：報告生成
- **Market (apps/market-service)**：市場數據與熱度
- **UI (apps/ui)**：前端頁面（Streamlit）

---

## 📊 系統架構流程圖

```mermaid
flowchart LR
    UI[使用者瀏覽器\nhttp://localhost:8501] -->|輸入問題| Gateway[Gateway Service\n:8000]
    Gateway -->|/rag/auto| RAG[RAG Service\n:8002]
    RAG -->|檢索向量| Qdrant[Qdrant 向量資料庫\n:6333]
    RAG -->|呼叫 Gemini API| Gemini[Google Gemini Embedding/LLM]
    RAG -->|組合回答| Gateway
    Gateway -->|傳回答案 JSON| UI




