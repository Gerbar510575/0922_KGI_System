FROM python:3.11-slim
WORKDIR /app
COPY apps/rag-service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY apps/rag-service /app
ENV PYTHONUNBUFFERED=1
CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8002"]
