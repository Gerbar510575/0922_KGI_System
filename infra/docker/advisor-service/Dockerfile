FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY apps/advisor-service/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY apps/advisor-service /app
EXPOSE 8003
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8003","--log-level","info"]
