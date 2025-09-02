FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY apps/market-service/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY apps/market-service /app
EXPOSE 8005
CMD ["uvicorn","service:app","--host","0.0.0.0","--port","8005","--log-level","info"]
