FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY apps/report-service/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY apps/report-service /app
EXPOSE 8004
CMD ["uvicorn","service:app","--host","0.0.0.0","--port","8004","--log-level","info"]