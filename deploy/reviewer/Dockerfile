FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "import duckdb; duckdb.connect().execute('INSTALL spatial')"
COPY app.py .
COPY lint/ lint/
COPY .streamlit/ .streamlit/
COPY start.sh .
RUN chmod +x start.sh
EXPOSE 8080
HEALTHCHECK CMD curl -f http://localhost:8080/_stcore/health || exit 1
ENTRYPOINT ["./start.sh"]
