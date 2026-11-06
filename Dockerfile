FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 LW_DATA_DIR=/data
WORKDIR /app
COPY requirements.txt requirements-snowflake.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-snowflake.txt
COPY . .
RUN useradd -m -u 1000 app && mkdir -p /data && chown app /data
USER app
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
