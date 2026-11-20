FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DUCKDB_MEMORY_LIMIT=180MB \
    DUCKDB_THREADS=2

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# DuckDB data file and temp spill directory live on the mounted volume
RUN mkdir -p /data/tmp
VOLUME ["/data"]

EXPOSE 8000
# Single worker: upload serialization and snapshot refcounts are in-process
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
