FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

# Requirements first so the dependency layer caches across code changes.
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY ml ./ml
COPY backend ./backend
COPY original.csv ./original.csv

# Cloud Run injects PORT; uvicorn must bind it rather than a fixed port.
CMD exec uvicorn backend.main:app --host 0.0.0.0 --port ${PORT} --workers 1
