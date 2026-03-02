FROM docker.io/library/python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_HOME=/app \
    APP_PORT=8000

WORKDIR ${APP_HOME}

# Minimal runtime deps for common HTTP/TLS/DNS behavior.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY app ./app
COPY run_worker.py ./run_worker.py

# Non-root runtime user.
RUN useradd --system --create-home --home-dir /home/appuser appuser \
    && chown -R appuser:appuser ${APP_HOME}
USER appuser

EXPOSE 8000

# Default process: FastAPI API server.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
