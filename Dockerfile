FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FASTEMBED_CACHE_PATH=/app/models \
    CHROMA_DB_DIR=/app/storage/chroma

WORKDIR /app

RUN useradd --create-home --uid 1000 app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Download the default embedding model while building, so the first upload
# does not wait for it. Build with --build-arg PRELOAD_MODEL=false to skip.
ARG PRELOAD_MODEL=true
ARG EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RUN mkdir -p /app/models /app/storage \
    && if [ "$PRELOAD_MODEL" = "true" ]; then \
        python -c "import sys; from fastembed import TextEmbedding; TextEmbedding(sys.argv[1])" \
            "$EMBEDDING_MODEL"; \
    fi \
    && chown -R app:app /app/models /app/storage

COPY --chown=app:app . .

USER app

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
