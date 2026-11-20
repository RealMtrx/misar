# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

LABEL org.opencontainers.image.title="Misar (مِسار)"
LABEL org.opencontainers.image.description="Sovereign Arabic LLM Preprocessing & Token Efficiency Engine"
LABEL org.opencontainers.image.authors="Muneer Jarab Matrak Al-Theeb <181740338+RealMtrx@users.noreply.github.com>"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.version="2.6.0"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install package
COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir .

# Create non-root user for enterprise production security
RUN useradd -m -u 10001 misaruser && \
    chown -R misaruser:misaruser /app
USER misaruser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; sys_exit = 0 if urllib.request.urlopen('http://localhost:8000/api/health').getcode() == 200 else 1; exit(sys_exit)"

CMD ["python", "-m", "misar", "serve", "--host", "0.0.0.0", "--port", "8000", "--no-browser"]
