FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dépendances (copiées en premier pour maximiser le cache Docker)
COPY requirements.txt /app/

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

# Code et assets
COPY src /app/src
COPY static /app/static
COPY streamlit_app.py run_dashboard.py openspartan_graph.py db_profiles.json metadata.json /app/

# Bonnes pratiques: ne pas tourner en root (meilleure compatibilité si /appdata est monté)
RUN adduser --disabled-password --gecos "" --uid 10001 appuser \
    && mkdir -p /appdata \
    && chown -R appuser:appuser /app /appdata

USER appuser

EXPOSE 8501

# Optionnel: préremplir la DB via OPENSPARTAN_DB=/data/xxx.db
ENV OPENSPARTAN_DB=""

# Healthcheck Streamlit (endpoint officiel)
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health').read()"]

CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
