FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dépendances
COPY requirements.txt /app/
COPY src /app/src
COPY static /app/static
COPY streamlit_app.py run_dashboard.py openspartan_graph.py /app/

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

# Optionnel: préremplir la DB via OPENSPARTAN_DB=/data/xxx.db
ENV OPENSPARTAN_DB=""

CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
