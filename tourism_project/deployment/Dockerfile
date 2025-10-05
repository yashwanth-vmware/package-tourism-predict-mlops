# Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ---- Create non-root user with writable HOME ----
RUN useradd -m -u 1000 appuser
ENV HOME=/home/appuser
WORKDIR $HOME/app

# Pre-create Streamlit dirs before first import (prevents /.streamlit PermissionError)
RUN mkdir -p $HOME/.streamlit $HOME/.cache/streamlit && chown -R appuser:appuser $HOME

# ---- System deps (optional but helpful for scientific stacks) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# ---- Python deps ----
# You can pin versions as needed
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ---- App code ----
COPY app.py ./app.py
RUN chown -R appuser:appuser $HOME

# ---- Drop privileges ----
USER appuser

# ---- Streamlit runtime env (set BEFORE process starts) ----
ENV STREAMLIT_CONFIG_DIR=$HOME/.streamlit \
    STREAMLIT_CACHE_DIR=$HOME/.cache/streamlit \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    PORT=7860 \
    STREAMLIT_SERVER_PORT=7860

EXPOSE 7860

# Hugging Face Spaces will provide HF_TOKEN automatically if you enable it.
# Provide your model repo via env at deploy time:
#   MODEL_REPO=Yashwanthsairam/package-tourism-predict
#   MODEL_FILENAME=best_tourism_pipeline.joblib
CMD streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0
