# Image of the Smart Reading Coach AI service. The model is trained while the image is built,
# so the image always contains a model that matches its own scikit-learn version.
#
# Where the dataset comes from:
# - Locally: data/raw/leescatalogus_recommender.xlsx (gitignored, so it never goes to GitHub).
# - On Render: the secret file "leescatalogus_recommender.xlsx.b64" (the xlsx as base64 text,
#   see the README). Render offers secret files to the build as a BuildKit secret.

# Stage 1: dependencies and code (the CI pipeline builds up to here, it has no dataset)
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first, so they are cached when only the code changes
COPY pyproject.toml ./
COPY src ./src
RUN pip install -e ".[api]"

COPY api ./api
COPY data/raw ./data/raw

# Stage 2: train the model and run the API
FROM base AS runtime

# Decode the dataset from the secret file when it is there, train, then remove the raw dataset
# from the image. Only the cleaned data and the model stay behind.
RUN --mount=type=secret,id=leescatalogus_recommender_xlsx_b64,dst=/run/secrets/dataset.b64 \
    if [ -s /run/secrets/dataset.b64 ]; then \
        base64 --decode --ignore-garbage /run/secrets/dataset.b64 > data/raw/leescatalogus_recommender.xlsx; \
    fi \
    && test -s data/raw/leescatalogus_recommender.xlsx \
    && python -m leescoach.train --skip-evaluation \
    && rm -f data/raw/leescatalogus_recommender.xlsx

# Do not run as root
RUN useradd --create-home --uid 10001 coach && chown -R coach /app
USER coach

# Render sets PORT itself, locally the service listens on 8000
ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/api/health', timeout=4)"

CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port \"$PORT\""]
