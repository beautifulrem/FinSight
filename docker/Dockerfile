# syntax=docker/dockerfile:1.7
# FinSight API + agent. Build from the repository root:
#   docker build -f docker/Dockerfile -t finsight .
# Torch/transformers are only needed for the optional FinBERT sentiment backend and the offline
# llm_response generator; they are skipped unless WITH_TORCH=1 (CPU wheels).
FROM python:3.13 AS builder
# numpy<2 (required by the shipped scikit-learn models) has no CPython 3.13 wheel. The full python
# image already ships a compiler, so wheels are built here and the slim runtime stays compiler-free.
ARG WITH_TORCH=0
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
COPY requirements.txt ./
# An optional CA bundle (e.g. a corporate proxy) can be supplied with
#   --secret id=ca,src=/path/to/ca-bundle.crt
RUN --mount=type=secret,id=ca,required=false \
    if [ -f /run/secrets/ca ]; then export PIP_CERT=/run/secrets/ca; fi; \
    grep -vE '^(torch|transformers|accelerate|sentencepiece)' requirements.txt > requirements-core.txt \
    && pip wheel -r requirements-core.txt -w /wheels \
    && if [ "$WITH_TORCH" = "1" ]; then \
         pip wheel --index-url https://download.pytorch.org/whl/cpu "torch>=2.0,<3" -w /wheels \
         && pip wheel "transformers>=4.40,<5" "accelerate>=0.30,<2" "sentencepiece>=0.2,<1" -w /wheels; \
       fi

FROM python:3.13-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    QI_MODELS_DIR=/app/models \
    QI_API_OUTPUT_DIR=/app/outputs/query_intelligence \
    QI_AGENT_TRACE_DIR=/app/outputs/traces \
    QI_AGENT_CHECKPOINT_DB=/app/state/agent.sqlite

WORKDIR /app
RUN --mount=type=bind,from=builder,source=/wheels,target=/wheels \
    pip install --no-index --find-links /wheels /wheels/*.whl

COPY query_intelligence ./query_intelligence
COPY sentiment ./sentiment
COPY scripts ./scripts
COPY training ./training
COPY evaluation ./evaluation
COPY config ./config
COPY schemas ./schemas
COPY models ./models
COPY data/runtime ./data/runtime
COPY data/*.json data/*.csv ./data/

RUN useradd --create-home --uid 10001 finsight \
    && mkdir -p /app/outputs /app/state \
    && chown -R finsight:finsight /app/outputs /app/state
USER finsight

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "query_intelligence.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
