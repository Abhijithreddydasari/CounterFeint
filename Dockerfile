############################################################
# CounterFeint — multi-agent ad-fraud FraudArena.
#
# Build:
#     docker build -t counterfeint:latest -f counterfeint/Dockerfile counterfeint/
# Run:
#     docker run --rm -p 8000:8000 counterfeint:latest
# Health:
#     curl http://localhost:8000/api/v1/health
############################################################

ARG PYTHON_VERSION=3.11

############################
# Stage 1 — builder (wheel)
############################
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

# Install build tooling.
RUN pip install --upgrade pip build wheel setuptools

# Leverage Docker's layer cache: copy manifest + deps first.
COPY pyproject.toml ./

# Copy the rest of the package (uses pyproject `package-dir = "."`).
COPY . .

# Pre-build wheels for all project dependencies (cached layer) AND the
# project itself so the runtime image only needs `pip install *.whl`.
RUN pip wheel --no-cache-dir --wheel-dir /wheels --no-deps . && \
    pip wheel --no-cache-dir --wheel-dir /wheels \
        "openenv-core[core]>=0.2.3" \
        "fastapi>=0.115.0" \
        "pydantic>=2.0.0" \
        "uvicorn[standard]>=0.24.0" \
        "websockets>=13.0" \
        "requests>=2.31.0" \
        "faker==33.1.0" \
        "networkx>=3.2" \
        "openai>=1.0.0" \
        "python-dotenv>=1.0.0"


############################
# Stage 2 — runtime
############################
FROM python:${PYTHON_VERSION}-slim AS runtime

ARG BUILD_SHA=dev
ARG BUILD_TIME=unknown

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    COUNTERFEINT_BUILD_SHA=${BUILD_SHA} \
    COUNTERFEINT_BUILD_TIME=${BUILD_TIME} \
    ENABLE_WEB_INTERFACE=false \
    PORT=8000

# Non-root user.
RUN groupadd --system counterfeint && \
    useradd --system --gid counterfeint --home /home/counterfeint counterfeint && \
    mkdir -p /home/counterfeint && \
    chown -R counterfeint:counterfeint /home/counterfeint

# Install deps from pre-built wheels.
COPY --from=builder /wheels /wheels
RUN pip install --upgrade pip && \
    pip install --no-cache-dir /wheels/*.whl && \
    rm -rf /wheels

# Drop to the non-root user.
USER counterfeint
WORKDIR /home/counterfeint

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import sys,urllib.request; r=urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=3); sys.exit(0 if r.status==200 else 1)" \
    || exit 1

CMD ["uvicorn", "counterfeint.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
