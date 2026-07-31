# Stage 1: Build dependencies
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml .
COPY src/ src/

# Install torch CPU-only first (smaller wheel, avoids 530MB CUDA download)
RUN pip install --no-cache-dir --timeout=300 --retries 5 torch --index-url https://download.pytorch.org/whl/cpu
# Both ML extras, not just [ml]. neuralforecast lives in [ml-neural] because
# its ray dependency has no Windows wheel past Python 3.12, but this image is
# Linux on 3.12, where ray is fine, so the image keeps TFT, PatchTST and N-HiTS.
RUN pip install --no-cache-dir --timeout=600 --retries 10 ".[ml,ml-neural]"

# Stage 2: Runtime (no build-essential)
FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY src/ src/

ENV PATH="/opt/venv/bin:$PATH"

VOLUME /app/data

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "src/quant_tools/dashboard/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
