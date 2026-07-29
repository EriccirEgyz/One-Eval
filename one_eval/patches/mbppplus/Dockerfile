FROM python:3.11-slim

ENV HF_ENDPOINT=https://hf-mirror.com
ENV HF_HOME=/root/.cache/huggingface
ENV HF_HUB_ENABLE_HF_TRANSFER=0

RUN pip install --no-cache-dir uv

RUN uv pip install --system evalplus openai httpx && \
    rm -rf /root/.cache/pip /root/.cache/uv

WORKDIR /workspace/repo
