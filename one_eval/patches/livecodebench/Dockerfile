FROM python:3.11-slim

ENV HF_ENDPOINT=https://hf-mirror.com
ENV HF_HOME=/root/.cache/huggingface
ENV HF_HUB_ENABLE_HF_TRANSFER=0

RUN pip install --no-cache-dir uv

# 将仓库源码复制进来用于安装依赖（运行时会被 volume 覆盖）
COPY repo /tmp/repo

# 安装全量依赖 + 降级 datasets（LiveCodeBench 依赖 >=3.2 但使用 HF loading script 需要 <3.0）
RUN uv pip install --system -e /tmp/repo && \
    uv pip install --system 'datasets<3.0' && \
    rm -rf /tmp/repo /root/.cache/pip /root/.cache/uv

WORKDIR /workspace/repo
