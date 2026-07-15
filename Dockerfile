# 文经客 AIP OPC 核心架构 - Docker 部署
FROM python:3.11-slim

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 应用代码
COPY src/ src/
COPY data/ data/

# 创建非 root 用户
RUN useradd -m -u 1000 aip && chown -R aip:aip /app

# 数据目录（确保存在且可写）
RUN mkdir -p /app/data/checkpoints /app/data/audio /app/data/digital_human /app/data/packages && \
    chown -R aip:aip /app/data
ENV DATA_DIR=/app/data

USER aip

# 健康检查
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "src.aip_core.main:app", "--host", "0.0.0.0", "--port", "8080"]
