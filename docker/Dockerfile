FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

COPY . .

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD ["uv", "run", "python", "src/main.py"]
