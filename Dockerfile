FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml uv.lock* README.md ./
COPY src ./src
RUN uv sync --no-dev
COPY data ./data
EXPOSE 8000
CMD ["uv", "run", "--no-dev", "uvicorn", "optirag.api:app", "--host", "0.0.0.0", "--port", "8000"]
