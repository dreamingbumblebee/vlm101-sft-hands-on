FROM python:3.12-slim
WORKDIR /app
COPY . /app
RUN pip install uv mcpo
RUN uv venv && uv sync
# CMD ["uvx", "mcpo", "--host", "0.0.0.0", "--port", "8003", "--", "uv", "run", "stock.py"] 
CMD ["mcpo", "--host", "0.0.0.0", "--port", "8003", "--config", "/app/mcp-config.json"]