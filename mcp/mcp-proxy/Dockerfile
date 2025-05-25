# Python 3.12이 설치된 슬림 이미지(필요 최소 패키지만 포함된 가벼운 이미지)를 기반으로 사용
FROM python:3.12-slim

# 이후 명령어들이 실행될 작업 디렉토리를 /app으로 설정
WORKDIR /app

# 필요한 패키지 설치
# mcpo: Expose any MCP tool as an OpenAPI-compatible HTTP server (MCP-to-OpenAI proxy server)
# uv: An extremely fast Python package and project manager, written in Rust
RUN pip install mcpo uv httpx mcp[cli]

# 설정 디렉토리 생성
RUN mkdir -p /app/config /app/mcp-servers

# MCP 설정 파일 복사
COPY mcp-proxy/mcp-config.json /app/config/mcp-config.json

# MCP 서버들 복사
COPY mcp-servers /app/mcp-servers

# 설정 파일을 사용하여 실행
CMD ["mcpo", "--host", "0.0.0.0", "--port", "8000", "--config", "/app/config/mcp-config.json"]
