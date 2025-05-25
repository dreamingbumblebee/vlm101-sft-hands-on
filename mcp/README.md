# MCP (Model Context Protocol) Examples

이 프로젝트는 MCP(Model Context Protocol)를 활용한 다양한 예제들을 포함하고 있습니다. MCP는 AI 모델이 외부 도구와 서비스에 안전하게 접근할 수 있도록 하는 표준 프로토콜입니다.

## 📋 목차

- [프로젝트 개요](#프로젝트-개요)
- [시스템 아키텍처](#시스템-아키텍처)
- [빠른 시작](#빠른-시작)
- [개발 환경 설정](#개발-환경-설정)
- [MCP 서버들](#mcp-서버들)
- [Ollama & Open WebUI](#ollama--open-webui)
- [문제 해결](#문제-해결)

## 🎯 프로젝트 개요

이 프로젝트는 다음과 같은 MCP 서버들을 포함합니다:

- **Weather MCP Server**: 날씨 정보 제공 (예보, 경보)
- **Stock MCP Server**: 주식 정보 제공
- **MCP Proxy**: 여러 MCP 서버를 통합 관리
- **Open WebUI**: 웹 기반 AI 채팅 인터페이스

## 🏗️ 시스템 아키텍처

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Open WebUI    │    │   MCP Proxy     │    │   MCP Servers   │
│   (Port: 3000)  │◄──►│   (Port: 8000)  │◄──►│   (Port: 8002+) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Ollama        │    │   Fetch Server  │    │   Weather/Stock │
│   (Port: 11434) │    │   Time Server   │    │   APIs          │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 빠른 시작

### 1. Docker Compose로 전체 시스템 실행

```bash
# 전체 시스템 시작 (Open WebUI + MCP Proxy + MCP Servers)
./start_docker_compose.sh

# 또는 MCP 도구들만 실행
docker compose --profile tools up -d
```

### 2. 개별 서비스 실행

```bash
# Open WebUI만 실행
docker compose up openwebui -d

# MCP Proxy만 실행
docker compose --profile tools up mcp-proxy -d

# Weather MCP Server만 실행
docker compose --profile tools up weather-mcp -d

# Stock MCP Server만 실행
docker compose --profile tools up stock-mcp -d
```

### 3. 서비스 접속

- **Open WebUI**: http://localhost:3000
- **MCP Proxy API**: http://localhost:8000
- **Weather MCP Server**: http://localhost:8002
- **Stock MCP Server**: http://localhost:8003

## 🛠️ 개발 환경 설정

### UV 패키지 매니저 사용

Python 패키지 관리를 위해 UV를 사용합니다.

**참고 자료**: [UV 사용법 가이드](https://devocean.sk.com/blog/techBoardDetail.do?ID=167420&boardType=techBlog)

```bash
# 프로젝트 초기화
uv init .

# transformers 패키지 추가
uv add transformers

# Python 3.12 설치
uv python install 3.12

# Python 3.12 가상환경 생성
uv venv --python 3.12
```

### 필수 도구 설치

```bash
# UV 설치
curl -LsSf https://astral.sh/uv/install.sh | sh

# MCP 관련 도구 설치
uvx install mcpo mcp-server-fetch mcp-server-time
```

## 🔧 MCP 서버들

### MCP Proxy

MCP Proxy는 여러 MCP 서버를 통합 관리하는 중앙 서버입니다.

**설정 파일**: `mcp-proxy/mcp-config.json`

```json
{
  "mcpServers": {
    "fetch": {
      "command": "uvx",
      "args": ["mcp-server-fetch"]
    },
    "time": {
      "command": "uvx",
      "args": ["mcp-server-time", "--local-timezone=Asia/Seoul"]
    }
  }
}
```

**실행 방법**:
```bash
# Docker로 실행
docker compose --profile tools up mcp-proxy -d

# 로컬에서 실행
uvx mcpo --host 0.0.0.0 --port 8000 --config mcp-proxy/mcp-config.json
```

### Weather MCP Server

날씨 정보를 제공하는 MCP 서버입니다.

**주요 기능**:
- `get_alerts`: 미국 주별 날씨 경보 조회
- `get_forecast`: 위도/경도 기반 날씨 예보 조회

**실행 방법**:
```bash
# Docker로 실행
docker compose --profile tools up weather-mcp -d

# 로컬에서 실행
cd mcp-servers/weather
uv run weather.py
```

**API 문서**: http://localhost:8002/docs

### Stock MCP Server

주식 정보를 제공하는 MCP 서버입니다.

**실행 방법**:
```bash
# Docker로 실행
docker compose --profile tools up stock-mcp -d

# 로컬에서 실행
cd mcp-servers/stock
uv run stock.py
```

## 🤖 Ollama & Open WebUI

Ollama와 Open WebUI를 연동하여 사용하는 방법입니다.

**참고 자료**: 
- [Ollama & Open WebUI 연동 가이드](https://devocean.sk.com/blog/techBoardDetail.do?ID=165685&boardType=techBlog)
- [Qwen3 모델 라이브러리](https://ollama.com/library/qwen3)

### 모델 다운로드 및 실행

```bash
# Qwen3 모델 실행
ollama run qwen3:1.7b
```

정상적으로 실행되면 아래와 같이 표기됩니다:
```
ollama run qwen3:1.7b
>>> Send a message (/? for help)
```

테스트 후 `/exit` 명령어를 통해 종료할 수 있습니다.

### Open WebUI 최신버전 유지

```bash
# Open WebUI 업데이트
docker run --rm --volume /var/run/docker.sock:/var/run/docker.sock containrrr/watchtower --run-once open-webui
```

**참고**: https://github.com/open-webui/open-webui?tab=readme-ov-file#keeping-your-docker-installation-up-to-date

### Docker를 통한 Open WebUI 실행

#### GPU 사용 환경
Ollama가 로컬 컴퓨터에 설치되어 있는 경우:

```bash
docker run -d -p 3000:8080 --gpus=all \
  --add-host=host.docker.internal:host-gateway \
  -v open-webui:/app/backend/data \
  --name open-webui \
  --restart always \
  ghcr.io/open-webui/open-webui:main
```

#### CPU 전용 환경
GPU를 사용하지 않는 경우:

```bash
docker run -d -p 3000:8080 \
  -v ollama:/root/.ollama \
  -v open-webui:/app/backend/data \
  --name open-webui \
  --restart always \
  ghcr.io/open-webui/open-webui:ollama
```

**참고 자료**:
- [ollama & OpenWeb-UI Installation](https://github.com/open-webui/open-webui?tab=readme-ov-file#installing-open-webui-with-bundled-ollama-support)

## 🐛 문제 해결

### Open WebUI 로그인 오류

**증상**: "The email or password provided is incorrect. Please check for typos and try logging in again." 오류 발생

**해결 방법**:

1. Docker Compose 중지:
   ```bash
   docker compose down
   ```

2. Open WebUI 데이터 볼륨 삭제:
   ```bash
   docker volume rm open-webui-data
   ```

3. Docker Compose 재시작:
   ```bash
   docker compose up -d
   ```

이렇게 하면 Open WebUI가 초기 상태로 리셋되어 새로운 계정으로 로그인할 수 있습니다.

### MCP 서버 연결 문제

**증상**: MCP 서버에 연결할 수 없음

**해결 방법**:

1. 서버 상태 확인:
   ```bash
   docker ps -a
   ```

2. 로그 확인:
   ```bash
   docker logs mcp-proxy
   docker logs weather-mcp
   docker logs stock-mcp
   ```

3. 포트 충돌 확인:
   ```bash
   netstat -tulpn | grep :800
   ```

### 컨테이너 재시작

```bash
# 특정 서비스 재시작
docker compose restart mcp-proxy

# 전체 시스템 재시작
docker compose down
docker compose up -d
```

## 📚 추가 자료

- [MCP 공식 문서](https://modelcontextprotocol.io/)
- [Open WebUI GitHub](https://github.com/open-webui/open-webui)
- [Ollama 공식 사이트](https://ollama.ai/)
- [UV 패키지 매니저](https://docs.astral.sh/uv/)
