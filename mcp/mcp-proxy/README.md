# MCP Proxy

MCP Proxy는 여러 MCP(Model Context Protocol) 서버를 통합 관리하는 중앙 서버입니다. 이 서버는 Open WebUI와 같은 AI 인터페이스에서 다양한 도구와 서비스에 접근할 수 있도록 해줍니다.

## 🎯 기능

- **통합 관리**: 여러 MCP 서버를 하나의 엔드포인트로 관리
- **Fetch 서버**: 웹 페이지 및 파일 내용 가져오기
- **Time 서버**: 시간 및 날짜 정보 제공 (한국 시간대)
- **확장 가능**: 새로운 MCP 서버 쉽게 추가 가능

## 📁 파일 구조

```
mcp-proxy/
├── Dockerfile          # Docker 이미지 빌드 설정
├── mcp-config.json     # MCP 서버 설정 파일
└── README.md          # 이 파일
```

## ⚙️ 설정

### mcp-config.json

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

**설정된 서버들**:
- **fetch**: 웹 페이지 및 파일 내용 가져오기
- **time**: 시간 정보 (한국 시간대 설정)

## 🚀 실행 방법

### Docker로 실행 (권장)

```bash
# MCP Proxy만 실행
docker compose --profile tools up mcp-proxy -d

# 전체 시스템과 함께 실행
./start_docker_compose.sh
```

### 로컬에서 실행

```bash
# UV 패키지 매니저 설치 (필요시)
curl -LsSf https://astral.sh/uv/install.sh | sh

# MCP 관련 도구 설치
uvx install mcpo mcp-server-fetch mcp-server-time

# MCP Proxy 실행
uvx mcpo --host 0.0.0.0 --port 8000 --config mcp-config.json
```

## 🔗 API 엔드포인트

- **기본 URL**: http://localhost:8000
- **API 문서**: http://localhost:8000/docs
- **OpenAPI 스펙**: http://localhost:8000/openapi.json

## 🛠️ 새로운 MCP 서버 추가

새로운 MCP 서버를 추가하려면 `mcp-config.json` 파일을 수정하세요:

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
    },
    "your-new-server": {
      "command": "uvx",
      "args": ["your-mcp-server-package"]
    }
  }
}
```

## 🔍 로그 확인

```bash
# Docker 로그 확인
docker logs mcp-proxy

# 실시간 로그 확인
docker logs -f mcp-proxy
```

## 🐛 문제 해결

### 포트 충돌
```bash
# 포트 사용 확인
netstat -tulpn | grep :8000

# 다른 포트로 실행
uvx mcpo --host 0.0.0.0 --port 8001 --config mcp-config.json
```

### 서버 연결 실패
```bash
# 컨테이너 상태 확인
docker ps -a

# 컨테이너 재시작
docker compose restart mcp-proxy
```

## 📚 참고 자료

- [MCP 공식 문서](https://modelcontextprotocol.io/)
- [MCPO GitHub](https://github.com/jlowin/mcpo)
- [MCP Server Fetch](https://github.com/jlowin/mcp-server-fetch)
- [MCP Server Time](https://github.com/jlowin/mcp-server-time)