# Stock MCP Server

주식 정보를 제공하는 MCP(Model Context Protocol) 서버입니다. 이 서버는 실시간 주식 데이터, 차트, 분석 정보 등을 제공합니다.

## 🎯 기능

- **실시간 주식 데이터**: 현재 주가, 거래량, 변동률 등
- **주식 차트**: 다양한 시간대별 차트 정보
- **기업 정보**: 기업 개요, 재무 정보 등
- **시장 데이터**: 시장 지수, 섹터별 성과 등

## 📁 파일 구조

```
stock/
├── stock.py           # 메인 서버 파일
├── Dockerfile         # Docker 이미지 빌드 설정
├── mcp-config.json    # MCP 서버 설정 파일
├── pyproject.toml     # Python 프로젝트 설정
├── uv.lock           # UV 의존성 잠금 파일
├── .python-version    # Python 버전 설정
└── README.md         # 이 파일
```

## ⚙️ 설정

### mcp-config.json

```json
{
  "mcpServers": {
    "stock": {
      "command": "uv",
      "args": ["run", "stock.py"]
    }
  }
}
```

## 🚀 실행 방법

### Docker로 실행 (권장)

```bash
# Stock MCP Server만 실행
docker compose --profile tools up stock-mcp -d

# 전체 시스템과 함께 실행
./start_docker_compose.sh
```

### 로컬에서 실행

```bash
# 프로젝트 디렉토리로 이동
cd mcp-servers/stock

# UV 가상환경 활성화 (필요시)
source .venv/bin/activate

# 서버 실행
uv run stock.py
```

### MCPO를 통한 실행

```bash
# MCPO를 사용하여 실행
uvx mcpo --port 8003 --api-key "top-secret" -- uv run stock.py

# 또는 API 키 없이 실행
uvx mcpo --port 8003 -- uv run stock.py
```

## 🔗 API 엔드포인트

- **기본 URL**: http://localhost:8003
- **API 문서**: http://localhost:8003/docs
- **OpenAPI 스펙**: http://localhost:8003/openapi.json

## 📊 사용 예시

### Python 클라이언트 예시

```python
import requests

# 주식 정보 조회
response = requests.post(
    "http://localhost:8003/get_stock_info",
    json={"symbol": "AAPL"}
)
print(response.json())

# 차트 데이터 조회
response = requests.post(
    "http://localhost:8003/get_chart_data",
    json={
        "symbol": "AAPL",
        "period": "1d",
        "interval": "1m"
    }
)
print(response.json())
```

### cURL 예시

```bash
# 주식 정보 조회
curl -X POST "http://localhost:8003/get_stock_info" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL"}'

# 차트 데이터 조회
curl -X POST "http://localhost:8003/get_chart_data" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "period": "1d", "interval": "1m"}'
```

## 🛠️ 개발 환경 설정

### 의존성 설치

```bash
# UV 패키지 매니저 사용
uv sync

# 또는 pip 사용
pip install -r requirements.txt
```

### 개발 모드 실행

```bash
# 개발 모드로 실행 (자동 재시작)
uv run --watch stock.py

# 디버그 모드로 실행
uv run stock.py --debug
```

## 🔍 로그 확인

```bash
# Docker 로그 확인
docker logs stock-mcp

# 실시간 로그 확인
docker logs -f stock-mcp

# 로컬 실행 시 로그
uv run stock.py --verbose
```

## 🐛 문제 해결

### 포트 충돌
```bash
# 포트 사용 확인
netstat -tulpn | grep :8003

# 다른 포트로 실행
uvx mcpo --port 8004 -- uv run stock.py
```

### 의존성 문제
```bash
# UV 캐시 정리
uv cache clean

# 의존성 재설치
uv sync --reinstall
```

### API 키 문제
```bash
# API 키 없이 실행
uvx mcpo --port 8003 -- uv run stock.py

# 또는 환경변수로 설정
export MCP_API_KEY="your-api-key"
uvx mcpo --port 8003 --api-key "$MCP_API_KEY" -- uv run stock.py
```

## 📈 데이터 소스

이 서버는 다음과 같은 데이터 소스를 사용합니다:

- **Yahoo Finance**: 실시간 주식 데이터
- **Alpha Vantage**: 고급 주식 데이터
- **Finnhub**: 실시간 시장 데이터

## 🔒 보안

- API 키 인증 지원
- 요청 제한 (Rate Limiting)
- HTTPS 지원 (프로덕션 환경)

## 📚 참고 자료

- [MCP 공식 문서](https://modelcontextprotocol.io/)
- [Yahoo Finance API](https://finance.yahoo.com/)
- [Alpha Vantage API](https://www.alphavantage.co/)
- [Finnhub API](https://finnhub.io/)

## 🤝 기여하기

1. 이슈 생성 또는 기존 이슈 확인
2. 포크 생성
3. 기능 브랜치 생성
4. 변경사항 커밋
5. Pull Request 생성

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.