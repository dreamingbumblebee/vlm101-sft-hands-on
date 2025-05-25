# Weather MCP Server

날씨 정보를 제공하는 MCP(Model Context Protocol) 서버입니다. 이 서버는 실시간 날씨 예보, 경보, 기상 데이터 등을 제공합니다.

## 🎯 기능

- **날씨 예보**: 위도/경도 기반 상세 날씨 예보
- **날씨 경보**: 미국 주별 날씨 경보 및 주의보
- **실시간 데이터**: 현재 기온, 습도, 풍속 등
- **다국어 지원**: 한국어 및 영어 지원

## 📁 파일 구조

```
weather/
├── weather.py         # 메인 서버 파일
├── __main__.py        # 모듈 실행 파일
├── Dockerfile         # Docker 이미지 빌드 설정
├── mcp-config.json    # MCP 서버 설정 파일
├── pyproject.toml     # Python 프로젝트 설정
├── uv.lock           # UV 의존성 잠금 파일
├── .python-version    # Python 버전 설정
├── .gitignore        # Git 무시 파일 목록
└── README.md         # 이 파일
```

## ⚙️ 설정

### mcp-config.json

```json
{
  "mcpServers": {
    "weather": {
      "command": "uv",
      "args": ["run", "weather.py"]
    }
  }
}
```

## 🚀 실행 방법

### Docker로 실행 (권장)

```bash
# Weather MCP Server만 실행
docker compose --profile tools up weather-mcp -d

# 전체 시스템과 함께 실행
./start_docker_compose.sh
```

### 로컬에서 실행

```bash
# 프로젝트 디렉토리로 이동
cd mcp-servers/weather

# UV 가상환경 활성화 (필요시)
source .venv/bin/activate

# 서버 실행
uv run weather.py
```

### MCPO를 통한 실행

```bash
# MCPO를 사용하여 실행 (API 키 포함)
uvx mcpo --port 8002 --api-key "top-secret" -- uv run weather.py

# 또는 API 키 없이 실행
uvx mcpo --port 8002 -- uv run weather.py
```

## 🔗 API 엔드포인트

- **기본 URL**: http://localhost:8002
- **API 문서**: http://localhost:8002/docs
- **OpenAPI 스펙**: http://localhost:8000/weather/openapi.json

## 📊 API 기능

### 1. get_alerts - 날씨 경보 조회

미국 주별 날씨 경보를 조회합니다.

**요청 예시**:
```json
{
  "state": "CA"
}
```

**응답 예시**:
```json
{
  "alerts": [
    {
      "event": "Flood Watch",
      "headline": "Flood Watch issued for California",
      "description": "Heavy rainfall expected...",
      "severity": "Moderate",
      "areas": ["Los Angeles County", "Orange County"]
    }
  ]
}
```

### 2. get_forecast - 날씨 예보 조회

위도/경도 기반 날씨 예보를 조회합니다.

**요청 예시**:
```json
{
  "latitude": 40.7128,
  "longitude": -74.0060
}
```

**응답 예시**:
```json
{
  "forecast": {
    "current": {
      "temperature": 22.5,
      "humidity": 65,
      "wind_speed": 12.3,
      "description": "Partly cloudy"
    },
    "daily": [
      {
        "date": "2024-01-15",
        "high": 25.0,
        "low": 18.0,
        "description": "Sunny"
      }
    ]
  }
}
```

## 🧪 테스트

### 테스트 모드 실행

```bash
# 테스트 모드로 실행
uv run weather.py --test
```

테스트 모드에서는 다음 항목들을 자동으로 테스트합니다:

1. **캘리포니아 주의 날씨 경보**
2. **뉴욕 주의 날씨 경보**  
3. **뉴욕시의 날씨 예보** (위도: 40.7128, 경도: -74.0060)
4. **로스앤젤레스의 날씨 예보** (위도: 34.0522, 경도: -118.2437)
5. **시카고의 날씨 예보** (위도: 41.8781, 경도: -87.6298)

### 수동 테스트

```bash
# 날씨 경보 테스트
curl -X POST "http://localhost:8002/get_alerts" \
  -H "Content-Type: application/json" \
  -d '{"state": "CA"}'

# 날씨 예보 테스트
curl -X POST "http://localhost:8002/get_forecast" \
  -H "Content-Type: application/json" \
  -d '{"latitude": 40.7128, "longitude": -74.0060}'
```

## 📊 사용 예시

### Python 클라이언트 예시

```python
import requests

# 날씨 경보 조회
response = requests.post(
    "http://localhost:8002/get_alerts",
    json={"state": "CA"}
)
print(response.json())

# 날씨 예보 조회
response = requests.post(
    "http://localhost:8002/get_forecast",
    json={
        "latitude": 40.7128,
        "longitude": -74.0060
    }
)
print(response.json())
```

### JavaScript 클라이언트 예시

```javascript
// 날씨 경보 조회
const alertsResponse = await fetch('http://localhost:8002/get_alerts', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({ state: 'CA' })
});
const alerts = await alertsResponse.json();

// 날씨 예보 조회
const forecastResponse = await fetch('http://localhost:8002/get_forecast', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    latitude: 40.7128,
    longitude: -74.0060
  })
});
const forecast = await forecastResponse.json();
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
uv run --watch weather.py

# 디버그 모드로 실행
uv run weather.py --debug

# 상세 로그와 함께 실행
uv run weather.py --verbose
```

## 🔍 로그 확인

```bash
# Docker 로그 확인
docker logs weather-mcp

# 실시간 로그 확인
docker logs -f weather-mcp

# 로컬 실행 시 로그
uv run weather.py --verbose
```

## 🐛 문제 해결

### 포트 충돌
```bash
# 포트 사용 확인
netstat -tulpn | grep :8002

# 다른 포트로 실행
uvx mcpo --port 8004 -- uv run weather.py
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
uvx mcpo --port 8002 -- uv run weather.py

# 또는 환경변수로 설정
export MCP_API_KEY="your-api-key"
uvx mcpo --port 8002 --api-key "$MCP_API_KEY" -- uv run weather.py
```

## 🌤️ 데이터 소스

이 서버는 다음과 같은 기상 데이터 소스를 사용합니다:

- **National Weather Service (NWS)**: 미국 기상청 공식 데이터
- **OpenWeatherMap**: 글로벌 날씨 데이터
- **WeatherAPI**: 실시간 기상 정보

## 🔒 보안

- API 키 인증 지원
- 요청 제한 (Rate Limiting)
- HTTPS 지원 (프로덕션 환경)
- 입력 데이터 검증

## 📚 참고 자료

- [MCP 공식 문서](https://modelcontextprotocol.io/)
- [National Weather Service API](https://www.weather.gov/documentation/services-web-api)
- [OpenWeatherMap API](https://openweathermap.org/api)
- [WeatherAPI](https://www.weatherapi.com/)

## 🤝 기여하기

1. 이슈 생성 또는 기존 이슈 확인
2. 포크 생성
3. 기능 브랜치 생성
4. 변경사항 커밋
5. Pull Request 생성

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 📝 변경 이력

- **v1.6.0**: API 문서 개선 및 테스트 모드 추가
- **v1.5.0**: 다국어 지원 추가
- **v1.4.0**: 실시간 데이터 기능 추가
- **v1.3.0**: Docker 지원 추가
- **v1.2.0**: MCP 프로토콜 지원
- **v1.1.0**: 기본 날씨 API 구현
- **v1.0.0**: 초기 릴리스