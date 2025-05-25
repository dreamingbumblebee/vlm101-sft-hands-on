from typing import Any, Dict
import httpx
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("weather")

# Constants
NWS_API_BASE = "https://api.weather.gov"
USER_AGENT = "weather-app/1.0"

async def make_nws_request(url: str) -> dict[str, Any] | None:
    """Make a request to the NWS API with proper error handling."""
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/geo+json"
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except Exception:
            return None

def format_alert(feature: dict) -> str:
    """Format an alert feature into a readable string."""
    props = feature["properties"]
    return f"""
Event: {props.get('event', 'Unknown')}
Area: {props.get('areaDesc', 'Unknown')}
Severity: {props.get('severity', 'Unknown')}
Description: {props.get('description', 'No description available')}
Instructions: {props.get('instruction', 'No specific instructions provided')}
"""

@mcp.tool()
# async def get_alerts(state: str) -> str:
async def get_alerts(state: str) -> dict[str, Any]:
    """Get weather alerts for a US state.

    Args:
        state: Two-letter US state code (e.g. CA, NY)
    """
    url = f"{NWS_API_BASE}/alerts/active/area/{state}"
    data = await make_nws_request(url)

    if not data or "features" not in data:
        return "Unable to fetch alerts or no alerts found."

    if not data["features"]:
        return "No active alerts for this state."

    alerts = [format_alert(feature) for feature in data["features"]]
    # return "\n---\n".join(alerts)
    return {"alerts": "\n---\n".join(alerts)}

@mcp.tool()
# async def get_forecast(latitude: float, longitude: str) -> str:
async def get_forecast(latitude: float, longitude: float) -> dict[str, Any]:
    """Get weather forecast for a location.

    Args:
        latitude: Latitude of the location
        longitude: Longitude of the location
    """
    # First get the forecast grid endpoint
    points_url = f"{NWS_API_BASE}/points/{latitude},{longitude}"
    points_data = await make_nws_request(points_url)

    if not points_data:
        return "Unable to fetch forecast data for this location."

    # Get the forecast URL from the points response
    forecast_url = points_data["properties"]["forecast"]
    forecast_data = await make_nws_request(forecast_url)

    if not forecast_data:
        return "Unable to fetch detailed forecast."

    # Format the periods into a readable forecast
    periods = forecast_data["properties"]["periods"]
    forecasts = []
    for period in periods[:5]:  # Only show next 5 periods
        forecast = f"""
{period['name']}:
Temperature: {period['temperature']}°{period['temperatureUnit']}
Wind: {period['windSpeed']} {period['windDirection']}
Forecast: {period['detailedForecast']}
"""
        forecasts.append(forecast)

    # return "\n---\n".join(forecasts)
    return {"forecast": "\n---\n".join(forecasts)}

async def run_tests():
    """테스트 함수를 실행합니다."""
    print("Weather 서버 테스트 시작...")
    
    # 1. 캘리포니아 주의 날씨 경보 조회
    print("\n1. 캘리포니아 주의 날씨 경보:")
    result1 = await get_alerts("CA")
    print(result1)
    
    # 2. 뉴욕 주의 날씨 경보 조회
    print("\n2. 뉴욕 주의 날씨 경보:")
    result2 = await get_alerts("NY")
    print(result2)
    
    # 3. 뉴욕시의 날씨 예보 (위도: 40.7128, 경도: -74.0060)
    print("\n3. 뉴욕시의 날씨 예보:")
    result3 = await get_forecast(40.7128, -74.0060)
    print(result3)
    
    # 4. 로스앤젤레스의 날씨 예보 (위도: 34.0522, 경도: -118.2437)
    print("\n4. 로스앤젤레스의 날씨 예보:")
    result4 = await get_forecast(34.0522, -118.2437)
    print(result4)
    
    # 5. 시카고의 날씨 예보 (위도: 41.8781, 경도: -87.6298)
    print("\n5. 시카고의 날씨 예보:")
    result5 = await get_forecast(41.8781, -87.6298)
    print(result5)
    
    print("\n테스트 완료!")

if __name__ == "__main__":
    # 모드 선택
    import sys
    import asyncio
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # 테스트 모드
        print("테스트 모드로 실행합니다...")
        asyncio.run(run_tests())
    else:
        # MCP 서버 모드
        print("Starting weather server...")
        mcp.run(transport='stdio')