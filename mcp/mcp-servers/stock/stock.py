from typing import Any, Dict, List, Optional, Union
import httpx
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from mcp.server.fastmcp import FastMCP
from yahooquery import search as yq_search
import re
from bs4 import BeautifulSoup

# Initialize FastMCP server
mcp = FastMCP("stock")

async def resolve_ticker(ticker_or_name: str) -> Dict[str, Any]:
    """
    종목명 또는 티커를 받아서 실제 티커로 변환하는 공통 함수
    
    Args:
        ticker_or_name: 종목명 또는 티커 심볼
        
    Returns:
        Dict[str, Any]: {"ticker": "실제티커", "name": "회사명", "error": "에러메시지"}
    """
    # 입력 정리 (공백 제거, 대소문자 정규화)
    ticker_or_name = ticker_or_name.strip()
    
    # 이미 티커인 경우 (한국 주식: .KS, .KQ, 미국 주식: 대문자이면서 5글자 이하)
    if ticker_or_name.endswith('.KS') or ticker_or_name.endswith('.KQ') or (ticker_or_name.isupper() and len(ticker_or_name) <= 5):
        return {"ticker": ticker_or_name, "name": ticker_or_name}
    
    # 종목명인 경우 웹 검색을 통해 티커 찾기
    print(f"종목명 '{ticker_or_name}'을(를) 티커로 변환 중...")
    
    try:
        # 웹 검색을 통해 티커 찾기
        search_result = await search_stocks(ticker_or_name, limit=1)
        
        if "error" in search_result:
            return {"error": f"'{ticker_or_name}'에 대한 주식 정보를 찾을 수 없습니다."}
        
        # 검색 결과에서 첫 번째 티커와 회사명 추출
        result_text = search_result.get("result", "")
        
        # 티커 추출: (티커) 패턴 찾기
        ticker_match = re.search(r'\(([^)]+)\)', result_text)
        if not ticker_match:
            return {"error": f"'{ticker_or_name}'에 대한 티커를 찾을 수 없습니다."}
        
        ticker = ticker_match.group(1)
        
        # 회사명 추출: 첫 번째 줄에서 티커 앞부분
        lines = result_text.strip().split('\n')
        if lines:
            first_line = lines[0]
            # "1. 회사명 (티커)" 패턴에서 회사명 추출
            name_match = re.search(r'^\d+\.\s*(.+?)\s*\([^)]+\)', first_line)
            if name_match:
                company_name = name_match.group(1).strip()
            else:
                company_name = ticker_or_name
        else:
            company_name = ticker_or_name
        
        print(f"찾은 티커: {ticker}, 회사명: {company_name}")
        return {"ticker": ticker, "name": company_name}
        
    except Exception as e:
        return {"error": f"티커 변환 중 오류 발생: {str(e)}"}

async def resolve_multiple_tickers(ticker_or_name_list: List[str]) -> Dict[str, Any]:
    """
    여러 종목명을 한번에 티커로 변환하는 함수
    
    Args:
        ticker_or_name_list: 종목명 또는 티커 심볼 리스트
        
    Returns:
        Dict[str, Any]: {"tickers": [{"ticker": "티커", "name": "회사명"}], "errors": ["에러메시지"]}
    """
    results = {"tickers": [], "errors": []}
    
    for ticker_or_name in ticker_or_name_list:
        ticker_info = await resolve_ticker(ticker_or_name)
        
        if "error" in ticker_info:
            results["errors"].append(f"{ticker_or_name}: {ticker_info['error']}")
        else:
            results["tickers"].append({
                "ticker": ticker_info["ticker"],
                "name": ticker_info["name"]
            })
    
    return results

async def get_stock_info(ticker: str) -> Dict[str, Any]:
    """
    주식 기본 정보를 가져오는 함수
    """
    try:
        stock = yf.Ticker(ticker)
        
        # 타임아웃 설정으로 빠른 응답
        import asyncio
        try:
            # 비동기로 정보 가져오기 시도
            info = stock.info
            
            # 기본 정보가 없는 경우 에러로 처리
            if not info or len(info) == 0:
                print(f"티커 {ticker}에 대한 정보가 없습니다")
                return {"error": "주식 정보가 없습니다"}
            
            # 필요한 정보만 선택
            result = {
                "symbol": info.get("symbol", ""),
                "name": info.get("longName", "") or info.get("shortName", ""),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "country": info.get("country", ""),
                "currency": info.get("currency", ""),
                "currentPrice": info.get("currentPrice", None),
                "previousClose": info.get("previousClose", None),
                "open": info.get("open", None),
                "dayLow": info.get("dayLow", None),
                "dayHigh": info.get("dayHigh", None),
                "volume": info.get("volume", None),
                "marketCap": info.get("marketCap", None),
                "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow", None),
                "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh", None),
                "forwardPE": info.get("forwardPE", None),
                "trailingPE": info.get("trailingPE", None),
                "dividend_rate": info.get("dividendRate", None),
                "dividend_yield": info.get("dividendYield", None)
            }
            
            # 최소한의 필수 정보가 있는지 확인
            if not result["symbol"] or not result["name"]:
                print(f"티커 {ticker}의 필수 정보가 없습니다")
                return {"error": "필수 주식 정보가 없습니다"}
                
            return result
            
        except Exception as e:
            # 개별 정보 가져오기 실패 시
            print(f"티커 {ticker} 정보 가져오기 실패: {str(e)}")
            return {"error": f"주식 정보 가져오기 실패: {str(e)}"}
            
    except Exception as e:
        print(f"티커 {ticker} 처리 중 오류: {str(e)}")
        return {"error": f"정보를 가져오는 중 오류 발생: {str(e)}"}

def format_history_data(data: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    가격 히스토리 데이터를 포맷팅하는 함수
    """
    if data.empty:
        return []
    
    result = []
    for date, row in data.iterrows():
        result.append({
            "date": date.strftime("%Y-%m-%d"),
            "open": round(row["Open"], 2) if not pd.isna(row["Open"]) else None,
            "high": round(row["High"], 2) if not pd.isna(row["High"]) else None,
            "low": round(row["Low"], 2) if not pd.isna(row["Low"]) else None,
            "close": round(row["Close"], 2) if not pd.isna(row["Close"]) else None,
            "volume": int(row["Volume"]) if not pd.isna(row["Volume"]) else None
        })
    return result

@mcp.tool()
# async def get_stock_price(ticker_or_name: str) -> str:
async def get_stock_price(ticker_or_name: str) -> dict[str, Any]:
    """
    특정 종목의 현재 주가 정보를 가져옵니다. 회사 이름이나 티커 심볼 모두 사용 가능합니다.

    Args:
        ticker_or_name: 회사 이름 또는 주식 티커 심볼
            - 회사 이름 예시: '삼성전자', '애플', '마이크로소프트', '네이버', '카카오'
            - 한국 주식 티커 예시: '005930.KS' (삼성전자), '035420.KS' (네이버)
            - 미국 주식 티커 예시: 'AAPL' (애플), 'MSFT' (마이크로소프트)

    Returns:
        str: 주식의 상세 정보 (현재가, 거래량, 시가총액 등)
    """
    try:
        # 종목명을 티커로 변환
        ticker_info = await resolve_ticker(ticker_or_name)
        
        if "error" in ticker_info:
            return {"error": ticker_info["error"]}
        
        ticker = ticker_info["ticker"]
        company_name = ticker_info["name"]
        
        stock_info = await get_stock_info(ticker)
        
        if "error" in stock_info:
            # return f"오류: {stock_info['error']}"
            return {"error": f"오류: {stock_info['error']}"}
        
        result = f"""
종목 정보: {stock_info.get('name', '')} ({stock_info.get('symbol', '')})
국가: {stock_info.get('country', '정보 없음')}
섹터: {stock_info.get('sector', '정보 없음')}
산업: {stock_info.get('industry', '정보 없음')}

현재가: {stock_info.get('currentPrice', '정보 없음')} {stock_info.get('currency', '')}
전일 종가: {stock_info.get('previousClose', '정보 없음')} {stock_info.get('currency', '')}
시가: {stock_info.get('open', '정보 없음')} {stock_info.get('currency', '')}
당일 최저가: {stock_info.get('dayLow', '정보 없음')} {stock_info.get('currency', '')}
당일 최고가: {stock_info.get('dayHigh', '정보 없음')} {stock_info.get('currency', '')}
거래량: {stock_info.get('volume', '정보 없음')}

시가총액: {stock_info.get('marketCap', '정보 없음')}
52주 최저가: {stock_info.get('fiftyTwoWeekLow', '정보 없음')} {stock_info.get('currency', '')}
52주 최고가: {stock_info.get('fiftyTwoWeekHigh', '정보 없음')} {stock_info.get('currency', '')}
PER(선행): {stock_info.get('forwardPE', '정보 없음')}
PER(후행): {stock_info.get('trailingPE', '정보 없음')}
배당금: {stock_info.get('dividend_rate', '정보 없음')} {stock_info.get('currency', '')}
배당수익률: {stock_info.get('dividend_yield', '정보 없음') * 100 if stock_info.get('dividend_yield') is not None else '정보 없음'}%
"""
        # return result
        return {"result": result}
    except Exception as e:
        # return f"주가 정보를 가져오는 중 오류가 발생했습니다: {str(e)}"
        return {"error": f"주가 정보를 가져오는 중 오류가 발생했습니다: {str(e)}"}

@mcp.tool()
# async def get_stock_history(ticker: str, period: str = "1mo", interval: str = "1d") -> str:
async def get_stock_history(ticker_or_name: str, period: str = "1mo", interval: str = "1d") -> dict[str, Any]:
    """
    특정 종목의 주가 히스토리를 가져옵니다.

    Args:
        ticker_or_name: 주식 티커 심볼 또는 회사명 (예: 삼성전자, '005930.KS', 애플, 'AAPL')
        period: 기간 (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max 중 선택)
        interval: 간격 (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo 중 선택)
    """
    try:
        # 기간과 간격 확인
        valid_periods = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
        valid_intervals = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]
        
        if period not in valid_periods:
            # return f"잘못된 기간입니다. 다음 중에서 선택해주세요: {', '.join(valid_periods)}"
            return {"error": f"잘못된 기간입니다. 다음 중에서 선택해주세요: {', '.join(valid_periods)}"}
        
        if interval not in valid_intervals:
            # return f"잘못된 간격입니다. 다음 중에서 선택해주세요: {', '.join(valid_intervals)}"
            return {"error": f"잘못된 간격입니다. 다음 중에서 선택해주세요: {', '.join(valid_intervals)}"}
        
        # 종목명을 티커로 변환
        ticker_info = await resolve_ticker(ticker_or_name)
        
        if "error" in ticker_info:
            return {"error": ticker_info["error"]}
        
        ticker = ticker_info["ticker"]
        company_name = ticker_info["name"]
        
        # 주가 데이터 가져오기
        stock = yf.Ticker(ticker)
        history = stock.history(period=period, interval=interval)
        
        if history.empty:
            return {"error": f"{company_name} ({ticker})에 대한 히스토리 데이터가 없습니다."}
        
        # 데이터 포맷팅
        formatted_data = format_history_data(history)
        
        # 기본 정보
        info = await get_stock_info(ticker)
        name = info.get("name", company_name)
        
        # 결과 문자열 생성
        result = f"{name} ({ticker}) 주가 히스토리 (기간: {period}, 간격: {interval})\n\n"
        
        # 최근 5개 데이터만 표시
        for entry in formatted_data[-5:]:
            result += f"날짜: {entry['date']}, 시가: {entry['open']}, 고가: {entry['high']}, 저가: {entry['low']}, 종가: {entry['close']}, 거래량: {entry['volume']}\n"
        
        # 간단한 통계 추가
        close_prices = [entry["close"] for entry in formatted_data if entry["close"] is not None]
        if close_prices:
            avg_price = sum(close_prices) / len(close_prices)
            min_price = min(close_prices)
            max_price = max(close_prices)
            
            result += f"\n기간 내 평균 종가: {round(avg_price, 2)}"
            result += f"\n기간 내 최저 종가: {min_price}"
            result += f"\n기간 내 최고 종가: {max_price}"
            
            if len(close_prices) >= 2:
                first_price = close_prices[0]
                last_price = close_prices[-1]
                change = last_price - first_price
                change_pct = (change / first_price) * 100
                
                result += f"\n기간 내 변동: {round(change, 2)} ({round(change_pct, 2)}%)"
        
        # return result
        return {"result": result}
    except Exception as e:
        # return f"주가 히스토리를 가져오는 중 오류가 발생했습니다: {str(e)}"
        return {"error": f"주가 히스토리를 가져오는 중 오류가 발생했습니다: {str(e)}"}

@mcp.tool()
# async def get_earnings(ticker: str) -> str:
async def get_earnings(ticker_or_name: str) -> dict[str, Any]:
    """
    특정 종목의 실적(어닝) 정보를 가져옵니다.

    Args:
        ticker_or_name: 주식 티커 심볼 또는 회사명 (예: 삼성전자, '005930.KS', 애플, 'AAPL')
    """
    try:
        # 종목명을 티커로 변환
        ticker_info = await resolve_ticker(ticker_or_name)
        
        if "error" in ticker_info:
            return {"error": ticker_info["error"]}
        
        ticker = ticker_info["ticker"]
        company_name = ticker_info["name"]
        
        stock = yf.Ticker(ticker)
        info = await get_stock_info(ticker)
        name = info.get("name", company_name)
        
        # 최근 어닝 정보
        earnings_dates = None
        try:
            earnings_dates = stock.earnings_dates
        except Exception:
            pass
        
        if earnings_dates is None or (hasattr(earnings_dates, 'empty') and earnings_dates.empty):
            recent_earnings = "어닝 날짜 정보가 없습니다."
        else:
            # 최근 4개 어닝 정보만 표시
            recent_earnings = "최근 어닝 정보:\n"
            try:
                for i, (date, row) in enumerate(earnings_dates.iloc[:4].iterrows()):
                    eps_est = row.get('EPS Estimate', None)
                    eps_actual = row.get('Reported EPS', None)
                    surprise = row.get('Surprise(%)', None)
                    
                    recent_earnings += f"{date.strftime('%Y-%m-%d')}: "
                    recent_earnings += f"예상 EPS: {eps_est if eps_est is not None and not pd.isna(eps_est) else '정보 없음'}, "
                    recent_earnings += f"실제 EPS: {eps_actual if eps_actual is not None and not pd.isna(eps_actual) else '정보 없음'}, "
                    recent_earnings += f"서프라이즈: {surprise}%\n" if surprise is not None and not pd.isna(surprise) else "서프라이즈: 정보 없음\n"
            except Exception as e:
                recent_earnings += f"어닝 날짜 처리 중 오류: {str(e)}\n"
        
        # 재무제표 정보
        income_stmt = None
        try:
            income_stmt = stock.income_stmt
        except Exception:
            pass
        
        # 분기별 실적 (먼저 데이터 타입 확인)
        quarterly_info = "분기별 실적 정보가 없습니다."
        if income_stmt is not None:
            try:
                # DataFrame인 경우
                if isinstance(income_stmt, pd.DataFrame):
                    quarterly_info = "\n분기별 실적 (최근 4분기):\n"
                    # 가장 최근 4개 분기 데이터 추출
                    recent_quarters = income_stmt.columns[:4]
                    
                    for date in recent_quarters:
                        # 총수익(매출)
                        revenue = income_stmt.loc['Total Revenue', date] if 'Total Revenue' in income_stmt.index else None
                        # 순이익
                        net_income = income_stmt.loc['Net Income', date] if 'Net Income' in income_stmt.index else None
                        
                        formatted_date = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
                        quarterly_info += f"{formatted_date}: "
                        quarterly_info += f"매출: {revenue:,.0f}" if revenue is not None and not pd.isna(revenue) else "매출: 정보 없음"
                        quarterly_info += f", 순이익: {net_income:,.0f}\n" if net_income is not None and not pd.isna(net_income) else ", 순이익: 정보 없음\n"
                
                # Dict인 경우 (최신 yfinance 버전에서는 Dict 형태로 반환)
                elif isinstance(income_stmt, dict):
                    quarterly_info = "\n분기별 실적 (최근 4분기):\n"
                    # 키는 날짜, 값은 각 항목의 값
                    recent_quarters = list(income_stmt.keys())[:4]
                    
                    for date in recent_quarters:
                        date_data = income_stmt[date]
                        # 총수익(매출)
                        revenue = date_data.get('Total Revenue', None)
                        # 순이익
                        net_income = date_data.get('Net Income', None)
                        
                        formatted_date = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
                        quarterly_info += f"{formatted_date}: "
                        quarterly_info += f"매출: {revenue:,.0f}" if revenue is not None and not pd.isna(revenue) else "매출: 정보 없음"
                        quarterly_info += f", 순이익: {net_income:,.0f}\n" if net_income is not None and not pd.isna(net_income) else ", 순이익: 정보 없음\n"
            except Exception as e:
                quarterly_info = f"분기별 실적 정보 처리 중 오류: {str(e)}"
        
        # 다른 재무 정보 확인
        balance_sheet = None
        cash_flow = None
        try:
            balance_sheet = stock.balance_sheet
            cash_flow = stock.cashflow
        except Exception:
            pass
        
        # 연간 실적
        annual_info = "연간 실적 정보가 없습니다."
        if balance_sheet is not None or cash_flow is not None:
            try:
                annual_info = "\n연간 실적 (최근 정보):\n"
                
                # 먼저 balance_sheet에서 정보 추출
                if isinstance(balance_sheet, pd.DataFrame) and not balance_sheet.empty:
                    # 가장 최근 연도 데이터
                    latest_year = balance_sheet.columns[0]
                    year = latest_year.year if hasattr(latest_year, 'year') else str(latest_year).split('-')[0]
                    
                    # 총자산
                    total_assets = balance_sheet.loc['Total Assets', latest_year] if 'Total Assets' in balance_sheet.index else None
                    # 총부채
                    total_liabilities = balance_sheet.loc['Total Liabilities Net Minority Interest', latest_year] if 'Total Liabilities Net Minority Interest' in balance_sheet.index else None
                    
                    annual_info += f"{year} 총자산: {total_assets:,.0f}" if total_assets is not None and not pd.isna(total_assets) else f"{year} 총자산: 정보 없음"
                    annual_info += f", 총부채: {total_liabilities:,.0f}\n" if total_liabilities is not None and not pd.isna(total_liabilities) else ", 총부채: 정보 없음\n"
                
                # cash_flow에서 추가 정보 추출
                if isinstance(cash_flow, pd.DataFrame) and not cash_flow.empty:
                    # 가장 최근 연도 데이터
                    latest_year = cash_flow.columns[0]
                    year = latest_year.year if hasattr(latest_year, 'year') else str(latest_year).split('-')[0]
                    
                    # 영업활동 현금흐름
                    operating_cash_flow = cash_flow.loc['Operating Cash Flow', latest_year] if 'Operating Cash Flow' in cash_flow.index else None
                    # 투자활동 현금흐름
                    investing_cash_flow = cash_flow.loc['Investing Cash Flow', latest_year] if 'Investing Cash Flow' in cash_flow.index else None
                    
                    annual_info += f"{year} 영업활동 현금흐름: {operating_cash_flow:,.0f}" if operating_cash_flow is not None and not pd.isna(operating_cash_flow) else f"{year} 영업활동 현금흐름: 정보 없음"
                    annual_info += f", 투자활동 현금흐름: {investing_cash_flow:,.0f}\n" if investing_cash_flow is not None and not pd.isna(investing_cash_flow) else ", 투자활동 현금흐름: 정보 없음\n"
            except Exception as e:
                annual_info += f"연간 실적 정보 처리 중 오류: {str(e)}"
        
        # 다음 어닝 예정일
        next_earnings = "다음 어닝 예정일 정보가 없습니다."
        try:
            next_earnings_date = stock.calendar
            
            if next_earnings_date is not None and hasattr(next_earnings_date, 'empty') and not next_earnings_date.empty:
                next_date = next_earnings_date.iloc[0, 0] if len(next_earnings_date) > 0 and len(next_earnings_date.columns) > 0 else None
                next_earnings = f"\n다음 어닝 예정일: {next_date.strftime('%Y-%m-%d') if next_date is not None and not pd.isna(next_date) else '정보 없음'}"
        except Exception as e:
            next_earnings = f"\n다음 어닝 예정일 정보 처리 중 오류: {str(e)}"
        
        result = f"{name} ({ticker}) 어닝 정보\n\n{recent_earnings}\n{quarterly_info}\n{annual_info}\n{next_earnings}"
        # return result
        return {"result": result}
    except Exception as e:
        # return f"어닝 정보를 가져오는 중 오류가 발생했습니다: {str(e)}"
        return {"error": f"어닝 정보를 가져오는 중 오류가 발생했습니다: {str(e)}"}

@mcp.tool()
# async def search_stocks(query: str, limit: int = 5) -> str:
async def search_stocks(query: str, limit: int = 5) -> dict[str, Any]:
    """
    주식 심볼을 검색합니다.

    Args:
        query: 검색어 (회사 이름이나 심볼의 일부)
        limit: 결과 제한 개수 (기본값: 5)
    """
    try:
        matches = []
        
        # 1. yahooquery의 search 기능 활용 (웹 검색)
        try:
            print(f"yahooquery로 '{query}' 검색 시작...")
            search_results = yq_search(query)
            quotes = search_results.get('quotes', [])
            print(f"yahooquery 검색 결과: {len(quotes)}개 발견")
            
            if quotes and len(quotes) > 0:
                for i, result in enumerate(quotes[:limit]):
                    try:
                        ticker = result.get('symbol', '')
                        name = result.get('shortname', '') or result.get('longname', '')
                        exchange = result.get('exchange', '')
                        
                        print(f"  처리 중: {ticker} ({name}) - {exchange}")
                        
                        # 티커가 유효한지 확인
                        if not ticker or len(ticker.strip()) == 0:
                            print(f"    티커가 비어있음, 건너뜀")
                            continue
                        
                        # 기본 정보 가져오기 (타임아웃 설정)
                        stock_info = await get_stock_info(ticker)
                        if "error" not in stock_info and stock_info.get("name"):
                            # 유효한 정보가 있는 경우만 추가
                            matches.append({
                                "symbol": ticker,
                                "name": stock_info.get("name", name),
                                "country": stock_info.get("country", "미국" if exchange in ["NYQ", "NMS", "ASE"] else "기타"),
                                "price": stock_info.get("currentPrice", "정보 없음"),
                                "exchange": exchange
                            })
                            print(f"    성공: {ticker} 추가됨")
                        else:
                            print(f"    실패: {ticker} - {stock_info.get('error', '알 수 없는 오류')}")
                    except Exception as e:
                        # 개별 결과 처리 중 오류가 발생해도 계속 진행
                        print(f"개별 검색 결과 처리 중 오류 ({ticker}): {str(e)}")
                        continue
                        
                # yahooquery에서 결과를 찾았으면 여기서 종료 (다른 검색 시도 안함)
                if matches:
                    print(f"yahooquery에서 {len(matches)}개 결과 찾음, 다른 검색 중단")
                    # 중복 제거 및 제한
                    unique_matches = []
                    seen_symbols = set()
                    for match in matches:
                        if match["symbol"] not in seen_symbols:
                            seen_symbols.add(match["symbol"])
                            unique_matches.append(match)
                            if len(unique_matches) >= limit:
                                break
                    
                    if unique_matches:
                        result = f"'{query}'에 대한 검색 결과:\n\n"
                        for i, match in enumerate(unique_matches, 1):
                            result += f"{i}. {match['name']} ({match['symbol']})\n"
                            result += f"   국가: {match['country'] or '정보 없음'}\n"
                            if 'exchange' in match:
                                result += f"   거래소: {match['exchange']}\n"
                            result += f"   현재가: {match['price'] if match['price'] else '정보 없음'}\n\n"
                        
                        return {"result": result}
                        
        except Exception as e:
            # yahooquery 검색 실패 시 기존 방식으로 fallback
            print(f"yahooquery 검색 실패: {str(e)}")
            pass
        
        # 2. 한국 주식 웹 검색 (yfinance 검색 결과에 없는 경우에만)
        if not matches:
            # 한국어 검색어인 경우 한국 주식 웹 검색 시도
            if any(ord(char) > 127 for char in query):  # 한글이 포함된 경우
                korean_matches = await search_korean_stocks_web(query, limit)
                matches.extend(korean_matches)
            
            # 기존 하드코딩된 매핑도 백업으로 사용
            korean_stocks = {
                "삼성": "005930.KS",  # 삼성전자
                "삼성전자": "005930.KS",
                "현대": "005380.KS",  # 현대자동차
                "현대차": "005380.KS",
                "현대자동차": "005380.KS",
                "SK": "034730.KS",  # SK
                "SK하이닉스": "000660.KS",
                "LG": "003550.KS",  # LG
                "LG전자": "066570.KS",
                "네이버": "035420.KS",
                "카카오": "035720.KS",
                "셀트리온": "068270.KS",
                "기아": "000270.KS",
                "기아차": "000270.KS",
                "삼성바이오로직스": "207940.KS",
                "삼성바이오": "207940.KS",
                "삼성생명": "032830.KS",
                "삼성SDI": "006400.KS",
                "삼성물산": "028260.KS",
                "삼성화재": "000810.KS",
                "삼성증권": "016360.KS",
                "삼성엔지니어링": "028050.KS",
                "SK이노베이션": "096770.KS",
                "SK텔레콤": "017670.KS",
                "LG화학": "051910.KS",
                "LG생활건강": "051900.KS",
                "LG디스플레이": "034220.KS",
                "포스코": "005490.KS",
                "POSCO": "005490.KS",
                "신한금융": "055550.KS",
                "신한은행": "055550.KS",
            }
            
            for company, ticker in korean_stocks.items():
                if query.lower() in company.lower():
                    kr_info = await get_stock_info(ticker)
                    if "error" not in kr_info:
                        # 이미 추가된 심볼인지 확인
                        if not any(m["symbol"] == kr_info.get("symbol", ticker) for m in matches):
                            matches.append({
                                "symbol": kr_info.get("symbol", ticker),
                                "name": kr_info.get("name", company),
                                "country": kr_info.get("country", "한국"),
                                "price": kr_info.get("currentPrice", "정보 없음")
                            })
        
        # 3. 한국 주식 코드 직접 검색 (6자리 숫자)
        if query.isdigit():
            kr_ticker = f"{query.zfill(6)}.KS"  # 코스피
            kr_info = await get_stock_info(kr_ticker)
            if "error" not in kr_info:
                # 이미 추가된 심볼인지 확인
                if not any(m["symbol"] == kr_info.get("symbol", kr_ticker) for m in matches):
                    matches.append({
                        "symbol": kr_info.get("symbol", kr_ticker),
                        "name": kr_info.get("name", ""),
                        "country": kr_info.get("country", "한국"),
                        "price": kr_info.get("currentPrice", "정보 없음")
                    })
            
            # 코스닥도 시도
            kq_ticker = f"{query.zfill(6)}.KQ"
            kq_info = await get_stock_info(kq_ticker)
            if "error" not in kq_info:
                # 이미 추가된 심볼인지 확인
                if not any(m["symbol"] == kq_info.get("symbol", kq_ticker) for m in matches):
                    matches.append({
                        "symbol": kq_info.get("symbol", kq_ticker),
                        "name": kq_info.get("name", ""),
                        "country": kq_info.get("country", "한국"),
                        "price": kq_info.get("currentPrice", "정보 없음")
                    })
        
        # 4. 미국 주요 기업 매핑 (yfinance 검색 결과에 없는 경우에만)
        if not matches:
            us_stocks = {
                "apple": "AAPL",
                "애플": "AAPL",
                "microsoft": "MSFT",
                "마이크로소프트": "MSFT",
                "구글": "GOOGL",
                "google": "GOOGL",
                "아마존": "AMZN",
                "amazon": "AMZN",
                "테슬라": "TSLA",
                "tesla": "TSLA",
                "메타": "META",
                "페이스북": "META",
                "facebook": "META",
                "meta": "META",
                "넷플릭스": "NFLX",
                "netflix": "NFLX",
                "nvidia": "NVDA",
                "엔비디아": "NVDA",
                "amd": "AMD",
                "intel": "INTC",
                "인텔": "INTC",
                "ibm": "IBM",
                "oracle": "ORCL",
                "walmart": "WMT",
                "월마트": "WMT",
                "disney": "DIS",
                "디즈니": "DIS",
                "coca": "KO",
                "코카콜라": "KO",
                "pepsi": "PEP",
                "펩시": "PEP",
                "mcdonalds": "MCD",
                "맥도날드": "MCD",
                "starbucks": "SBUX",
                "스타벅스": "SBUX",
                "nike": "NKE",
                "나이키": "NKE",
                "boeing": "BA",
                "보잉": "BA",
                "johnson": "JNJ",
                "visa": "V",
                "mastercard": "MA",
                "paypal": "PYPL",
                "페이팔": "PYPL",
                "jpmorgan": "JPM",
                "goldman": "GS",
                "exxon": "XOM",
                "chevron": "CVX",
                "toyota": "TM",
                "토요타": "TM",
                "honda": "HMC",
                "혼다": "HMC",
            }
            
            # 미국 회사 이름 검색
            for company, ticker in us_stocks.items():
                if query.lower() in company.lower():
                    us_info = await get_stock_info(ticker)
                    if "error" not in us_info:
                        matches.append({
                            "symbol": us_info.get("symbol", ticker),
                            "name": us_info.get("name", company),
                            "country": us_info.get("country", "미국"),
                            "price": us_info.get("currentPrice", "정보 없음")
                        })
        
        # 5. 심볼 직접 검색 (미국 주식) - yahooquery 검색 결과가 없는 경우에만
        if not matches:
            us_ticker = query.upper()
            if len(us_ticker) <= 5 and us_ticker.isalpha():  # 보통 미국 심볼은 1-5 글자 알파벳
                us_info = await get_stock_info(us_ticker)
                if "error" not in us_info:
                    # 이미 추가된 심볼인지 확인
                    if not any(m["symbol"] == us_info.get("symbol", us_ticker) for m in matches):
                        matches.append({
                            "symbol": us_info.get("symbol", us_ticker),
                            "name": us_info.get("name", ""),
                            "country": us_info.get("country", "미국"),
                            "price": us_info.get("currentPrice", "정보 없음")
                        })
        
        # 중복 제거 및 제한
        unique_matches = []
        seen_symbols = set()
        for match in matches:
            if match["symbol"] not in seen_symbols:
                seen_symbols.add(match["symbol"])
                unique_matches.append(match)
                if len(unique_matches) >= limit:
                    break
        
        if not unique_matches:
            return {"error": f"'{query}'에 대한 검색 결과가 없습니다. 정확한 종목 이름이나 심볼을 입력해보세요."}
        
        result = f"'{query}'에 대한 검색 결과:\n\n"
        for i, match in enumerate(unique_matches, 1):
            result += f"{i}. {match['name']} ({match['symbol']})\n"
            result += f"   국가: {match['country'] or '정보 없음'}\n"
            if 'exchange' in match:
                result += f"   거래소: {match['exchange']}\n"
            result += f"   현재가: {match['price'] if match['price'] else '정보 없음'}\n\n"
        
        return {"result": result}
    except Exception as e:
        return {"error": f"주식 검색 중 오류가 발생했습니다: {str(e)}"}

@mcp.tool()
# async def compare_stocks(tickers: str, period: str = "1y") -> str:
async def compare_stocks(tickers: str, period: str = "1y") -> dict[str, Any]:
    """
    여러 종목의 성과를 비교합니다.

    Args:
        tickers: 쉼표로 구분된 종목명 또는 티커 심볼 목록 (예: 'Apple,Microsoft,삼성전자' 또는 'AAPL,MSFT,005930.KS')
        period: 비교 기간 (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max 중 선택)
    """
    try:
        # 기간 확인
        valid_periods = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
        if period not in valid_periods:
            # return f"잘못된 기간입니다. 다음 중에서 선택해주세요: {', '.join(valid_periods)}"
            return {"error": f"잘못된 기간입니다. 다음 중에서 선택해주세요: {', '.join(valid_periods)}"}
        
        # 티커 목록 파싱
        ticker_list = [t.strip() for t in tickers.split(",")]
        if len(ticker_list) < 2:
            # return "비교하려면 최소 2개 이상의 종목이 필요합니다."
            return {"error": "비교하려면 최소 2개 이상의 종목이 필요합니다."}
        if len(ticker_list) > 5:
            # return "비교는 최대 5개 종목까지만 가능합니다."
            return {"error": "비교는 최대 5개 종목까지만 가능합니다."}
        
        results = []
        
        for ticker_or_name in ticker_list:
            try:
                # 종목명을 티커로 변환
                ticker_info = await resolve_ticker(ticker_or_name)
                
                if "error" in ticker_info:
                    results.append({
                        "ticker": ticker_or_name,
                        "name": ticker_or_name,
                        "error": ticker_info["error"]
                    })
                    continue
                
                ticker = ticker_info["ticker"]
                company_name = ticker_info["name"]
                
                # 주가 데이터 가져오기
                stock = yf.Ticker(ticker)
                history = stock.history(period=period)
                
                if history.empty:
                    results.append({
                        "ticker": ticker,
                        "name": company_name,
                        "error": "데이터 없음"
                    })
                    continue
                
                # 기본 정보
                info = await get_stock_info(ticker)
                name = info.get("name", company_name)
                
                # 성과 계산
                close_prices = history["Close"].dropna()
                if len(close_prices) < 2:
                    results.append({
                        "ticker": ticker,
                        "name": name,
                        "error": "충분한 데이터 없음"
                    })
                    continue
                
                first_price = close_prices.iloc[0]
                last_price = close_prices.iloc[-1]
                change = last_price - first_price
                change_pct = (change / first_price) * 100
                
                # 통계 계산
                avg_price = close_prices.mean()
                min_price = close_prices.min()
                max_price = close_prices.max()
                volatility = close_prices.std()
                
                results.append({
                    "ticker": ticker,
                    "name": name,
                    "first_price": round(first_price, 2),
                    "last_price": round(last_price, 2),
                    "change": round(change, 2),
                    "change_pct": round(change_pct, 2),
                    "avg_price": round(avg_price, 2),
                    "min_price": round(min_price, 2),
                    "max_price": round(max_price, 2),
                    "volatility": round(volatility, 2)
                })
                
            except Exception as e:
                results.append({
                    "ticker": ticker,
                    "name": ticker,
                    "error": str(e)
                })
        
        # 결과 포맷팅
        result = f"종목 비교 결과 (기간: {period})\n\n"
        
        # 성과 순으로 정렬 (변동률 기준)
        valid_results = [r for r in results if "error" not in r]
        error_results = [r for r in results if "error" in r]
        
        if valid_results:
            valid_results.sort(key=lambda x: x["change_pct"], reverse=True)
            
            result += "=== 성과 순위 ===\n"
            for i, r in enumerate(valid_results, 1):
                result += f"{i}. {r['name']} ({r['ticker']})\n"
                result += f"   변동률: {r['change_pct']:+.2f}% ({r['change']:+.2f})\n"
                result += f"   시작가: {r['first_price']} → 종가: {r['last_price']}\n"
                result += f"   평균가: {r['avg_price']}, 변동성: {r['volatility']:.2f}\n"
                result += f"   최저가: {r['min_price']}, 최고가: {r['max_price']}\n\n"
        
        if error_results:
            result += "=== 오류 발생 종목 ===\n"
            for r in error_results:
                result += f"- {r['ticker']}: {r['error']}\n"
        
        # return result
        return {"result": result}
    except Exception as e:
        # return f"종목 비교 중 오류가 발생했습니다: {str(e)}"
        return {"error": f"종목 비교 중 오류가 발생했습니다: {str(e)}"}

async def search_korean_stocks_web(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    웹에서 한국 주식을 검색하는 함수
    """
    try:
        matches = []
        
        # 1. 네이버 금융 검색 API 사용 (실제로는 더 복잡한 구현이 필요)
        try:
            # 네이버 금융 검색 URL (최신 엔드포인트)
            search_url = f"https://finance.naver.com/search/search.naver?query={query}"
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(search_url)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    # 종목 검색 결과 테이블 파싱
                    table = soup.find('table', class_='tbl_search')
                    if table:
                        rows = table.find_all('tr')
                        for row in rows[1:]:  # 첫 번째는 헤더
                            cols = row.find_all('td')
                            if len(cols) >= 2:
                                # 종목명 및 링크
                                name_tag = cols[0].find('a')
                                if name_tag and 'href' in name_tag.attrs:
                                    name = name_tag.text.strip()
                                    href = name_tag['href']
                                    # 종목코드 추출
                                    code_match = re.search(r'code=(\d+)', href)
                                    code = code_match.group(1) if code_match else None
                                    ticker = f"{code}.KS" if code else None
                                    # 현재가
                                    price = cols[1].text.strip().replace(',', '')
                                    # 결과 추가
                                    if ticker:
                                        stock_info = await get_stock_info(ticker)
                                        if "error" not in stock_info:
                                            matches.append({
                                                "symbol": ticker,
                                                "name": stock_info.get("name", name),
                                                "country": "한국",
                                                "price": stock_info.get("currentPrice", price),
                                                "source": "naver"
                                            })
                                            if len(matches) >= limit:
                                                break
        except Exception as e:
            print(f"네이버 금융 검색 중 오류: {str(e)}")
        
        # 2. 한국투자증권 종목검색 (실제 API 사용 시)
        try:
            # 한국투자증권 API 사용 예시 (실제로는 API 키가 필요)
            # 실제 구현에서는 한국투자증권 API를 사용하여 검색
            pass
            
        except Exception as e:
            print(f"한국투자증권 API 검색 중 오류: {str(e)}")
        
        # 3. 대안: 한국 주식 코드 데이터베이스 검색
        # 실제 구현에서는 한국 주식 목록을 데이터베이스나 파일에서 로드하여 검색
        korean_stock_database = {
            # 코스피 상위 종목들
            "005930": "삼성전자",
            "000660": "SK하이닉스", 
            "035420": "NAVER",
            "051910": "LG화학",
            "006400": "삼성SDI",
            "035720": "카카오",
            "068270": "셀트리온",
            "207940": "삼성바이오로직스",
            "323410": "카카오뱅크",
            "005380": "현대자동차",
            "000270": "기아",
            "051900": "LG생활건강",
            "034220": "LG디스플레이",
            "017670": "SK텔레콤",
            "096770": "SK이노베이션",
            "028260": "삼성물산",
            "032830": "삼성생명",
            "000810": "삼성화재",
            "016360": "삼성증권",
            "028050": "삼성엔지니어링",
        }
        
        # 검색어와 매칭되는 종목 찾기
        for code, name in korean_stock_database.items():
            if query.lower() in name.lower() or query.lower() in code.lower():
                try:
                    ticker = f"{code}.KS"
                    stock_info = await get_stock_info(ticker)
                    if "error" not in stock_info:
                        matches.append({
                            "symbol": ticker,
                            "name": stock_info.get("name", name),
                            "country": "한국",
                            "price": stock_info.get("currentPrice", "정보 없음"),
                            "source": "database"
                        })
                        if len(matches) >= limit:
                            break
                except Exception:
                    continue
        
        return matches
        
    except Exception as e:
        print(f"한국 주식 웹 검색 중 오류: {str(e)}")
        return []

async def search_stocks_alternative_api(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    대체 API를 사용한 주식 검색 (Alpha Vantage, IEX Cloud 등)
    """
    try:
        # Alpha Vantage API 사용 예시 (실제로는 API 키가 필요)
        # url = f"https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords={query}&apikey=YOUR_API_KEY"
        
        # IEX Cloud API 사용 예시
        # url = f"https://cloud.iexapis.com/stable/search/{query}?token=YOUR_API_KEY"
        
        # 임시로 빈 리스트 반환 (실제 구현 시 위 주석을 참고하여 구현)
        return []
        
    except Exception as e:
        print(f"대체 API 검색 중 오류: {str(e)}")
        return []

async def run_tests():
    """테스트 함수를 실행합니다."""
    print("테스트 시작...")
    
    # 삼성전자 주가 정보 조회
    print("\n1. 삼성전자 주가 정보:")
    result1 = await get_stock_price("005930.KS")
    print(result1)
    
    # 애플 최근 3개월 주가 히스토리 (일봉)
    print("\n2. 애플 최근 3개월 주가 히스토리:")
    result2 = await get_stock_history("AAPL", period="3mo", interval="1d")
    print(result2)
    
    # 마이크로소프트 어닝 정보
    print("\n3. 마이크로소프트 어닝 정보:")
    result3 = await get_earnings("MSFT")
    print(result3)
    
    # 삼성전자와 애플 성과 비교 (1년)
    print("\n4. 삼성전자와 애플 성과 비교:")
    result4 = await compare_stocks("005930.KS,AAPL", period="1y")
    print(result4)
    
    # "삼성" 관련 주식 검색
    print("\n5. '삼성' 관련 주식 검색:")
    result5 = await search_stocks("삼성", limit=5)
    print(result5)
    
    # 새로운 웹 검색 기능 테스트
    print("\n6. 웹 검색 기능 테스트:")
    print("6-1. 'Apple' 검색 (영어):")
    result6_1 = await search_stocks("Apple", limit=3)
    print(result6_1)
    
    print("\n6-2. 'Tesla' 검색 (영어):")
    result6_2 = await search_stocks("Tesla", limit=3)
    print(result6_2)
    
    print("\n6-3. 'Microsoft' 검색 (영어):")
    result6_3 = await search_stocks("Microsoft", limit=3)
    print(result6_3)
    
    # 회사명으로 주가 정보 가져오기 테스트
    print("\n7. 회사명으로 주가 정보 가져오기 테스트:")
    print("7-1. 'Apple' 주가 정보:")
    result7_1 = await get_stock_price("Apple")
    print(result7_1)
    
    print("\n7-2. '삼성전자' 주가 정보:")
    result7_2 = await get_stock_price("삼성전자")
    print(result7_2)
    
    print("\n7-3. 'Tesla' 주가 히스토리 (1개월):")
    result7_3 = await get_stock_history("Tesla", period="1mo")
    print(result7_3)
    
    print("\n7-4. 'Microsoft' 어닝 정보:")
    result7_4 = await get_earnings("Microsoft")
    print(result7_4)
    
    # 종목명으로 종목 비교 테스트
    print("\n8. 종목명으로 종목 비교 테스트:")
    print("8-1. 'Apple,Microsoft,Tesla' 비교:")
    result8_1 = await compare_stocks("Apple,Microsoft,Tesla", period="3mo")
    print(result8_1)
    
    print("\n8-2. '삼성전자,Apple,Microsoft' 비교:")
    result8_2 = await compare_stocks("삼성전자,Apple,Microsoft", period="1y")
    print(result8_2)
    
    # 공통 함수 테스트
    print("\n9. 공통 함수 테스트:")
    print("9-1. 'Apple' 티커 변환:")
    result9_1 = await resolve_ticker("Apple")
    print(result9_1)
    
    print("\n9-2. '삼성전자' 티커 변환:")
    result9_2 = await resolve_ticker("삼성전자")
    print(result9_2)
    
    print("\n9-3. 'AAPL' 티커 변환 (이미 티커인 경우):")
    result9_3 = await resolve_ticker("AAPL")
    print(result9_3)
    
    # 새로운 LLM 친화적 함수 테스트
    print("\n10. LLM 친화적 함수 테스트:")
    print("10-1. 다중 종목 주가 조회 ('삼성전자,하이닉스,애플'):")
    result10_1 = await get_multiple_stock_prices("삼성전자,하이닉스,Apple")
    print(result10_1)
    
    print("\n10-2. 종목 비교 분석 ('삼성전자,하이닉스'):")
    result10_2 = await analyze_stock_comparison("삼성전자,하이닉스", period="6mo")
    print(result10_2)
    
    print("\n10-3. 단일 종목 종합 분석 ('Apple'):")
    result10_3 = await get_stock_analysis("Apple")
    print(result10_3)
    
    # AMD 검색 테스트
    print("\n11. AMD 검색 테스트:")
    print("11-1. 'AMD' 검색:")
    result11_1 = await search_stocks("AMD", limit=3)
    print(result11_1)
    
    print("\n11-2. 'AMD' 티커 변환:")
    result11_2 = await resolve_ticker("AMD")
    print(result11_2)
    
    print("\n11-3. 'NVIDIA,AMD' 비교:")
    result11_3 = await compare_stocks("NVIDIA,AMD", period="1y")
    print(result11_3)
    
    print("\n11-4. 'NVDA,AMD' 비교:")
    result11_4 = await compare_stocks("NVDA,AMD", period="1y")
    print(result11_4)

@mcp.tool()
async def analyze_stock_comparison(stock_names: str, period: str = "1y") -> dict[str, Any]:
    """
    여러 종목을 분석하여 투자 리포트를 생성합니다.
    
    Args:
        stock_names: 쉼표로 구분된 종목명 목록 (예: '삼성전자,하이닉스,애플')
        period: 분석 기간 (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max 중 선택)
    """
    try:
        # 종목명 리스트로 변환
        stock_list = [name.strip() for name in stock_names.split(",")]
        
        if len(stock_list) < 2:
            return {"error": "비교하려면 최소 2개 이상의 종목이 필요합니다."}
        if len(stock_list) > 5:
            return {"error": "비교는 최대 5개 종목까지만 가능합니다."}
        
        # 모든 종목을 티커로 변환
        ticker_results = await resolve_multiple_tickers(stock_list)
        
        if not ticker_results["tickers"]:
            return {"error": "모든 종목에 대해 티커를 찾을 수 없습니다."}
        
        # 티커 목록 생성
        ticker_string = ",".join([item["ticker"] for item in ticker_results["tickers"]])
        
        # 종목 비교 실행
        comparison_result = await compare_stocks(ticker_string, period)
        
        # 에러가 있는 종목들 추가 정보
        if ticker_results["errors"]:
            comparison_result["warnings"] = ticker_results["errors"]
        
        return comparison_result
        
    except Exception as e:
        return {"error": f"종목 분석 중 오류가 발생했습니다: {str(e)}"}

@mcp.tool()
async def get_stock_analysis(stock_name: str) -> dict[str, Any]:
    """
    특정 종목에 대한 종합 분석을 제공합니다.
    
    Args:
        stock_name: 종목명 또는 티커 심볼 (예: '삼성전자', 'Apple', '005930.KS')
    """
    try:
        # 종목명을 티커로 변환
        ticker_info = await resolve_ticker(stock_name)
        
        if "error" in ticker_info:
            return {"error": ticker_info["error"]}
        
        ticker = ticker_info["ticker"]
        company_name = ticker_info["name"]
        
        # 1. 기본 주가 정보
        price_info = await get_stock_price(stock_name)
        
        # 2. 최근 3개월 히스토리
        history_info = await get_stock_history(stock_name, period="3mo")
        
        # 3. 어닝 정보
        earnings_info = await get_earnings(stock_name)
        
        # 종합 분석 결과 생성
        analysis_result = f"""
=== {company_name} ({ticker}) 종합 분석 ===

1. 기본 주가 정보:
{price_info.get('result', '정보 없음')}

2. 최근 3개월 주가 동향:
{history_info.get('result', '정보 없음')}

3. 실적 정보:
{earnings_info.get('result', '정보 없음')}

=== 분석 완료 ===
"""
        
        return {"result": analysis_result}
        
    except Exception as e:
        return {"error": f"종목 분석 중 오류가 발생했습니다: {str(e)}"}

@mcp.tool()
async def get_multiple_stock_prices(stock_names: str) -> dict[str, Any]:
    """
    여러 종목의 현재 주가를 한번에 조회합니다.
    
    Args:
        stock_names: 쉼표로 구분된 종목명 목록 (예: '삼성전자,하이닉스,애플,마이크로소프트')
    """
    try:
        # 종목명 리스트로 변환
        stock_list = [name.strip() for name in stock_names.split(",")]
        
        if len(stock_list) > 10:
            return {"error": "한번에 최대 10개 종목까지만 조회 가능합니다."}
        
        results = []
        errors = []
        
        for stock_name in stock_list:
            try:
                price_info = await get_stock_price(stock_name)
                if "error" in price_info:
                    errors.append(f"{stock_name}: {price_info['error']}")
                else:
                    results.append(f"{stock_name}: {price_info['result']}")
            except Exception as e:
                errors.append(f"{stock_name}: {str(e)}")
        
        # 결과 포맷팅
        result_text = f"=== 다중 종목 주가 정보 ===\n\n"
        
        if results:
            for i, result in enumerate(results, 1):
                result_text += f"{i}. {result}\n\n"
        
        if errors:
            result_text += "=== 오류 발생 종목 ===\n"
            for error in errors:
                result_text += f"- {error}\n"
        
        return {"result": result_text}
        
    except Exception as e:
        return {"error": f"다중 종목 조회 중 오류가 발생했습니다: {str(e)}"}

if __name__ == "__main__":
    # 모드 선택
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("테스트 모드로 실행합니다...")
        import asyncio
        asyncio.run(run_tests())
    elif len(sys.argv) > 1 and sys.argv[1] == "--amd-test":
        print("AMD 테스트 모드로 실행합니다...")
        import asyncio
        
        async def amd_test():
            print("1. AMD 직접 티커 테스트:")
            result1 = await get_stock_info("AMD")
            print(result1)
            
            print("\n2. AMD 검색 테스트:")
            result2 = await search_stocks("AMD", limit=3)
            print(result2)
            
            print("\n3. AMD resolve_ticker 테스트:")
            result3 = await resolve_ticker("AMD")
            print(result3)
            
            print("\n4. NVIDIA,AMD 비교 테스트:")
            result4 = await compare_stocks("NVIDIA,AMD", period="1y")
            print(result4)
            
            print("\n5. NVIDIA 검색 테스트:")
            result5 = await search_stocks("NVIDIA", limit=3)
            print(result5)
            
            print("\n6. NVIDIA resolve_ticker 테스트:")
            result6 = await resolve_ticker("NVIDIA")
            print(result6)
            
            print("\n7. NVDA,AMD 비교 테스트 (정확한 티커 사용):")
            result7 = await compare_stocks("NVDA,AMD", period="1y")    
        
        asyncio.run(amd_test())
    else:
        print("MCP 서버를 시작합니다...")
        mcp.run()