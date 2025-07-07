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

def is_ticker(symbol: str) -> bool:
    """
    심볼이 티커인지 판별 (한국/미국)
    Check if the symbol is a ticker (KR/US)
    """
    symbol = symbol.strip()
    # 한국 주식은 .KS 또는 .KQ로 끝나야 함
    if symbol.endswith('.KS') or symbol.endswith('.KQ'):
        return True
    # 미국 주식은 대문자이고 1-5글자이면서 숫자가 포함되어 있거나 특정 패턴을 가져야 함
    if symbol.isupper() and 1 <= len(symbol) <= 5:
        # 숫자가 포함되어 있거나 특정 패턴인 경우만 티커로 인식
        if any(char.isdigit() for char in symbol) or symbol in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'AMD', 'INTC']:
            return True
    return False

async def resolve_tickers(
    ticker_or_names: Union[str, List[str]]
) -> Union[Dict[str, Any], Dict[str, List[Dict[str, Any]]]]:
    """
    종목명 또는 티커(들)를 받아 실제 티커와 회사명으로 변환합니다.
    (단일 입력/복수 입력 모두 지원)
    
    Given stock name(s) or ticker(s), resolve to actual ticker and company name.
    (Supports both single and multiple input)
    
    Args:
        ticker_or_names (str or List[str]): 종목명/티커 또는 그 리스트
    Returns:
        dict: {"ticker": ..., "name": ...} 또는 {"tickers": [...], "errors": [...]}
    """
    if isinstance(ticker_or_names, str):
        ticker_or_name = ticker_or_names.strip()
        
        # 이미 티커인 경우
        if is_ticker(ticker_or_name):
            return {"ticker": ticker_or_name, "name": ticker_or_name}
        
        try:
            search_result = await search_stocks(ticker_or_name, limit=1)
            if "error" not in search_result:
                # 검색 결과에서 첫 번째 항목 추출
                lines = search_result["result"].split("\n")
                for line in lines:
                    if "(" in line and ")" in line:
                        # "1. Company Name (TICKER)" 형식에서 티커 추출
                        ticker_start = line.find("(") + 1
                        ticker_end = line.find(")")
                        if ticker_start > 0 and ticker_end > ticker_start:
                            ticker = line[ticker_start:ticker_end]
                            company_name = line.split("(")[0].split(". ")[-1].strip()
                            return {"ticker": ticker, "name": company_name}
                return {"error": f"No ticker found for '{ticker_or_name}'"}
            else:
                return {"error": search_result["error"]}
        except Exception as e:
            return {"error": f"Error occurred during ticker resolution: {str(e)}"}
    # 복수 입력
    results = {"tickers": [], "errors": []}
    for item in ticker_or_names:
        info = await resolve_tickers(item)
        if "error" in info:
            results["errors"].append(f"{item}: {info['error']}")
        else:
            results["tickers"].append({"ticker": info["ticker"], "name": info["name"]})
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
            print(f"Error processing ticker {ticker}: {str(e)}")
            return {"error": f"Error occurred while fetching information: {str(e)}"}
            
    except Exception as e:
        print(f"티커 {ticker} 처리 중 오류: {str(e)}")
        return {"error": f"Error occurred while fetching information: {str(e)}"}

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
async def get_stock_price(ticker_or_name: str) -> dict[str, Any]:
    """
    특정 종목의 현재 주가 정보를 가져옵니다. (회사 이름 또는 티커 심볼 입력 가능)
    Get the current stock price info for a given company name or ticker symbol.

    Args:
        ticker_or_name (str): 회사 이름 또는 주식 티커 심볼
    Returns:
        dict: 주식의 상세 정보 (현재가, 거래량, 시가총액 등)
    """
    try:
        ticker_info = await resolve_tickers(ticker_or_name)
        if "error" in ticker_info:
            return {"error": ticker_info["error"]}
        ticker = ticker_info["ticker"]
        company_name = ticker_info["name"]
        stock_info = await get_stock_info(ticker)
        if "error" in stock_info:
            return {"error": f"Error: {stock_info['error']}"}
        
        result = f"\nStock Information: {stock_info.get('name', company_name)} ({ticker})\n"
        result += f"Country: {stock_info.get('country', 'Unknown')}\n"
        result += f"Sector: {stock_info.get('sector', 'Unknown')}\n"
        result += f"Industry: {stock_info.get('industry', 'Unknown')}\n\n"
        
        result += f"Current Price: {stock_info.get('currentPrice', 'N/A')} {stock_info.get('currency', '')}\n"
        result += f"Previous Close: {stock_info.get('previousClose', 'N/A')} {stock_info.get('currency', '')}\n"
        result += f"Open: {stock_info.get('open', 'N/A')} {stock_info.get('currency', '')}\n"
        result += f"Day Low: {stock_info.get('dayLow', 'N/A')} {stock_info.get('currency', '')}\n"
        result += f"Day High: {stock_info.get('dayHigh', 'N/A')} {stock_info.get('currency', '')}\n"
        result += f"Volume: {stock_info.get('volume', 'N/A')}\n\n"
        
        result += f"Market Cap: {stock_info.get('marketCap', 'N/A')}\n"
        result += f"52 Week Low: {stock_info.get('fiftyTwoWeekLow', 'N/A')} {stock_info.get('currency', '')}\n"
        result += f"52 Week High: {stock_info.get('fiftyTwoWeekHigh', 'N/A')} {stock_info.get('currency', '')}\n"
        result += f"Forward P/E: {stock_info.get('forwardPE', 'N/A')}\n"
        result += f"Trailing P/E: {stock_info.get('trailingPE', 'N/A')}\n"
        result += f"Dividend: {stock_info.get('dividendRate', 'N/A')} {stock_info.get('currency', '')}\n"
        result += f"Dividend Yield: {stock_info.get('dividendYield', 'N/A')}%\n"
        
        return {"result": result}
    except Exception as e:
        return {"error": f"Error occurred while fetching stock price: {str(e)}"}

@mcp.tool()
async def get_stock_history(ticker_or_name: str, period: str = "1mo", interval: str = "1d") -> dict[str, Any]:
    """
    특정 종목의 주가 히스토리를 가져옵니다.
    Get historical stock price data for a given company name or ticker symbol.

    Args:
        ticker_or_name (str): 주식 티커 심볼 또는 회사명
        period (str): 기간 (e.g. '1w', '1mo', '1y', ...)
        interval (str): 간격 (e.g. '1d', '1wk', ...)
    Returns:
        dict: 주가 히스토리 및 통계 정보
    """
    try:
        if period == "1w":
            period = "5d"
        valid_periods = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
        valid_intervals = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]
        if period not in valid_periods:
            return {"error": f"Invalid period. Please choose from: {', '.join(valid_periods)}"}
        if interval not in valid_intervals:
            return {"error": f"Invalid interval. Please choose from: {', '.join(valid_intervals)}"}
        ticker_info = await resolve_tickers(ticker_or_name)
        if "error" in ticker_info:
            return {"error": ticker_info["error"]}
        ticker = ticker_info["ticker"]
        company_name = ticker_info["name"]
        stock = yf.Ticker(ticker)
        history = stock.history(period=period, interval=interval)
        if history.empty:
            return {"error": f"No history data available for {company_name} ({ticker})"}
        formatted_data = format_history_data(history)
        info = await get_stock_info(ticker)
        name = info.get("name", company_name)
        result = f"{name} ({ticker}) Stock History (Period: {period}, Interval: {interval})\n\n"
        for entry in formatted_data[-5:]:
            result += f"Date: {entry['date']}, Open: {entry['open']}, High: {entry['high']}, Low: {entry['low']}, Close: {entry['close']}, Volume: {entry['volume']}\n"
        close_prices = [entry["close"] for entry in formatted_data if entry["close"] is not None]
        if close_prices:
            avg_price = sum(close_prices) / len(close_prices)
            min_price = min(close_prices)
            max_price = max(close_prices)
            result += f"\nAverage Close Price: {round(avg_price, 2)}"
            result += f"\nLowest Close Price: {min_price}"
            result += f"\nHighest Close Price: {max_price}"
            if len(close_prices) >= 2:
                first_price = close_prices[0]
                last_price = close_prices[-1]
                change = last_price - first_price
                change_pct = (change / first_price) * 100
                result += f"\nPrice Change: {round(change, 2)} ({round(change_pct, 2)}%)"
        return {"result": result}
    except Exception as e:
        return {"error": f"Error occurred while fetching stock history: {str(e)}"}

@mcp.tool()
async def get_earnings(ticker_or_name: str) -> dict[str, Any]:
    """
    특정 종목의 실적(어닝) 정보를 가져옵니다.
    Get earnings (financial results) info for a given company name or ticker symbol.

    Args:
        ticker_or_name (str): 주식 티커 심볼 또는 회사명
    Returns:
        dict: 어닝/재무제표 정보
    """
    try:
        ticker_info = await resolve_tickers(ticker_or_name)
        if "error" in ticker_info:
            return {"error": ticker_info["error"]}
        ticker = ticker_info["ticker"]
        company_name = ticker_info["name"]
        stock = yf.Ticker(ticker)
        info = await get_stock_info(ticker)
        name = info.get("name", company_name)
        earnings_dates = None
        try:
            earnings_dates = stock.earnings_dates
        except Exception:
            pass
        if earnings_dates is None or (hasattr(earnings_dates, 'empty') and earnings_dates.empty):
            recent_earnings = "No earnings date information available."
        else:
            recent_earnings = "Recent Earnings Information:\n"
            try:
                for i, (date, row) in enumerate(earnings_dates.iloc[:4].iterrows()):
                    eps_est = row.get('EPS Estimate', None)
                    eps_actual = row.get('Reported EPS', None)
                    surprise = row.get('Surprise(%)', None)
                    recent_earnings += f"{date.strftime('%Y-%m-%d')}: "
                    recent_earnings += f"Expected EPS: {eps_est if eps_est is not None and not pd.isna(eps_est) else 'N/A'}, "
                    recent_earnings += f"Actual EPS: {eps_actual if eps_actual is not None and not pd.isna(eps_actual) else 'N/A'}, "
                    recent_earnings += f"Surprise: {surprise}%\n" if surprise is not None and not pd.isna(surprise) else "Surprise: N/A\n"
            except Exception as e:
                recent_earnings += f"Error processing earnings dates: {str(e)}\n"
        income_stmt = None
        try:
            income_stmt = stock.income_stmt
        except Exception:
            pass
        quarterly_info = "No quarterly earnings information available."
        if income_stmt is not None:
            try:
                if isinstance(income_stmt, pd.DataFrame):
                    quarterly_info = "\nQuarterly Earnings (Last 4 Quarters):\n"
                    recent_quarters = income_stmt.columns[:4]
                    for date in recent_quarters:
                        revenue = income_stmt.loc['Total Revenue', date] if 'Total Revenue' in income_stmt.index else None
                        net_income = income_stmt.loc['Net Income', date] if 'Net Income' in income_stmt.index else None
                        formatted_date = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
                        quarterly_info += f"{formatted_date}: "
                        quarterly_info += f"Revenue: {revenue:,.0f}" if revenue is not None and not pd.isna(revenue) else "Revenue: N/A"
                        quarterly_info += f", Net Income: {net_income:,.0f}\n" if net_income is not None and not pd.isna(net_income) else ", Net Income: N/A\n"
                elif isinstance(income_stmt, dict):
                    quarterly_info = "\nQuarterly Earnings (Last 4 Quarters):\n"
                    recent_quarters = list(income_stmt.keys())[:4]
                    for date in recent_quarters:
                        date_data = income_stmt[date]
                        revenue = date_data.get('Total Revenue', None)
                        net_income = date_data.get('Net Income', None)
                        formatted_date = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
                        quarterly_info += f"{formatted_date}: "
                        quarterly_info += f"Revenue: {revenue:,.0f}" if revenue is not None and not pd.isna(revenue) else "Revenue: N/A"
                        quarterly_info += f", Net Income: {net_income:,.0f}\n" if net_income is not None and not pd.isna(net_income) else ", Net Income: N/A\n"
            except Exception as e:
                quarterly_info = f"Error processing quarterly earnings information: {str(e)}"
        balance_sheet = None
        cash_flow = None
        try:
            balance_sheet = stock.balance_sheet
            cash_flow = stock.cashflow
        except Exception:
            pass
        annual_info = "No annual earnings information available."
        if balance_sheet is not None or cash_flow is not None:
            try:
                annual_info = "\nAnnual Earnings (Latest Information):\n"
                if isinstance(balance_sheet, pd.DataFrame) and not balance_sheet.empty:
                    latest_year = balance_sheet.columns[0]
                    year = latest_year.year if hasattr(latest_year, 'year') else str(latest_year).split('-')[0]
                    total_assets = balance_sheet.loc['Total Assets', latest_year] if 'Total Assets' in balance_sheet.index else None
                    total_liabilities = balance_sheet.loc['Total Liabilities Net Minority Interest', latest_year] if 'Total Liabilities Net Minority Interest' in balance_sheet.index else None
                    annual_info += f"{year} Total Assets: {total_assets:,.0f}" if total_assets is not None and not pd.isna(total_assets) else f"{year} Total Assets: N/A"
                    annual_info += f", Total Liabilities: {total_liabilities:,.0f}\n" if total_liabilities is not None and not pd.isna(total_liabilities) else ", Total Liabilities: N/A\n"
                if isinstance(cash_flow, pd.DataFrame) and not cash_flow.empty:
                    latest_year = cash_flow.columns[0]
                    year = latest_year.year if hasattr(latest_year, 'year') else str(latest_year).split('-')[0]
                    operating_cash_flow = cash_flow.loc['Operating Cash Flow', latest_year] if 'Operating Cash Flow' in cash_flow.index else None
                    investing_cash_flow = cash_flow.loc['Investing Cash Flow', latest_year] if 'Investing Cash Flow' in cash_flow.index else None
                    annual_info += f"{year} Operating Cash Flow: {operating_cash_flow:,.0f}" if operating_cash_flow is not None and not pd.isna(operating_cash_flow) else f"{year} Operating Cash Flow: N/A"
                    annual_info += f", Investing Cash Flow: {investing_cash_flow:,.0f}\n" if investing_cash_flow is not None and not pd.isna(investing_cash_flow) else ", Investing Cash Flow: N/A\n"
            except Exception as e:
                annual_info += f"Error processing annual earnings information: {str(e)}"
        next_earnings = "No next earnings date information available."
        try:
            next_earnings_date = stock.calendar
            if next_earnings_date is not None and hasattr(next_earnings_date, 'empty') and not next_earnings_date.empty:
                next_date = next_earnings_date.iloc[0, 0] if len(next_earnings_date) > 0 and len(next_earnings_date.columns) > 0 else None
                next_earnings = f"\nNext Earnings Date: {next_date.strftime('%Y-%m-%d') if next_date is not None and not pd.isna(next_date) else 'N/A'}"
        except Exception as e:
            next_earnings = f"\nError processing next earnings date: {str(e)}"
        result = f"{name} ({ticker}) Earnings Information\n\n{recent_earnings}\n{quarterly_info}\n{annual_info}\n{next_earnings}"
        return {"result": result}
    except Exception as e:
        return {"error": f"Error occurred while fetching earnings information: {str(e)}"}

@mcp.tool()
async def search_stocks(query: str, limit: int = 5) -> dict[str, Any]:
    """
    주식 심볼을 검색합니다.
    Search for stock symbols.

    Args:
        query (str): 검색어 (회사 이름이나 심볼의 일부)
        limit (int): 결과 제한 개수 (기본값: 5)
    Returns:
        dict: 검색 결과 목록
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
                                "price": stock_info.get("currentPrice", "N/A"),
                                "exchange": exchange
                            })
                            print(f"    성공: {ticker} 추가됨")
                        else:
                            print(f"    실패: {ticker} - {stock_info.get('error', '알 수 없는 오류')}")
                    except Exception as e:
                        # 개별 결과 처리 중 오류가 발생해도 계속 진행
                        print(f"Error processing individual search result ({ticker}): {str(e)}")
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
                            result += f"    Country: {match['country'] or 'Unknown'}\n"
                            if 'exchange' in match:
                                result += f"    Exchange: {match['exchange']}\n"
                            result += f"    Current Price: {match['price'] if match['price'] else 'N/A'}\n\n"
                        
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
                "NAVER": "035420.KS",
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
            
            # 한국어 검색어인 경우 한국 주식 매핑을 우선적으로 확인
            if any(ord(char) > 127 for char in query):  # 한글이 포함된 경우
                for company, ticker in korean_stocks.items():
                    if query.lower() in company.lower() or company.lower() in query.lower():
                        kr_info = await get_stock_info(ticker)
                        if "error" not in kr_info:
                            # 이미 추가된 심볼인지 확인
                            if not any(m["symbol"] == kr_info.get("symbol", ticker) for m in matches):
                                matches.append({
                                    "symbol": kr_info.get("symbol", ticker),
                                    "name": kr_info.get("name", company),
                                    "country": "South Korea",
                                    "price": kr_info.get("currentPrice", "N/A")
                                })
                                if len(matches) >= limit:
                                    break
        
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
                        "country": "South Korea",
                        "price": kr_info.get("currentPrice", "N/A")
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
                        "country": "South Korea",
                        "price": kq_info.get("currentPrice", "N/A")
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
                            "country": "United States",
                            "price": us_info.get("currentPrice", "N/A")
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
                            "country": "United States",
                            "price": us_info.get("currentPrice", "N/A")
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
            return {"error": f"No search results found for '{query}'. Please enter a valid stock name or symbol."}
        
        result = f"Search results for '{query}':\n\n"
        for i, match in enumerate(unique_matches, 1):
            result += f"{i}. {match['name']} ({match['symbol']})\n"
            result += f"   Country: {match['country'] or 'Unknown'}\n"
            if 'exchange' in match:
                result += f"   Exchange: {match['exchange']}\n"
            result += f"   Current Price: {match['price'] if match['price'] else 'N/A'}\n\n"
        
        return {"result": result}
    except Exception as e:
        return {"error": f"Error occurred while searching stocks: {str(e)}"}

@mcp.tool()
async def compare_stocks(tickers: str, period: str = "1y") -> dict[str, Any]:
    """
    여러 종목의 성과를 비교합니다.
    Compare the performance of multiple stocks.

    Args:
        tickers (str): 쉼표로 구분된 종목명 또는 티커 심볼 목록
        period (str): 비교 기간 (e.g. '1w', '1y', '6mo', ...)
    Returns:
        dict: 비교 결과 및 통계
    """
    try:
        valid_periods = ["1d", "5d", "1w", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
        if period not in valid_periods:
            return {"error": f"Invalid period. Please choose from: {', '.join(valid_periods)}"}
        ticker_list = [t.strip() for t in tickers.split(",")]
        if len(ticker_list) < 2:
            return {"error": "At least 2 stocks are required for comparison."}
        if len(ticker_list) > 5:
            return {"error": "Comparison is limited to 5 stocks maximum."}
        results = []
        for ticker_or_name in ticker_list:
            try:
                ticker_info = await resolve_tickers(ticker_or_name)
                if "error" in ticker_info:
                    results.append({
                        "ticker": ticker_or_name,
                        "name": ticker_or_name,
                        "error": ticker_info["error"]
                    })
                    continue
                ticker = ticker_info["ticker"]
                company_name = ticker_info["name"]
                stock = yf.Ticker(ticker)
                history = stock.history(period=period)
                if history.empty:
                    results.append({
                        "ticker": ticker,
                        "name": company_name,
                        "error": "No data available"
                    })
                    continue
                info = await get_stock_info(ticker)
                name = info.get("name", company_name)
                close_prices = history["Close"].dropna()
                if len(close_prices) < 2:
                    results.append({
                        "ticker": ticker,
                        "name": name,
                        "error": "Not enough data"
                    })
                    continue
                first_price = close_prices.iloc[0]
                last_price = close_prices.iloc[-1]
                change = last_price - first_price
                change_pct = (change / first_price) * 100
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
                    "ticker": ticker_or_name,
                    "name": ticker_or_name,
                    "error": str(e)
                })
        result = f"Stock Comparison Results (Period: {period})\n\n"
        valid_results = [r for r in results if "error" not in r]
        error_results = [r for r in results if "error" in r]
        if valid_results:
            valid_results.sort(key=lambda x: x["change_pct"], reverse=True)
            result += "=== Performance Ranking ===\n"
            for i, r in enumerate(valid_results, 1):
                result += f"{i}. {r['name']} ({r['ticker']})\n"
                result += f"   Change: {r['change_pct']:+.2f}% ({r['change']:+.2f})\n"
                result += f"   Start Price: {r['first_price']} → End Price: {r['last_price']}\n"
                result += f"   Average Price: {r['avg_price']}, Volatility: {r['volatility']:.2f}\n"
                result += f"   Low: {r['min_price']}, High: {r['max_price']}\n\n"
        if error_results:
            result += "=== Stocks with Errors ===\n"
            for r in error_results:
                result += f"- {r['ticker']}: {r['error']}\n"
        return {"result": result}
    except Exception as e:
        return {"error": f"Error occurred while comparing stocks: {str(e)}"}

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
                                                "country": "South Korea",
                                                "price": stock_info.get("currentPrice", price),
                                                "source": "naver"
                                            })
                                            if len(matches) >= limit:
                                                break
        except Exception as e:
            print(f"Error in Naver Finance search: {str(e)}")
        
        # 2. 한국투자증권 종목검색 (실제 API 사용 시)
        try:
            # 한국투자증권 API 사용 예시 (실제로는 API 키가 필요)
            # 실제 구현에서는 한국투자증권 API를 사용하여 검색
            pass
            
        except Exception as e:
            print(f"Error in Korea Investment Securities API search: {str(e)}")
        
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
                            "country": "South Korea",
                            "price": stock_info.get("currentPrice", "N/A"),
                            "source": "database"
                        })
                        if len(matches) >= limit:
                            break
                except Exception:
                    continue
        
        return matches
        
    except Exception as e:
        print(f"Error in Korean stock web search: {str(e)}")
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
        print(f"Error in alternative API search: {str(e)}")
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
    result9_1 = await resolve_tickers("Apple")
    print(result9_1)
    
    print("\n9-2. '삼성전자' 티커 변환:")
    result9_2 = await resolve_tickers("삼성전자")
    print(result9_2)
    
    print("\n9-3. 'AAPL' 티커 변환 (이미 티커인 경우):")
    result9_3 = await resolve_tickers("AAPL")
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
    result11_2 = await resolve_tickers("AMD")
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
    Analyze and compare multiple stocks to generate an investment report.

    Args:
        stock_names (str): 쉼표로 구분된 종목명 목록
        period (str): 분석 기간 (e.g. '1w', '1y', '6mo', ...)
    Returns:
        dict: 비교 분석 결과
    """
    try:
        stock_list = [name.strip() for name in stock_names.split(",")]
        if len(stock_list) < 2:
            return {"error": "At least 2 stocks are required for comparison."}
        if len(stock_list) > 5:
            return {"error": "Comparison is limited to 5 stocks maximum."}
        ticker_results = await resolve_tickers(stock_list)
        if "error" in ticker_results:
            return {"error": ticker_results["error"]}
        ticker_string = ",".join([item["ticker"] for item in ticker_results["tickers"]])
        comparison_result = await compare_stocks(ticker_string, period)
        if ticker_results["errors"]:
            comparison_result["warnings"] = ticker_results["errors"]
        return comparison_result
    except Exception as e:
        return {"error": f"Error occurred while analyzing stocks: {str(e)}"}

@mcp.tool()
async def get_stock_analysis(stock_name: str) -> dict[str, Any]:
    """
    특정 종목에 대한 종합 분석을 제공합니다.
    Provide a comprehensive analysis for a given stock.

    Args:
        stock_name (str): 종목명 또는 티커 심볼
    Returns:
        dict: 종합 분석 결과
    """
    try:
        ticker_info = await resolve_tickers(stock_name)
        if "error" in ticker_info:
            return {"error": ticker_info["error"]}
        ticker = ticker_info["ticker"]
        company_name = ticker_info["name"]
        price_info = await get_stock_price(stock_name)
        history_info = await get_stock_history(stock_name, period="3mo")
        earnings_info = await get_earnings(stock_name)
        analysis_result = f"""
=== {company_name} ({ticker}) Comprehensive Analysis ===

1. Basic Stock Information:
{price_info.get('result', 'N/A')}

2. Recent 3-Month Price Trend:
{history_info.get('result', 'N/A')}

3. Earnings Information:
{earnings_info.get('result', 'N/A')}

=== Analysis Complete ===
"""
        return {"result": analysis_result}
    except Exception as e:
        return {"error": f"Error occurred while analyzing stock: {str(e)}"}

@mcp.tool()
async def get_multiple_stock_prices(stock_names: str) -> dict[str, Any]:
    """
    여러 종목의 현재 주가를 한번에 조회합니다.
    Get the current prices of multiple stocks at once.

    Args:
        stock_names (str): 쉼표로 구분된 종목명 목록
    Returns:
        dict: 다중 종목 주가 정보
    """
    try:
        stock_list = [name.strip() for name in stock_names.split(",")]
        if len(stock_list) > 10:
            return {"error": "Maximum 10 stocks can be queried at once."}
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
        result_text = f"=== Multiple Stock Price Information ===\n\n"
        if results:
            for i, result in enumerate(results, 1):
                result_text += f"{i}. {result}\n\n"
        if errors:
            result_text += "=== Stocks with Errors ===\n"
            for error in errors:
                result_text += f"- {error}\n"
        return {"result": result_text}
    except Exception as e:
        return {"error": f"Error occurred while fetching multiple stock prices: {str(e)}"}

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
            result3 = await resolve_tickers("AMD")
            print(result3)
            
            print("\n4. NVIDIA,AMD 비교 테스트:")
            result4 = await compare_stocks("NVIDIA,AMD", period="1y")
            print(result4)
            
            print("\n5. NVIDIA 검색 테스트:")
            result5 = await search_stocks("NVIDIA", limit=3)
            print(result5)
            
            print("\n6. NVIDIA resolve_ticker 테스트:")
            result6 = await resolve_tickers("NVIDIA")
            print(result6)
            
            print("\n7. NVDA,AMD 비교 테스트 (정확한 티커 사용):")
            result7 = await compare_stocks("NVDA,AMD", period="1y")    
        
        asyncio.run(amd_test())
    else:
        print("MCP 서버를 시작합니다...")
        mcp.run()