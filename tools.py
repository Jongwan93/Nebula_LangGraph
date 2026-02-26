# tools.py
"""Helper functions for yfinance, Tavily, and Google Sheets."""

import os
from datetime import datetime, timedelta
from typing import Any

import yfinance as yf
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain_core.tools import tool

load_dotenv()


# ---------------------------------------------------------------------------
# yfinance: historical price and financial metrics
# ---------------------------------------------------------------------------


@tool
def fetch_stock_data(ticker: str, days_back: int = 30) -> dict[str, Any]:
    """
    Fetch historical price data and key metrics for a stock using yfinance.
    Uses at least 30 days of data for recent price fluctuation context.
    Returns OHLCV summary and recent price stats.
    """
    days_back = max(30, int(days_back))
    sym = yf.Ticker(ticker)
    end = datetime.now()
    start = end - timedelta(days=days_back)
    hist = sym.history(start=start, end=end)
    info = sym.info

    if hist.empty:
        return {"ticker": ticker, "error": "No history", "prices": None, "info": {}}

    recent = hist.tail(14)
    prices = {
        "current": float(hist["Close"].iloc[-1]) if len(hist) else None,
        "high_14d": float(recent["High"].max()) if len(recent) else None,
        "low_14d": float(recent["Low"].min()) if len(recent) else None,
        "volume_avg": float(hist["Volume"].mean()) if "Volume" in hist.columns else None,
        "last_dates": hist.index.strftime("%Y-%m-%d").tolist()[-5:],
        "closes": hist["Close"].tolist()[-14:],
    }
    return {
        "ticker": ticker,
        "prices": prices,
        "info": {
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "marketCap": info.get("marketCap"),
        },
    }


# ---------------------------------------------------------------------------
# Tavily: news and economic macro
# ---------------------------------------------------------------------------

_tavily_tool: TavilySearch | None = None


def get_tavily_tool(max_results: int = 5) -> TavilySearch:
    """Return a configured TavilySearch tool (singleton). Uses TAVILY_API_KEY from env."""
    global _tavily_tool
    if _tavily_tool is None:
        if not os.getenv("TAVILY_API_KEY"):
            raise ValueError("TAVILY_API_KEY not set in environment")
        _tavily_tool = TavilySearch(
            max_results=max_results,
            search_depth="advanced",
            include_answer=True,
        )
    return _tavily_tool


def search_news_and_macro(ticker: str, include_macro: bool = True) -> dict[str, Any]:
    """
    Use Tavily to fetch recent news for the ticker and optional macro/economic context.
    Returns structured dict with 'news' and optionally 'macro' snippets.
    """
    tavily = get_tavily_tool(max_results=5)
    out: dict[str, Any] = {"news": [], "macro": []}

    # Stock-specific news
    try:
        news_result = tavily.invoke(
            {"query": f"latest news {ticker} stock earnings revenue"}
        )
        if isinstance(news_result, list):
            out["news"] = [r.get("content", str(r)) for r in news_result[:5]]
        elif isinstance(news_result, dict):
            if "results" in news_result:
                out["news"] = [
                    r.get("content", r.get("raw_content", str(r)))
                    for r in news_result["results"][:5]
                ]
            if "answer" in news_result and news_result["answer"]:
                out["news"].insert(0, news_result["answer"])
    except Exception as e:
        out["news"] = [f"Tavily error: {e}"]

    if include_macro:
        try:
            macro_result = tavily.invoke(
                {"query": "US economic macro data inflation Fed interest rates latest"}
            )
            if isinstance(macro_result, list):
                out["macro"] = [r.get("content", str(r)) for r in macro_result[:3]]
            elif isinstance(macro_result, dict) and "results" in macro_result:
                out["macro"] = [
                    r.get("content", r.get("raw_content", str(r)))
                    for r in macro_result["results"][:3]
                ]
        except Exception as e:
            out["macro"] = [f"Macro search error: {e}"]

    return out


# ---------------------------------------------------------------------------
# Google Sheets via gspread (service account)
# ---------------------------------------------------------------------------


def get_sheets_client():
    """Return gspread client authenticated with service_account.json."""
    import gspread
    from google.oauth2.service_account import Credentials

    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "service_account.json"
    )
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "service_account.json not found in project root. "
            "Create it from Google Cloud Console for Sheets API."
        )
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_file(path, scopes=scopes)
    return gspread.authorize(creds)


def append_results_to_sheet(
    sheet_name_or_id: str,
    rows: list[dict[str, Any]],
    columns: list[str] | None = None,
) -> int:
    """
    Append analysis results to a Google Sheet.
    columns: [date, stock ticker, % change in stock price, reason]
    Returns number of rows appended.
    """
    if columns is None:
        columns = ["date", "stock ticker", "% change in stock price", "reason"]
    client = get_sheets_client()
    try:
        sheet = client.open_by_key(sheet_name_or_id)
    except Exception:
        sheet = client.open(sheet_name_or_id)
    worksheet = sheet.sheet1

    # Ensure header row
    existing = worksheet.get_all_values()
    if not existing or existing[0] != columns:
        worksheet.update("A1:D1", [columns])
        existing = worksheet.get_all_values()

    row_data = []
    for r in rows:
        row_data.append(
            [
                r.get("date", ""),
                r.get("ticker", r.get("stock ticker", "")),
                r.get("predicted_change_pct", r.get("% change in stock price", "")),
                r.get("reason", ""),
            ]
        )
    if not row_data:
        return 0
    next_row = len(existing) + 1 if existing else 2
    worksheet.update(f"A{next_row}:D{next_row + len(row_data) - 1}", row_data)
    return len(row_data)
