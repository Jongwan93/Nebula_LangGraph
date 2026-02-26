# tools.py
"""Helper functions for yfinance, Tavily, and Google Sheets."""

import os
from datetime import datetime, timedelta
from typing import Any

import yfinance as yf
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

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
    import logging

    logger = logging.getLogger(__name__)

    # Check if API key is set
    if not os.getenv("TAVILY_API_KEY"):
        error_msg = "TAVILY_API_KEY not set in environment"
        logger.error(error_msg)
        return {"news": [f"Error: {error_msg}"], "macro": [f"Error: {error_msg}"]}

    try:
        tavily = get_tavily_tool(max_results=5)
    except Exception as e:
        error_msg = f"Failed to initialize Tavily tool: {e}"
        logger.error(error_msg)
        return {"news": [f"Error: {error_msg}"], "macro": [f"Error: {error_msg}"]}

    out: dict[str, Any] = {"news": [], "macro": []}

    # Stock-specific news
    try:
        news_result = tavily.invoke({"query": f"latest news {ticker} stock earnings revenue"})
        logger.info(
            f"Tavily news result for {ticker}: type={type(news_result)}, value={str(news_result)[:500]}"
        )

        # Handle different response formats
        if isinstance(news_result, list):
            out["news"] = [r.get("content", str(r)) for r in news_result[:5] if r]
        elif isinstance(news_result, dict):
            # Check for error first
            if "error" in news_result:
                error = news_result["error"]
                error_msg = str(error) if isinstance(error, Exception) else str(error)
                logger.error(f"Tavily API error for {ticker}: {error_msg}")
                out["news"] = [f"Tavily API error: {error_msg}"]
            # Check for results array
            elif "results" in news_result and isinstance(news_result["results"], list):
                out["news"] = [
                    r.get("content", r.get("raw_content", str(r)))
                    for r in news_result["results"][:5]
                    if r
                ]
                # Check for answer field
                if "answer" in news_result and news_result["answer"]:
                    out["news"].insert(0, news_result["answer"])
            # Check if it's a direct content field
            elif "content" in news_result:
                out["news"] = [news_result["content"]]
        elif isinstance(news_result, str):
            # Sometimes Tavily returns a string directly
            out["news"] = [news_result]

        if not out["news"]:
            logger.warning(
                f"No news results returned from Tavily for {ticker}. Raw result: {news_result}"
            )
    except Exception as e:
        logger.exception(f"Tavily news search error for {ticker}: {e}")
        out["news"] = [f"Tavily error: {e}"]

    if include_macro:
        try:
            macro_result = tavily.invoke(
                {"query": "US economic macro data inflation Fed interest rates latest"}
            )
            logger.info(
                f"Tavily macro result: type={type(macro_result)}, value={str(macro_result)[:500]}"
            )

            # Handle different response formats
            if isinstance(macro_result, list):
                out["macro"] = [r.get("content", str(r)) for r in macro_result[:3] if r]
            elif isinstance(macro_result, dict):
                # Check for error first
                if "error" in macro_result:
                    error = macro_result["error"]
                    error_msg = str(error) if isinstance(error, Exception) else str(error)
                    logger.error(f"Tavily API error for macro: {error_msg}")
                    out["macro"] = [f"Tavily API error: {error_msg}"]
                elif "results" in macro_result and isinstance(macro_result["results"], list):
                    out["macro"] = [
                        r.get("content", r.get("raw_content", str(r)))
                        for r in macro_result["results"][:3]
                        if r
                    ]
                    if "answer" in macro_result and macro_result["answer"]:
                        out["macro"].insert(0, macro_result["answer"])
                elif "content" in macro_result:
                    out["macro"] = [macro_result["content"]]
            elif isinstance(macro_result, str):
                out["macro"] = [macro_result]

            if not out["macro"]:
                logger.warning(f"No macro results returned from Tavily. Raw result: {macro_result}")
        except Exception as e:
            logger.exception(f"Tavily macro search error: {e}")
            out["macro"] = [f"Macro search error: {e}"]

    return out


# ---------------------------------------------------------------------------
# Google Sheets via gspread (service account)
# ---------------------------------------------------------------------------


def get_sheets_client():
    """Return gspread client authenticated with service_account.json."""
    import gspread
    from google.oauth2.service_account import Credentials

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "service_account.json")
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


def write_results_to_new_sheet(
    sheet_name: str,
    rows: list[dict[str, Any]],
    columns: list[str] | None = None,
) -> str:
    """
    Create a new Google Sheet and write analysis results to it.
    Returns the sheet ID of the created sheet.
    """
    if columns is None:
        columns = ["date", "stock ticker", "% change in stock price", "reason"]
    client = get_sheets_client()

    # Create a new spreadsheet
    sheet = client.create(sheet_name)
    worksheet = sheet.sheet1

    # Write header row
    worksheet.update("A1:D1", [columns])

    # Write all data rows
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

    if row_data:
        worksheet.update(f"A2:D{len(row_data) + 1}", row_data)

    return sheet.id


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
