# nodes.py
"""LangGraph node logic: batch data gathering, LLM analysis, ranking, and Google Sheets write."""

import asyncio
import logging
import os
from datetime import datetime
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from state import AgentState
from tools import (
    append_results_to_sheet,
    fetch_stock_data,
    search_news_and_macro,
    write_results_to_new_sheet,
)

logger = logging.getLogger(__name__)

# Max concurrent operations per batch node
CONCURRENCY_LIMIT = 5
MAX_RETRIES = 3

# LLM prompt must include the required analysis instruction (English equivalent)
ANALYST_SYSTEM = """You are a stock analyst. Based on the given data, output a short prediction.

Required instruction: Based on recent news, price fluctuations, and economic macros of this stock, predict what percentage the price will change in exactly one week and provide a short explanation of the reason.

Respond in JSON only, with exactly these keys:
- "predicted_change_pct": a number (e.g. 2.5 for +2.5%, -1.2 for -1.2%)
- "reason": short explanation in one or two sentences."""

ANALYST_USER = """Ticker: {ticker}

Price/data summary:
{price_summary}

News and macro:
{news_macro}

Output JSON only (predicted_change_pct, reason)."""


def _format_price_summary(data: dict[str, Any]) -> str:
    """Format price data for the LLM. data may be {prices: <fetch_stock_data result>, news, macro}."""
    if not data:
        return "No price data available."
    raw = data.get("prices")
    if isinstance(raw, dict) and raw.get("error"):
        return "No price data available."
    p = raw.get("prices", raw) if isinstance(raw, dict) else raw
    if not p or not isinstance(p, dict):
        return "No price data available."
    parts = [
        f"Current: {p.get('current')}",
        f"14d high: {p.get('high_14d')}",
        f"14d low: {p.get('low_14d')}",
    ]
    if p.get("closes"):
        parts.append(f"Recent closes: {p['closes'][-7:]}")
    return "\n".join(parts)


def _format_news_macro(data: dict[str, Any]) -> str:
    """Format news and macro for the LLM."""
    if not data:
        return "No news or macro data."
    lines = []
    if data.get("news"):
        lines.append("News: " + " | ".join(data["news"][:5]))
    if data.get("macro"):
        lines.append("Macro: " + " | ".join(data["macro"][:3]))
    return "\n".join(lines) if lines else "No news or macro data."


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
    reraise=True,
)
def _fetch_one_ticker_sync(ticker: str) -> dict[str, Any]:
    """
    Fetch price (yfinance) and news/macro (Tavily) for one ticker. Used in thread pool.
    Raises after MAX_RETRIES for transient errors; caller should catch and skip.
    """
    gathered: dict[str, Any] = {}
    try:
        price_result = fetch_stock_data.invoke({"ticker": ticker, "days_back": 30})
        if isinstance(price_result, dict):
            gathered["prices"] = price_result
        else:
            gathered["prices"] = {"ticker": ticker, "error": str(price_result)}
    except Exception as e:
        gathered["prices"] = {"ticker": ticker, "error": str(e)}
    try:
        news_macro = search_news_and_macro(ticker, include_macro=True)
        gathered["news"] = news_macro.get("news", [])
        gathered["macro"] = news_macro.get("macro", [])
        if not gathered["news"] and not gathered["macro"]:
            logger.warning(f"No news/macro data retrieved for {ticker}")
    except Exception as e:
        logger.exception(f"Failed to fetch news/macro for {ticker}: {e}")
        gathered["news"] = [f"Error: {e}"]
        gathered["macro"] = [f"Error: {e}"]
    return gathered


def select_ticker_node(state: AgentState) -> dict[str, Any]:
    """
    Batch mode: normalize ticker list and pass through. No popping; tickers stay for gather_data.
    """
    tickers = state.get("tickers") or []
    normalized = [(t or "").strip().upper() for t in tickers if (t or "").strip()]
    return {"tickers": normalized}


async def gather_data_node(state: AgentState) -> dict[str, Any]:
    """
    Fetch data for ALL tickers with max CONCURRENCY_LIMIT (5) at a time.
    Uses asyncio.gather(return_exceptions=True); failed tickers are logged and skipped.
    """
    tickers = state.get("tickers") or []
    if not tickers:
        return {"gathered_data": {}}

    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)

    async def fetch_one(ticker: str) -> tuple[str, Any]:
        async with sem:
            try:
                loop = asyncio.get_event_loop()
                data = await loop.run_in_executor(None, _fetch_one_ticker_sync, ticker)
                return (ticker, data)
            except Exception as e:
                logger.exception("Gather failed for %s after retries: %s", ticker, e)
                print(f"  [Skip] {ticker}: gather failed after retries.", flush=True)
                return (ticker, None)

    results = await asyncio.gather(
        *[fetch_one(t) for t in tickers],
        return_exceptions=True,
    )

    gathered_data: dict[str, Any] = {}
    for r in results:
        if isinstance(r, Exception):
            continue
        ticker, data = r
        if data is not None:
            gathered_data[ticker] = data
            print(f"  Gathered: {ticker}", flush=True)

    return {"gathered_data": gathered_data}


def _extract_text_from_llm_response(raw: Any) -> str:
    """
    Extract the actual string content from an LLM response that may be str, list, or dict.
    If dict, use the 'text' key. If list, extract 'text' from each item if dict, else str(item).
    """
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, dict):
        return (
            (raw.get("text") or "").strip()
            if isinstance(raw.get("text"), str)
            else str(raw).strip()
        )
    if isinstance(raw, list):
        parts = []
        for item in raw:
            if isinstance(item, dict):
                part = (
                    item.get("text") if isinstance(item.get("text"), str) else str(item)
                )
            else:
                part = str(item)
            parts.append(part)
        return " ".join(parts).strip()
    return str(raw).strip()


def _parse_llm_output(text: str | list, ticker: str, today: str) -> dict[str, Any]:
    """
    Parse and validate LLM JSON into a single result dict for sheets_writer.
    Ensures keys: date, ticker, predicted_change_pct (float), reason (str).
    """
    import ast
    import json
    import re

    text = _extract_text_from_llm_response(text)
    if not text:
        return {
            "date": today,
            "ticker": ticker,
            "predicted_change_pct": 0.0,
            "reason": "Could not parse LLM output (empty).",
        }

    def find_json_block(s: str) -> str:
        start = s.find("{")
        if start == -1:
            return s
        depth = 0
        for i in range(start, len(s)):
            if s[i] == "{":
                depth += 1
            elif s[i] == "}":
                depth -= 1
                if depth == 0:
                    return s[start : i + 1]
        return s[start:]

    block = find_json_block(text)
    if "```" in block:
        block = find_json_block(block.replace("```json", "").replace("```", ""))

    try:
        parsed = json.loads(block)
    except json.JSONDecodeError:
        normalized = re.sub(r"'([^']*)'(\s*[:,\]])", r'"\1"\2', block)
        try:
            parsed = json.loads(normalized)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(block)
            except (ValueError, SyntaxError):
                return {
                    "date": today,
                    "ticker": ticker,
                    "predicted_change_pct": 0.0,
                    "reason": "Could not parse LLM output.",
                }

    if not isinstance(parsed, dict):
        return {
            "date": today,
            "ticker": ticker,
            "predicted_change_pct": 0.0,
            "reason": "Could not parse LLM output.",
        }
    try:
        pct = float(parsed.get("predicted_change_pct", 0))
    except (TypeError, ValueError):
        pct = 0.0
    reason = str(parsed.get("reason", "") or "").strip() or "No reason provided."
    return {
        "date": today,
        "ticker": ticker,
        "predicted_change_pct": pct,
        "reason": reason,
    }


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
    reraise=True,
)
def _analyze_one_ticker_sync(
    ticker: str,
    data: dict[str, Any],
    chain: Any,
    today: str,
) -> dict[str, Any]:
    """Run LLM for one ticker (sync). Used in executor or with ainvoke."""
    price_summary = _format_price_summary(data)
    news_macro = _format_news_macro(data)
    try:
        msg = chain.invoke(
            {
                "ticker": ticker,
                "price_summary": price_summary,
                "news_macro": news_macro,
            }
        )
        raw = msg.content if hasattr(msg, "content") else msg
        text = _extract_text_from_llm_response(raw)
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error analyzing {ticker}: {error_msg}", exc_info=True)
        return {
            "date": today,
            "ticker": ticker,
            "predicted_change_pct": 0.0,
            "reason": f"Analysis error: {error_msg}",
        }
    return _parse_llm_output(text, ticker, today)


async def analyst_node(state: AgentState) -> dict[str, Any]:
    """
    Run LLM for ALL tickers in parallel (max CONCURRENCY_LIMIT concurrent).
    Uses asyncio.gather(return_exceptions=True); failed tickers are logged and skipped.
    """
    gathered_data = state.get("gathered_data") or {}
    if not gathered_data:
        return {"analysis_results": []}

    # Check for API key before creating LLM
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError(
            "DEEPSEEK_API_KEY not set in environment. "
            "Set it in your .env file or export it as an environment variable. "
            "Get your API key from https://platform.deepseek.com/"
        )

    # Deepseek uses OpenAI-compatible API with custom base_url
    llm = ChatOpenAI(
        model="deepseek-reasoner",
        temperature=0,
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ANALYST_SYSTEM),
            ("human", ANALYST_USER),
        ]
    )
    chain = prompt | llm
    today = datetime.now().strftime("%Y-%m-%d")

    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)
    loop = asyncio.get_event_loop()

    async def analyze_one(ticker: str, data: dict[str, Any]) -> dict[str, Any] | None:
        async with sem:
            # Skip when there is no valid price data so we don't call the LLM with empty input
            prices = data.get("prices")
            if isinstance(prices, dict) and prices.get("error"):
                print(f"  [Skip] {ticker}: No valid price data to analyze.", flush=True)
                return None

            try:
                result = await loop.run_in_executor(
                    None,
                    lambda t=ticker, d=data: _analyze_one_ticker_sync(
                        t, d, chain, today
                    ),
                )
                reason = result.get("reason", "")
                # Show full error message if there's an error, otherwise truncate for success
                if (
                    result.get("predicted_change_pct", 0) == 0.0
                    and "Analysis error" in reason
                ):
                    # Show full error message for debugging
                    print(
                        f"  -> {result['ticker']}: {result['predicted_change_pct']}% | {reason}",
                        flush=True,
                    )
                else:
                    # Truncate only for successful predictions
                    print(
                        f"  -> {result['ticker']}: {result['predicted_change_pct']}% | {reason[:60]}...",
                        flush=True,
                    )
                return result
            except Exception as e:
                logger.exception("Analyze failed for %s after retries: %s", ticker, e)
                print(f"  [Skip] {ticker}: analysis failed after retries.", flush=True)
                return {
                    "date": today,
                    "ticker": ticker,
                    "predicted_change_pct": 0.0,
                    "reason": f"Error: {e}",
                }

    results = await asyncio.gather(
        *[analyze_one(ticker, data) for ticker, data in gathered_data.items()],
        return_exceptions=True,
    )

    analysis_results: list[dict[str, Any]] = []
    for r in results:
        if isinstance(r, Exception):
            continue
        if isinstance(r, dict):
            analysis_results.append(r)

    return {"analysis_results": analysis_results}


TOP_N = 5


def ranking_node(state: AgentState) -> dict[str, Any]:
    """
    Filter to positive predicted_change_pct, sort descending, take top TOP_N.
    If fewer than 5 positive, write whatever is available.
    """
    results = state.get("analysis_results") or []
    positive = [r for r in results if (r.get("predicted_change_pct") or 0) > 0]
    positive.sort(key=lambda r: r.get("predicted_change_pct", 0), reverse=True)
    ranked = positive[:TOP_N]
    print(
        f"  Ranked {len(ranked)} positive predictions (top 5 written to sheet).",
        flush=True,
    )
    return {"ranked_results": ranked}


def write_all_analysis_to_sheet_node(state: AgentState) -> dict[str, Any]:
    """
    Write ALL analysis results to a new Google Sheet (before ranking).
    Creates a new sheet with timestamp name if sheet_name not provided in state.
    """
    from datetime import datetime

    analysis_results = state.get("analysis_results") or []
    if not analysis_results:
        return {}

    # Get sheet name from state or use timestamp
    sheet_name = state.get("analysis_sheet_name")
    if not sheet_name:
        sheet_name = f"Stock Analysis {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    try:
        sheet_id = write_results_to_new_sheet(sheet_name, analysis_results)
        print(f"  Created new Google Sheet with all analysis results: {sheet_name}")
        print(f"  Sheet ID: {sheet_id}")
        print(f"  Sheet URL: https://docs.google.com/spreadsheets/d/{sheet_id}")
        return {}
    except Exception as e:
        print(
            f"  Warning: Failed to create Google Sheet for analysis results: {e}",
            flush=True,
        )
        return {}


def sheets_writer_node(state: AgentState) -> dict[str, Any]:
    """
    Write ranked top 5 (or fewer) results to Google Sheet in one operation.
    """
    import os

    ranked = state.get("ranked_results") or []
    if not ranked:
        return {}

    sheet_id = state.get("sheet_id") or os.getenv("GOOGLE_SHEET_ID", "")
    if not sheet_id:
        return {}

    try:
        append_results_to_sheet(sheet_id, ranked)
        return {}
    except Exception as e:
        raise RuntimeError(f"Google Sheets write failed: {e}") from e
