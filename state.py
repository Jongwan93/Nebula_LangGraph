# state.py
"""LangGraph state definition for the stock analysis pipeline (linear batch)."""

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    """State passed between pipeline nodes. Batch mode: all tickers at once."""

    # List of stock tickers to process
    tickers: list[str]
    # Combined price and news data for ALL tickers: {ticker: {prices, news, macro}}
    gathered_data: dict[str, Any]
    # Analysis results for ALL tickers: list of {date, ticker, predicted_change_pct, reason}
    analysis_results: list[dict[str, Any]]
    # Top 5 (or fewer) stocks with positive predicted_change_pct, sorted descending
    ranked_results: list[dict[str, Any]]
    # Google Sheet ID or title for output (optional)
    sheet_id: str
