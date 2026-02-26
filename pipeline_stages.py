# pipeline_stages.py
"""Run pipeline stages separately with persisted state (read/write JSON in pipeline_state/)."""

import asyncio
from typing import Any

from nodes import (
    analyst_node,
    gather_data_node,
    ranking_node,
    select_ticker_node,
    sheets_writer_node,
)
from state import AgentState
from state_io import (
    load_analysis_results,
    load_gathered_data,
    load_ranked_results,
    save_analysis_results,
    save_gathered_data,
    save_ranked_results,
)
from tools import write_results_to_new_sheet


def run_stage_gather(tickers: list[str]) -> dict[str, Any]:
    """
    Stage 1: Normalize tickers, run gather_data_node once for all tickers. Save gathered_data.json.
    Returns final state (gathered_data key).
    """
    state: AgentState = {
        "tickers": list(tickers),
        "gathered_data": {},
        "analysis_results": [],
        "ranked_results": [],
        "sheet_id": "",
    }
    update = select_ticker_node(state)
    state = {**state, **update}
    update = asyncio.run(gather_data_node(state))
    state = {**state, **update}
    out_path = save_gathered_data(state.get("gathered_data") or {})
    print(f"  Saved gathered_data to {out_path}", flush=True)
    return state


def run_stage_analyze(
    create_new_sheet: bool = False, sheet_name: str | None = None
) -> list[dict[str, Any]]:
    """
    Stage 2: Load gathered_data.json, run analyst_node once for all tickers, save analysis_results.json.
    If create_new_sheet is True, writes all analysis results to a new Google Sheet.
    Returns analysis_results list.
    """
    gathered_data = load_gathered_data()
    if not gathered_data:
        print("  No gathered data to analyze.", flush=True)
        return []
    state: AgentState = {
        "tickers": list(gathered_data.keys()),
        "gathered_data": gathered_data,
        "analysis_results": [],
        "ranked_results": [],
        "sheet_id": "",
    }
    update = asyncio.run(analyst_node(state))
    analysis_results = update.get("analysis_results") or []
    out_path = save_analysis_results(analysis_results)
    print(f"  Saved analysis_results to {out_path}", flush=True)

    # Write to new sheet if requested
    if create_new_sheet and analysis_results:
        if sheet_name is None:
            from datetime import datetime

            sheet_name = (
                f"Stock Analysis {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
        try:
            sheet_id = write_results_to_new_sheet(sheet_name, analysis_results)
            print(f"  Created new Google Sheet: {sheet_name}")
            print(f"  Sheet ID: {sheet_id}")
            print(f"  Sheet URL: https://docs.google.com/spreadsheets/d/{sheet_id}")
        except Exception as e:
            print(f"  Warning: Failed to create Google Sheet: {e}", flush=True)

    return analysis_results


def run_stage_rank() -> list[dict[str, Any]]:
    """
    Stage 3: Load analysis_results.json, run ranking_node, save ranked_results.json.
    Returns ranked_results list.
    """
    analysis_results = load_analysis_results()
    state: AgentState = {"analysis_results": analysis_results, "ranked_results": []}
    update = ranking_node(state)
    ranked = update.get("ranked_results") or []
    out_path = save_ranked_results(ranked)
    print(f"  Saved ranked_results to {out_path}", flush=True)
    return ranked


def run_stage_sheets(sheet_id: str) -> None:
    """
    Stage 4: Load ranked_results.json, run sheets_writer_node.
    """
    ranked = load_ranked_results()
    if not ranked:
        print("  No ranked results to write.", flush=True)
        return
    state: AgentState = {"ranked_results": ranked, "sheet_id": sheet_id}
    sheets_writer_node(state)
    print("  Wrote ranked results to Google Sheet.", flush=True)
