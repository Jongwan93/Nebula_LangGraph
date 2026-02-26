# graph.py
"""LangGraph workflow: linear batch pipeline (select -> gather -> analyst -> ranking -> sheets)."""

import asyncio

from langgraph.graph import END, StateGraph

from state import AgentState
from nodes import (
    select_ticker_node,
    gather_data_node,
    analyst_node,
    ranking_node,
    sheets_writer_node,
)


def build_graph():
    """
    Build the compiled LangGraph pipeline.
    Linear path: select_ticker -> gather_data -> analyst -> ranking -> sheets_writer -> END.
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("select_ticker", select_ticker_node)
    workflow.add_node("gather_data", gather_data_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("ranking", ranking_node)
    workflow.add_node("sheets_writer", sheets_writer_node)

    workflow.set_entry_point("select_ticker")
    workflow.add_edge("select_ticker", "gather_data")
    workflow.add_edge("gather_data", "analyst")
    workflow.add_edge("analyst", "ranking")
    workflow.add_edge("ranking", "sheets_writer")
    workflow.add_edge("sheets_writer", END)

    return workflow.compile()


def run_pipeline(
    tickers: list[str],
    sheet_id: str = "",
) -> dict:
    """
    Run the full batch pipeline for the given tickers.
    Uses ainvoke so async nodes (gather_data, analyst) run correctly.
    sheet_id: optional Google Sheet ID or title for output.
    Returns final state (including analysis_results, ranked_results).
    """
    graph = build_graph()
    initial: AgentState = {
        "tickers": tickers,
        "gathered_data": {},
        "analysis_results": [],
        "ranked_results": [],
        "sheet_id": sheet_id,
    }
    config = {"configurable": {}}

    async def _run():
        return await graph.ainvoke(initial, config=config)

    final_state = asyncio.run(_run())
    return final_state
