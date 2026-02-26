# state_io.py
"""Persist pipeline state between stages (JSON in pipeline_state/)."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


def _default_serializer(obj: Any) -> Any:
    """Convert non-JSON-serializable values for state persistence."""
    if hasattr(obj, "item") and callable(getattr(obj, "item", None)):  # numpy scalar
        return obj.item()
    if hasattr(obj, "tolist") and callable(getattr(obj, "tolist", None)):  # numpy array
        return obj.tolist()
    if isinstance(obj, (datetime,)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _get_state_dir() -> Path:
    """Return pipeline state directory; create if missing."""
    path = Path(os.getenv("PIPELINE_STATE_DIR", "pipeline_state"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_gathered_data(gathered_data: dict[str, Any], path: str | Path | None = None) -> Path:
    """Save gathered_data to JSON. Returns path used."""
    dir_path = _get_state_dir()
    file_path = path or dir_path / "gathered_data.json"
    file_path = Path(file_path)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(gathered_data, f, default=_default_serializer, indent=2, ensure_ascii=False)
    return file_path


def load_gathered_data(path: str | Path | None = None) -> dict[str, Any]:
    """Load gathered_data from JSON."""
    file_path = path or _get_state_dir() / "gathered_data.json"
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"State file not found: {file_path}. Run stage 'gather' first.")
    with open(file_path, encoding="utf-8") as f:
        return json.load(f)


def save_analysis_results(analysis_results: list[dict[str, Any]], path: str | Path | None = None) -> Path:
    """Save analysis_results to JSON. Returns path used."""
    dir_path = _get_state_dir()
    file_path = path or dir_path / "analysis_results.json"
    file_path = Path(file_path)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(analysis_results, f, default=_default_serializer, indent=2, ensure_ascii=False)
    return file_path


def load_analysis_results(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load analysis_results from JSON."""
    file_path = path or _get_state_dir() / "analysis_results.json"
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"State file not found: {file_path}. Run stage 'analyze' first.")
    with open(file_path, encoding="utf-8") as f:
        return json.load(f)


def save_ranked_results(ranked_results: list[dict[str, Any]], path: str | Path | None = None) -> Path:
    """Save ranked_results to JSON. Returns path used."""
    dir_path = _get_state_dir()
    file_path = path or dir_path / "ranked_results.json"
    file_path = Path(file_path)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(ranked_results, f, default=_default_serializer, indent=2, ensure_ascii=False)
    return file_path


def load_ranked_results(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load ranked_results from JSON."""
    file_path = path or _get_state_dir() / "ranked_results.json"
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"State file not found: {file_path}. Run stage 'rank' first.")
    with open(file_path, encoding="utf-8") as f:
        return json.load(f)
