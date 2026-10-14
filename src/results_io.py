"""`results.json` serialization for `run_smoke_pipeline`'s output (split
out of `run_pipeline.py`, kept a separate module from the Hydra wiring/
top-level orchestration in `run_pipeline.py` and the DB-backed step logic in
`src.pipeline_steps`, so each file stays inside this project's 200-400 line
budget).

The only public entry point is `write_results_json`; the rest are the
JSON-safety helpers it composes, kept private since nothing outside this
module needs them directly.
"""
import dataclasses
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np

from src.integration_module.plan_capture import PlanResult

__all__ = ["write_results_json"]


def _scan_nodes(node: dict) -> list[dict]:
    """Recursively collect every scan node of an EXPLAIN plan subtree, as
    ``{"node_type", "relation", "index"}`` rows in plan order."""
    scans: list[dict] = []
    node_type = node.get("Node Type", "")
    if "Scan" in node_type:
        scans.append({
            "node_type": node_type,
            "relation": node.get("Relation Name"),
            "index": node.get("Index Name"),
        })
    for child in node.get("Plans", []):
        scans.extend(_scan_nodes(child))
    return scans


def _plan_result_summary(pr: PlanResult) -> dict:
    """Summarize a `PlanResult` for JSON output.

    The raw `plan_json` EXPLAIN (ANALYZE, FORMAT JSON) tree is replaced by a
    compact plan-shape summary rather than dropped: which access method the
    planner chose under each estimator's hinted cardinality IS the
    headline signal of the systems-impact evaluation, so a `results.json` with only latency/row scalars
    cannot answer that question. The full tree is still omitted
    (it is deeply nested and mostly per-node timing noise); what is kept is
    ``top_node_type`` (with `build_hinted_query`'s join wrapper this is a
    join node, e.g. ``Nested Loop``) and one row per scan node
    (``Seq Scan`` vs ``Index Scan``/``Bitmap Heap Scan``/..., with relation
    and index name where present) -- i.e. the plan *choice*, which is what
    changes when a cardinality hint changes.
    """
    plan_node = pr.plan_json.get("Plan", {}) if isinstance(pr.plan_json, dict) else {}
    return {
        "actual_latency_ms": pr.actual_latency_ms,
        "estimated_rows": pr.estimated_rows,
        "actual_rows": pr.actual_rows,
        "top_node_type": plan_node.get("Node Type"),
        "scan_nodes": _scan_nodes(plan_node),
    }


def _result_to_json_safe(result: dict) -> dict:
    """Make `run_smoke_pipeline`'s result dict JSON-serializable.

    Every other branch of `result` (`accuracy`, `correlation_sweep`,
    `design_guidance`, and the scalar fields of `systems_impact`) is already
    built from native ``float``/``int``/``str``/``dict``/``list`` values --
    every eval-module function in this codebase explicitly casts numpy
    scalars to Python floats before returning (see e.g. `evaluate_accuracy`,
    `measure_correlation_strength`). The one exception is
    `systems_impact[name]["plan_results"]`, a list of `PlanResult` dataclass
    instances, which `json.dump` cannot serialize on its own.
    """
    safe = dict(result)
    if "systems_impact" in safe:
        safe["systems_impact"] = {
            name: {
                **{k: v for k, v in stats.items() if k != "plan_results"},
                "plan_results": [_plan_result_summary(pr) for pr in stats.get("plan_results", [])],
            }
            for name, stats in safe["systems_impact"].items()
        }
    return safe


def _json_default(obj: object) -> object:
    """Defense-in-depth `json.dump(default=...)` handler: `_result_to_json_safe`
    already strips the one known non-serializable type (`PlanResult`), but
    this catches any other type that might otherwise slip through a future
    eval-module change instead of crashing the whole run at the very last
    step. `evaluate_temporal_drift`'s ``table_start``/
    ``table_end`` provenance fields are raw ``datetime`` objects (see
    `src.eval_module.robustness._fetch_table_time_bounds`), which
    `json.dump` cannot serialize on its own either -- handled here via
    isoformat rather than in `_result_to_json_safe`, since a raw
    ``datetime`` can in principle appear anywhere in the result tree, not
    only at one known key path the way `PlanResult` does."""
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_results_json(result: dict, results_path: Path) -> None:
    """Serialize `run_smoke_pipeline`'s result to ``results_path``.

    Serialization happens fully in memory before anything is opened for
    writing, so a `TypeError` from an unserializable value leaves no
    half-written `results.json`; the finished text is then written to a
    sibling temp file and `os.replace`d into place, which is atomic on POSIX
    -- a reader never observes a truncated results file.
    """
    payload = json.dumps(_result_to_json_safe(result), default=_json_default, indent=2)
    tmp_path = results_path.with_suffix(results_path.suffix + ".tmp")
    tmp_path.write_text(payload)
    os.replace(tmp_path, results_path)
