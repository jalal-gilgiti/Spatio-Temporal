"""Plan-sensitive PostgreSQL evaluation primitives used by the plan study.

The served predicate retains exact ``atStbox`` semantics and adds ``&&`` only as a
logically necessary GiST prefilter.  Cardinalities are injected through the existing
pg_hint_plan one-row-dummy wrapper; this is an injection harness, not a native
MobilityDB selectivity hook.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

from psycopg import sql

from src.data_module.workload_generator import STPredicate, stbox_literal


_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_JOIN_NODES = {"Nested Loop", "Hash Join", "Merge Join"}
_FINGERPRINT_KEYS = (
    "Node Type", "Join Type", "Relation Name", "Alias", "Index Name",
    "Scan Direction", "Strategy", "Parent Relationship", "Partial Mode",
)


def _quoted_identifier(name: str) -> str:
    if not _IDENTIFIER.fullmatch(name):
        raise ValueError(f"unsafe SQL identifier: {name!r}")
    return sql.Identifier(name).as_string(None)


def exact_indexable_predicate(predicate: STPredicate, alias: str = "t") -> str:
    """Return a GiST-prefiltered predicate equivalent to the exact path test."""
    if not _IDENTIFIER.fullmatch(alias):
        raise ValueError(f"unsafe SQL alias: {alias!r}")
    box = stbox_literal(predicate)
    return (
        f"{alias}.traj && stbox '{box}' AND "
        f"atStbox({alias}.traj, stbox '{box}') IS NOT NULL"
    )


def build_plan_sensitive_query(
    table: str,
    metadata_table: str,
    predicate: STPredicate,
    estimated_rows: float | None,
) -> str:
    """Build the realistic one-to-one trip-metadata join used in the plan study."""
    table_sql = _quoted_identifier(table)
    metadata_sql = _quoted_identifier(metadata_table)
    base = (
        f"SELECT count(*) FROM {table_sql} t "
        "CROSS JOIN (SELECT 1 AS x OFFSET 0) AS dummy "
        f"JOIN {metadata_sql} m ON m.trajectory_id = t.trajectory_id "
        f"WHERE dummy.x = 1 AND m.active AND ({exact_indexable_predicate(predicate)})"
    )
    if estimated_rows is None:
        return base
    if not math.isfinite(estimated_rows):
        raise ValueError(f"estimated_rows must be finite, got {estimated_rows}")
    rows = max(1, round(estimated_rows))
    return f"/*+ Rows(t dummy #{rows}) */ {base}"


def normalized_plan_tree(node: dict[str, Any]) -> dict[str, Any]:
    """Strip volatile costs/timings/row counts while retaining physical choices."""
    normalized = {key: node[key] for key in _FINGERPRINT_KEYS if key in node}
    if node.get("Plans"):
        normalized["Plans"] = [normalized_plan_tree(child) for child in node["Plans"]]
    return normalized


def plan_fingerprint(plan: dict[str, Any]) -> str:
    payload = json.dumps(
        normalized_plan_tree(plan["Plan"]), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _relation_names(node: dict[str, Any]) -> set[str]:
    names = {node["Relation Name"]} if "Relation Name" in node else set()
    for child in node.get("Plans", []):
        names.update(_relation_names(child))
    return names


def _find_relation_scan(node: dict[str, Any], relation: str) -> dict[str, Any] | None:
    if node.get("Relation Name") == relation:
        return node
    for child in node.get("Plans", []):
        found = _find_relation_scan(child, relation)
        if found is not None:
            return found
    return None


def _find_core_join(
    node: dict[str, Any], source_table: str, metadata_table: str
) -> dict[str, Any] | None:
    # Children first returns the deepest join containing both real relations and skips
    # the outer one-row dummy Nested Loop.
    for child in node.get("Plans", []):
        found = _find_core_join(child, source_table, metadata_table)
        if found is not None:
            return found
    if node.get("Node Type") in _JOIN_NODES:
        names = _relation_names(node)
        if source_table in names and metadata_table in names:
            return node
    return None


def plan_summary(
    plan: dict[str, Any], source_table: str, metadata_table: str
) -> dict[str, Any]:
    root = plan["Plan"]
    scan = _find_relation_scan(root, source_table)
    join = _find_core_join(root, source_table, metadata_table)
    if scan is None or join is None:
        raise ValueError("plan does not contain the expected source scan and metadata join")
    children = join.get("Plans", [])
    outer_names = _relation_names(children[0]) if children else set()
    outer = (
        "source" if source_table in outer_names
        else "metadata" if metadata_table in outer_names
        else "other"
    )
    return {
        "fingerprint": plan_fingerprint(plan),
        "source_scan": scan["Node Type"],
        "source_index": scan.get("Index Name"),
        "source_planned_rows": int(scan["Plan Rows"]),
        "metadata_join": join["Node Type"],
        "join_outer": outer,
        "plan_family": f"{scan['Node Type']} + {join['Node Type']} ({outer} outer)",
        "total_cost": float(root["Total Cost"]),
    }


def root_buffer_counts(plan: dict[str, Any]) -> dict[str, int]:
    root = plan["Plan"]
    keys = (
        "Shared Hit Blocks", "Shared Read Blocks", "Shared Dirtied Blocks",
        "Shared Written Blocks", "Local Hit Blocks", "Local Read Blocks",
        "Temp Read Blocks", "Temp Written Blocks",
    )
    return {key: int(root.get(key, 0)) for key in keys}
