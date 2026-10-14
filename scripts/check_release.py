"""Rebuild what the released files allow and compare it with the paper.

T-Drive and AIS per-query data are not distributed (see README), so this script works in a
temporary copy of the repository and

1. reruns the analysis steps on the released per-query data (Porto), then compares every
   rebuilt Porto row with the released result tables;
2. evaluates every number of the paper (the macros of paper/numbers.tex) twice: from the
   rebuilt tables, and from the released tables, while recording whether the computation
   used T-Drive or AIS values. Each number is then classed as
   - "rebuilt": recomputed from released per-query data (Porto, synthetic) and equal to
     the paper;
   - "from released aggregates": equal to the paper, but taken from released aggregate
     T-Drive/AIS results (or other aggregate experiment outputs) rather than recomputed;
   - "needs withheld data": needs T-Drive or AIS per-query data, which the experiment
     scripts rebuild from the original downloads.

Nothing in the repository is modified. A summary is printed; the per-macro status is
written to rebuilt/macro_status.csv.
"""
from __future__ import annotations

import contextlib
import csv
import importlib.util
import io
import json
import types
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
STEPS = [
    ["scripts/vldbj_merge_primary_native.py"],
    ["scripts/vldbj_task1_table.py"],
    ["scripts/vldbj_task1_table.py", "--clamp", "planner", "--subset", "all"],
    ["scripts/vldbj_task1_table.py", "--clamp", "planner", "--subset", "nonempty"],
    ["scripts/vldbj_native_decomposition.py", "--datasets", "porto"],
    ["scripts/vldbj_paired_bootstrap.py"],
    ["scripts/vldbj_plan_rerun_analysis.py", "--input-dir", "run/vldbj_rev/plan_porto_nolearned",
     "--dataset", "porto", "--conditions", "native", "trajectory_sampling", "true_cardinality_input"],
]
# rebuilt table -> columns that depend on the withheld datasets even for Porto rows
COMPARE = {
    "run/vldbj_rev/native/explain_estimates_primary.csv": (),
    "run/vldbj_rev/native/task1_accuracy_table.csv": (),
    "run/vldbj_rev/native/task1_accuracy_table__planner_all.csv": (),
    # the non-empty bootstrap draws continue the random stream after the T-Drive draws
    "run/vldbj_rev/native/task1_accuracy_table__planner_nonempty.csv":
        ("median_ci95_lower", "median_ci95_upper", "p95_ci95_lower", "p95_ci95_upper"),
    "run/vldbj_rev/native/task1_bucket_table.csv": (),
    "run/vldbj_rev/native/task1_bucket_table__planner_all.csv": (),
    "run/vldbj_rev/native/task1_bucket_table__planner_nonempty.csv": (),
    "run/vldbj_rev/native/task1_clamp_shares.csv": (),
    "run/vldbj_rev/native/native_decomposition.csv": (),
    # p-values and intervals use the full test families (Holm) and one random stream
    "run/vldbj_rev/stats/paired_bootstrap_clamped.csv":
        ("family_size", "p_holm", "p_raw", "ratio_ci95_lower", "ratio_ci95_upper", "significant_holm_0_05"),
    "run/vldbj_rev/plan_porto_nolearned/analysis/per_query_plan_metrics.csv": (),
    "run/vldbj_rev/plan_porto_nolearned/analysis/plan_bootstrap_intervals.csv": (),
    "run/vldbj_rev/plan_porto_nolearned/analysis/plan_metrics_summary.csv": (),
}


WITHHELD = ("tdrive", "ais")
ID_KEYS = {"dataset", "workload", "estimator", "form", "target", "original_target", "role", "subset",
           "family", "left", "right", "condition", "metric", "bucket", "setting", "scale",
           "training_budget", "resource_kind", "rows", "list"}


# result files that hold only T-Drive or AIS results (no dataset column to inspect)
WITHHELD_PATHS = ("/plan_rerun/", "/plan_retime_lowload/", "/ais/", "/task7b/", "/headline_retrain/",
                  "/task7_dataset_validity/", "tdrive")


class Usage:
    used = False


_open, _read_text = Path.open, Path.read_text


def _mark(path) -> None:
    if any(w in str(path) for w in WITHHELD_PATHS):
        Usage.used = True


def _tracked_open(self, *a, **k):
    _mark(self)
    return _open(self, *a, **k)


def _tracked_read_text(self, *a, **k):
    _mark(self)
    return _read_text(self, *a, **k)


class TrackedRow(dict):
    """CSV row that notes when a value (not an identifier) of a T-Drive/AIS row is read."""
    def __getitem__(self, key):
        if key not in ID_KEYS and dict.get(self, "dataset") in WITHHELD:
            Usage.used = True
        return dict.__getitem__(self, key)

    def get(self, key, default=None):
        return self[key] if key in self else default


def track(obj, flagged=False):
    if isinstance(obj, dict):
        return TrackedJSON({k: track(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [track(v) for v in obj]
    return obj


class TrackedJSON(dict):
    """JSON object that notes when a key naming T-Drive or AIS is read."""
    def __getitem__(self, key):
        if isinstance(key, str) and any(w in key for w in WITHHELD):
            Usage.used = True
        return dict.__getitem__(self, key)

    def get(self, key, default=None):
        return self[key] if key in self else default


def load_generator(root: Path):
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location(f"gen_{abs(hash(root))}", root / "scripts/vldbj_generate_numbers.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    sys.path.remove(str(root))
    plain_rows, plain_tsv = gen.rows, gen.rows_tsv
    gen.rows = lambda path: [TrackedRow(r) for r in plain_rows(path)]
    gen.rows_tsv = lambda path: [TrackedRow(r) for r in plain_tsv(path)]
    gen.json = types.SimpleNamespace(loads=lambda text: track(json.loads(text)), dumps=json.dumps)
    return gen


def evaluate(gen, name: str):
    Usage.used = False
    Path.open, Path.read_text = _tracked_open, _tracked_read_text
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return gen.MACROS[name](), Usage.used, None
    except Exception as exc:  # missing T-Drive/AIS per-query files or rows
        return None, Usage.used, f"{type(exc).__name__}: {exc}"[:200]
    finally:
        Path.open, Path.read_text = _open, _read_text


def rows(path: Path) -> list[dict]:
    with path.open() as fh:
        return [r for r in csv.DictReader(fh) if r.get("dataset", "porto") == "porto"]


def shipped_macros() -> dict[str, str]:
    text = (ROOT / "paper/numbers.tex").read_text()
    out, pos = {}, 0
    pattern = re.compile(r"\\newcommand\{\\(Num[A-Za-z]+)\}\{")
    while (m := pattern.search(text, pos)):
        depth, i = 1, m.end()
        while depth:
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        out[m.group(1)] = text[m.end():i - 1]
        pos = i
    return out


def main() -> None:
    work = Path(tempfile.mkdtemp(prefix="spatiotemp_check_"))
    try:
        for d in ("scripts", "src", "run", "paper"):
            shutil.copytree(ROOT / d, work / d)
        print("1. Rebuilding result tables from the released per-query data (Porto)")
        for step in STEPS:
            r = subprocess.run([PY, *step], cwd=work, capture_output=True, text=True)
            if r.returncode:
                raise SystemExit(f"{' '.join(step)} failed:\n{r.stderr}")
        n_rows = n_cols_skipped = 0
        for rel, skip in COMPARE.items():
            rebuilt, shipped = rows(work / rel), rows(ROOT / rel)
            if len(rebuilt) != len(shipped):
                raise SystemExit(f"{rel}: {len(rebuilt)} rebuilt Porto rows, {len(shipped)} released")
            for a, b in zip(rebuilt, shipped):
                for k in a:
                    if k not in skip and a[k] != b[k]:
                        raise SystemExit(f"{rel}: column {k} differs ({a[k]} vs {b[k]})")
            n_rows += len(rebuilt)
            note = f" (all columns except {', '.join(skip)})" if skip else ""
            print(f"   identical: {rel}, {len(rebuilt)} Porto rows{note}")
        print(f"   {n_rows} Porto rows in {len(COMPARE)} tables match the released results")

        print("2. Evaluating every number of the paper")
        rebuilt_gen, released_gen = load_generator(work), load_generator(ROOT)
        expected = shipped_macros()
        status = []
        for name in rebuilt_gen.macro_list():
            v1, used1, err1 = evaluate(rebuilt_gen, name)
            v2, used2, err2 = evaluate(released_gen, name)
            if err1 is None and not used1 and not used2 and v1 == expected[name]:
                state, detail = "rebuilt", ""
            else:
                if err2 is None and v2 == expected[name]:
                    state, detail = "from released aggregates", ""
                elif err2 is not None:
                    state, detail = "needs withheld data", err2
                else:
                    # the full data reproduce every number exactly, so a different value here
                    # comes from rows removed with the withheld data (e.g. a count over them)
                    state, detail = "needs withheld data", f"value without the withheld rows: {v2}"
            status.append({"macro": name, "status": state, "detail": detail})
        out = ROOT / "rebuilt"
        out.mkdir(exist_ok=True)
        with (out / "macro_status.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["macro", "status", "detail"], lineterminator="\n")
            w.writeheader()
            w.writerows(status)
        counts = {}
        for s in status:
            counts[s["status"]] = counts.get(s["status"], 0) + 1
        for k, v in sorted(counts.items()):
            print(f"   {v:4d} {k}")
        print("   per-macro status: rebuilt/macro_status.csv")
        if counts.get("DIFFERS"):
            for s in status:
                if s["status"] == "DIFFERS":
                    print(f"   DIFFERS: {s['macro']}: {s['detail']}")
            raise SystemExit(1)
    finally:
        shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(ROOT / "run/vldbj_rev/native/__pycache__", ignore_errors=True)


if __name__ == "__main__":
    main()
