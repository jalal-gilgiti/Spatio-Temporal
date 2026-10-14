"""Design-guidance synthesis: turn raw correlation and robustness results into a
decision framework, not just numbers.

`build_design_guidance` is a pure function -- no DB access, no estimator
training -- over already-computed result dicts from `run_correlation_sweep`,
`evaluate_accuracy`-derived per-dataset summaries, and
the robustness/failure-analysis functions. It summarizes observed
comparisons without turning the lower boundary of a sweep into an unsupported
deployment threshold.

All correlation/MI values this module consumes and reports are on the
`adjusted_mutual_info_score` (AMI) scale (see
`correlation_analysis.measure_correlation_strength`), not raw
`mutual_info_score`'s unbounded-nats scale. AMI is ~0 for independent
space-time data regardless of sample size and rises toward 1.0 as they
become more dependent; this codebase's own measured sweep found AMI values
around 0.0-0.08 even at "full" nominal correlation strength, so a computed
`correlation_threshold_mi` anywhere near 1.0 would be implausible for real
pipeline data -- small, and possibly near-zero, is the expected regime.
"""


def build_design_guidance(
    correlation_sweep_results: list[dict],
    accuracy_by_dataset: dict[str, dict],
    robustness_by_dataset: dict[str, dict],
) -> dict:
    """Synthesize a coupling-vs-spatial-only decision framework from raw
    correlation-sweep, per-dataset accuracy and robustness results.

    Args:
        correlation_sweep_results: `run_correlation_sweep` rows,
            each with at least ``"measured_mi"``, ``"coupled_median_qerror"``,
            and ``"spatial_only_median_qerror"``. May be empty (e.g. no
            sweep was run) -- handled without raising.
        accuracy_by_dataset: Per-dataset accuracy summaries, keyed by dataset name, each with at least
            ``"coupled_median_qerror"`` and ``"spatial_only_median_qerror"``.
            May be empty. Note: this exact shape is the expected input for
            a caller not yet written -- no current function assembles
            per-dataset accuracy dicts in this form; it is documented here
            as the target interface.
        robustness_by_dataset: Per-dataset robustness/failure-analysis
            results (e.g. `evaluate_temporal_drift` or
            `analyze_transfer_failure` output), keyed by dataset name.
            Accepted for interface completeness but not yet folded into
            the computed threshold/verdict/limitations below. This mirrors
            `evaluate_selectivity_extremes(extreme_fraction=...)`:
            an accepted, currently-inert parameter reserved for a later,
            more granular revision (e.g. surfacing a dataset-specific
            robustness caveat next to its verdict) rather than a silent
            oversight.

    Returns:
        ``{"correlation_threshold_mi": float | None, "recommendation": str,
        "per_dataset_verdict": {dataset: "coupling recommended" |
        "coupling not justified"}, "limitations": list[str]}``.

        ``correlation_threshold_mi`` is the lowest ``measured_mi`` in
        ``correlation_sweep_results`` (scanned in ascending ``measured_mi``
        order, independent of input order) at which
        ``coupled_median_qerror < spatial_only_median_qerror`` strictly
        begins after at least one lower-MI row where that inequality does not hold,
        PROVIDED that the same strict inequality also holds for
        every other row in the sweep whose ``measured_mi`` is at or above
        that candidate value (a persistence check). Without this check, a
        single noisy low-MI row where coupling happens to win by chance
        would be reported as a threshold even if coupling actually loses
        at several higher-MI rows above it -- i.e. the ordering is
        non-monotone in ``measured_mi``, which the realistic AMI values
        this study measures (~0.0-0.08, a small and compressed range where
        median q-errors are finite-sample statistics prone to non-monotone
        flips) make a real, not merely theoretical, risk. ``None`` if no
        candidate row exists, or if a candidate is found but fails the
        persistence check, including when ``correlation_sweep_results`` is
        empty. If coupling wins at the lowest sampled MI, no crossover is observed:
        the lower boundary of the sweep must not be reported as a threshold.

        ``per_dataset_verdict`` resolves ties (``coupled_median_qerror ==
        spatial_only_median_qerror``) to "coupling not justified": the
        comparison is a strict ``<``, so absent a measured accuracy edge
        for coupling, the simpler spatial-only estimator is preferred.
    """
    sorted_rows = sorted(correlation_sweep_results, key=lambda r: r["measured_mi"])

    crossover_mi = None
    candidate_found_but_not_persistent = False
    lower_row_without_win = False
    wins_everywhere = bool(sorted_rows) and all(
        row["coupled_median_qerror"] < row["spatial_only_median_qerror"]
        for row in sorted_rows
    )
    for row in sorted_rows:
        if row["coupled_median_qerror"] < row["spatial_only_median_qerror"]:
            candidate_mi = row["measured_mi"]
            # Persistence check: the candidate is only a valid threshold if
            # coupling also wins at every OTHER row at or above this MI --
            # otherwise a single noisy low-MI win would be mistaken for a
            # stable "above this MI, coupling always wins" cutoff.
            persists = all(
                r["coupled_median_qerror"] < r["spatial_only_median_qerror"]
                for r in sorted_rows
                if r["measured_mi"] >= candidate_mi
            )
            if persists and lower_row_without_win:
                crossover_mi = candidate_mi
            else:
                candidate_found_but_not_persistent = True
            break
        lower_row_without_win = True

    per_dataset_verdict = {}
    for name, stats in accuracy_by_dataset.items():
        recommended = stats["coupled_median_qerror"] < stats["spatial_only_median_qerror"]
        per_dataset_verdict[name] = "coupling recommended" if recommended else "coupling not justified"

    if crossover_mi is not None:
        recommendation = (
            f"An observed crossover occurs near AMI {crossover_mi:.3f} in this sweep; "
            f"report the full curve and validate the boundary on target workloads "
            f"before using it for model selection."
        )
    elif wins_everywhere:
        recommendation = (
            "No crossover was observed: coupling won at every sampled strength, so "
            "the sweep's lower boundary is not a threshold. Report the full curve."
        )
    elif candidate_found_but_not_persistent:
        recommendation = (
            "No single correlation-strength threshold reliably separates when "
            "coupling helps -- report the full accuracy-vs-correlation-strength "
            "curve rather than a single cutoff."
        )
    else:
        recommendation = "No clear crossover found in the swept range; report both estimators' full curves rather than a single threshold."

    limitations = [
        "Spatio-temporal joins and kNN cost estimation are out of scope; this guidance applies to range predicates only.",
        "Any observed correlation-strength crossover is empirical and workload-specific, not a theoretical or deployment guarantee.",
        "Temporal-drift robustness was checked without retraining; a production deployment would need a retraining policy this study does not evaluate.",
    ]

    return {
        "correlation_threshold_mi": crossover_mi,
        "recommendation": recommendation,
        "per_dataset_verdict": per_dataset_verdict,
        "limitations": limitations,
    }
