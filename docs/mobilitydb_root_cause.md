# Root cause of MobilityDB's `&&` undercount: source citations

**Source:** MobilityDB v1.3.0, commit `8aa274c71a59170381fcf2ddef9d5907f3076e11`
(2026-05-01). This is the build used for every experiment (`pg_extension`:
`mobilitydb 1.3.0`). The checkout is at `.build/MobilityDB` in the project directory.
Line numbers refer to that commit. Quotes are verbatim.

**Evidence:** `run/vldbj_rev/independence/` (script `scripts/vldbj_independence_check.py`)
and the box-count oracle (family (d) in `run/vldbj_rev/stats/paired_bootstrap_clamped.csv`).

## 1. Which function estimates `tgeompoint && stbox`

`mobilitydb/sql/geo/060_tpoint_boxops.in.sql:440-443` declares the operator:

```sql
  PROCEDURE = temporal_overlaps,
  LEFTARG = tgeompoint, RIGHTARG = stbox,
  COMMUTATOR = &&,
  RESTRICT = tspatial_sel, JOIN = tspatial_joinsel
```

The live catalogue confirms this: `pg_operator` gives `&&(tgeompoint, stbox)` →
`oprrest = tspatial_sel`. The SQL function `tspatial_sel` is declared in
`mobilitydb/sql/geo/051_stbox.in.sql:388`.

`mobilitydb/src/temporal/temporal_selfuncs.c:750-758`:

```c
PG_FUNCTION_INFO_V1(Tspatial_sel);
/**
 * @brief Estimate the restriction selectivity of the operators for temporal
 * points
 */
Datum
Tspatial_sel(PG_FUNCTION_ARGS)
{
  return Float8GetDatum(temporal_sel_family(fcinfo, TSPATIALTYPE));
```

This reaches `temporal_sel()` (`temporal_selfuncs.c:557`).

## 2. Space and time are multiplied as independent

`mobilitydb/src/temporal/temporal_selfuncs.c:669-696` (inside `temporal_sel()`, branch
`tempfamily == TSPATIALTYPE`):

```c
    /* Enable the multiplication of the selectivity of the spatial and time
     * dimensions since either may be missing */
    selec = 1.0;
    ...
        selec *= geo_sel(&vardata, &box, oper);                      /* line 683 */
    ...
      selec *= temporal_sel_tstzspan(&vardata, &period, oper);       /* line 695 */
```

The spatial part `geo_sel()` uses only the x/y dimensions of the N-D histogram.
`mobilitydb/src/geo/tspatial_selfuncs.c:523-524`:

```c
  /* Calculate the number of common coordinate dimensions  on the histogram */
  ndims = (int) Min(nd_stats->ndims, MEOS_FLAGS_GET_Z(box->flags) ? 3 : 2);
```

**Measured:** the planner's `&&` estimate equals the product of its own spatial-only and
temporal-only estimates (`median_combination_residual` ≤ 0.011 log10 on all three
datasets; macro `\NumCombResidAbsMax`).

## 3. The spatial histogram estimates area mass, not overlap counts

**Building the histogram (ANALYZE).** Each sampled trajectory box adds the *fraction of its
own area* inside each cell. `mobilitydb/src/geo/tspatial_analyze.c:914-920`:

```c
      /*
       * If a feature box is completely inside one cell the ratio will be
       * 1.0. If a feature box is 50% in two cells, each cell will get
       * 0.5 added on.
       */
      ratio = nd_box_ratio_overlaps(&nd_cell, nd_box, (int) nd_stats->ndims);
      nd_stats->value[nd_stats_value_index(nd_stats, at)] += (float4) ratio;
```

`nd_box_ratio_overlaps` (`tspatial_analyze.c:244-248`): "Return the proportion of b2 that
is covered by b1".

**Using it (query time).** `geo_sel()` sums each cell's value multiplied by the fraction of
that cell the query covers. `mobilitydb/src/geo/tspatial_selfuncs.c:652-671`:

```c
    /* We have to pro-rate partially overlapped cells. */
    ...
      ratio = (float4) (nd_box_ratio_overlaps(&nd_box, &nd_cell, ndims));
    cell_count = nd_stats->value[nd_stats_value_index(nd_stats, at)];

    /* Add the pro-rated count for this cell to the overall total */
    total_count += (double) cell_count * ratio;
  ...
  /* Scale by the number of features in our histogram to get the proportion */
  selec = total_count / nd_stats->histogram_features;
```

The result is the expected share of trajectory-box *area* inside the query box. A
trajectory whose box only partly overlaps the query contributes less than 1, although it
satisfies `&&`. So the estimate is a systematic undercount of the overlap count, and the
undercount grows with box size relative to the query.

**Measured** (fixed-extent workloads, median log10, `run/vldbj_rev/independence/`):

| | T-Drive | Porto | AIS |
|---|---|---|---|
| Spatial marginal `geo_sel` vs exact spatial-only count | −0.30 | −0.43 | −0.44 |
| Share of queries on which `geo_sel` undercounts | 98.7% | 100% | 100% |
| Temporal marginal vs exact temporal-only count | ±0.001 | ±0.001 | ±0.001 |
| Independence error (product of EXACT marginals vs exact joint) | +0.06 | +0.00 | −0.17 |

Macros: `\NumSpatMarg{TD,PO,AIS}Err`, `\NumSpatMargUnderMin`, `\NumTempMargAbsMax`,
`\NumIndep{TD,PO,AIS}Err`.

So the dominant cause is the spatial marginal. The independence assumption contributes
noticeably only on AIS, where long vessel trajectories make space and time positively
dependent.

## 4. Join selectivity uses the same product structure

`mobilitydb/src/temporal/temporal_selfuncs.c:916-947`, "Multiply the components of the
join selectivity estimation": `selec *= span_joinsel(...)` (line 930) and
`selec *= geo_joinsel(stats1, stats2)` (line 947), with `tspatial_joinsel_default(oper)`
(line 945) as the fallback. This is consistent with the systematic join underestimation in
`run/vldbj_rev/join/join_estimates.csv`.

## 5. Headroom: what an exact box count would give

Using the exact `traj && B` count as the estimate (box-count oracle; an upper bound on the
path count, not deployable; `\NumBoxOracleRows`, family (d)):
- **Taxi settings:** the oracle beats the best learned model, Separable (ratios
  `\NumBoxSepRatioMinTaxi`–`\NumBoxSepRatioMaxTaxi`). Fixing the histogram would remove most
  of Native's error.
- **AIS mixed:** the oracle is worse than Native-BBox (`\NumAISMixBoxNatRatio`). The
  box/path gap is large for ships, and Native's undercount was partly cancelling it.
