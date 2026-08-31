from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


TIERS = ("tier1_k8", "tier2_augmented", "tier3_full")
COST_METRICS = ("normalized_total_cost", "paired_replay_count")


def canonical_sha256(value: Any) -> str:
    payload = dict(value) if isinstance(value, dict) else value
    if isinstance(payload, dict):
        payload.pop("semantic_sha256", None)
    return hashlib.sha256(
        (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def row_certificate_pass(row: Mapping[str, Any], qualification: Mapping[str, Any]) -> bool:
    required_finite = (
        "total_energy", "fp32_robust_margin", "bf16_robust_margin",
        "candidate_permutation_error",
    )
    return bool(
        row.get("certified") is True
        and all(_finite(row.get(field)) for field in required_finite)
        and row.get("hard_pass") is True
        and row.get("information_path_pass", True) is True
        and row.get("exact_zero_pass") is True
        and row.get("fp32_top1_pass") is True
        and row.get("bf16_top1_pass") is True
        and row.get("fp32_effect_peak_pass") is True
        and row.get("bf16_effect_peak_pass") is True
        and float(row["fp32_robust_margin"]) >= float(qualification["robust_margin_minimum"])
        and float(row["bf16_robust_margin"]) >= float(qualification["robust_margin_minimum"])
        # A single committed action is governed by the hard lease.  The
        # stricter 0.018 threshold is a population P99 gate.
        and float(row["total_energy"]) <= float(qualification["energy_hard_max"])
        and float(row["candidate_permutation_error"])
        <= float(qualification["candidate_permutation_error_max"])
    )


def row_invalid(row: Mapping[str, Any], qualification: Mapping[str, Any]) -> bool:
    """Return true for failures that must stop the scientific run, not escalate."""
    required_finite = (
        "total_energy", "fp32_robust_margin", "bf16_robust_margin",
        "candidate_permutation_error", "per_utility_normalized_cost",
        "per_utility_paired_replay_count",
    )
    if not all(_finite(row.get(field)) for field in required_finite):
        return True
    if row.get("hard_pass") is not True or row.get("information_path_pass") is not True:
        return True
    if row.get("exact_zero_pass") is not True:
        return True
    if float(row["total_energy"]) > float(qualification["energy_hard_max"]):
        return True
    if float(row["candidate_permutation_error"]) > float(
        qualification["candidate_permutation_error_max"]
    ):
        return True
    # A producer that claims certification while failing the independently
    # derived admission gates is an unsafe/identity inconsistency, never a miss.
    return bool(row.get("certified") is True and not row_certificate_pass(row, qualification))


def _row_key(row: Mapping[str, Any]) -> tuple[int, int]:
    return int(row["context_rank"]), int(row["utility_ordinal"])


def index_rows(rows: Iterable[Mapping[str, Any]], *, allow_partial: bool) -> dict[tuple[int, int], dict[str, Any]]:
    result: dict[tuple[int, int], dict[str, Any]] = {}
    for source in rows:
        row = dict(source)
        key = _row_key(row)
        if key in result:
            raise ValueError(f"duplicate row: {key}")
        result[key] = row
    if not allow_partial:
        expected = {(context, utility) for context in range(4) for utility in range(64)}
        if set(result) != expected:
            raise ValueError("full baseline must contain exactly 4 contexts x 64 Utilities")
    return result


def shared_cost(mapping: Mapping[tuple[int, int], Mapping[str, Any]], context: int, metric: str) -> float:
    field = (
        "shared_context_normalized_cost_total"
        if metric == "normalized_total_cost"
        else "shared_context_paired_replay_total"
    )
    values = {
        round(float(row[field]), 12)
        for (row_context, _), row in mapping.items()
        if row_context == context and field in row
    }
    if not values:
        return 0.0
    if len(values) != 1:
        raise ValueError(f"non-constant shared cost: context={context}, metric={metric}")
    value = float(next(iter(values)))
    if not _finite(value) or value < 0:
        raise ValueError("shared cost must be finite and non-negative")
    return value


def marginal_cost(row: Mapping[str, Any], metric: str) -> float:
    field = (
        "per_utility_normalized_cost"
        if metric == "normalized_total_cost"
        else "per_utility_paired_replay_count"
    )
    value = float(row[field])
    if not _finite(value) or value < 0:
        raise ValueError("marginal cost must be finite and non-negative")
    return value


@dataclass(frozen=True)
class Window:
    cascade_cost: float
    full_cost: float
    cascade_certified: frozenset[int]
    full_certified: frozenset[int]


def recompose_window(
    tier_maps: Mapping[str, Mapping[tuple[int, int], Mapping[str, Any]]],
    baseline: Mapping[tuple[int, int], Mapping[str, Any]],
    qualification: Mapping[str, Any],
    *,
    context: int,
    utilities: Sequence[int],
    metric: str,
) -> Window:
    unresolved = set(map(int, utilities))
    cascade_certified: set[int] = set()
    cascade_cost = 0.0
    for tier in TIERS:
        if not unresolved:
            break
        mapping = tier_maps[tier]
        entered = sorted(unresolved)
        if any((context, utility) not in mapping for utility in entered):
            raise ValueError(f"missing attempted tier row: {tier}, context={context}")
        cascade_cost += shared_cost(mapping, context, metric)
        for utility in entered:
            row = mapping[(context, utility)]
            cascade_cost += marginal_cost(row, metric)
            if row_certificate_pass(row, qualification):
                cascade_certified.add(utility)
                unresolved.remove(utility)

    full_cost = shared_cost(baseline, context, metric)
    full_certified: set[int] = set()
    for utility in utilities:
        row = baseline[(context, int(utility))]
        full_cost += marginal_cost(row, metric)
        if row_certificate_pass(row, qualification):
            full_certified.add(int(utility))
    return Window(
        cascade_cost=cascade_cost,
        full_cost=full_cost,
        cascade_certified=frozenset(cascade_certified),
        full_certified=frozenset(full_certified),
    )


def bootstrap_median_lcb(values: Sequence[float], *, seed: int, resamples: int) -> float:
    source = np.asarray(values, dtype=np.float64)
    if not len(source) or not np.isfinite(source).all():
        raise ValueError("bootstrap values must be finite and non-empty")
    rng = np.random.default_rng(seed)
    medians = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        medians[index] = np.median(source[rng.integers(0, len(source), len(source))])
    return float(np.quantile(medians, 0.025, method="linear"))


def _window_summary(
    tier_maps: Mapping[str, Mapping[tuple[int, int], Mapping[str, Any]]],
    baseline: Mapping[tuple[int, int], Mapping[str, Any]],
    qualification: Mapping[str, Any],
    *,
    active_count: int,
    metric: str,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    reductions: list[float] = []
    ratios: list[float] = []
    cascade_costs: list[float] = []
    full_costs: list[float] = []
    coverage_misses: list[list[int]] = []
    for context in range(4):
        for start in range(64):
            utilities = [(start + offset) % 64 for offset in range(active_count)]
            window = recompose_window(
                tier_maps, baseline, qualification,
                context=context, utilities=utilities, metric=metric,
            )
            if not window.full_certified.issubset(window.cascade_certified):
                coverage_misses.append([context, start])
            denominator = max(window.full_cost, 1e-12)
            reductions.append(1.0 - window.cascade_cost / denominator)
            ratios.append(window.cascade_cost / denominator)
            cascade_costs.append(window.cascade_cost)
            full_costs.append(window.full_cost)
    return {
        "active_utility_count": active_count,
        "window_count": len(reductions),
        "median_reduction": float(statistics.median(reductions)),
        "mean_reduction": float(np.mean(reductions)),
        "median_reduction_lcb95": bootstrap_median_lcb(
            reductions, seed=seed + active_count, resamples=resamples
        ),
        "p95_cost_ratio": float(np.quantile(ratios, 0.95, method="higher")),
        "median_cascade_cost": float(statistics.median(cascade_costs)),
        "median_full_cost": float(statistics.median(full_costs)),
        "coverage_miss_windows": coverage_misses,
    }


def adjudicate(
    *,
    protocol: Mapping[str, Any],
    tier_rows: Mapping[str, Iterable[Mapping[str, Any]]],
    full_baseline_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    qualification = protocol["qualification"]
    tier_maps = {
        tier: index_rows(tier_rows[tier], allow_partial=True)
        for tier in TIERS
    }
    baseline = index_rows(full_baseline_rows, allow_partial=False)
    expected = {(context, utility) for context in range(4) for utility in range(64)}
    cascade_cells: set[tuple[int, int]] = set()
    unsafe_commits: list[list[Any]] = []
    terminal_rows: dict[tuple[int, int], Mapping[str, Any]] = {}
    for cell in sorted(expected):
        for tier in TIERS:
            row = tier_maps[tier].get(cell)
            if row is None:
                raise ValueError(f"cascade did not attempt unresolved cell: {tier}, {cell}")
            derived = row_certificate_pass(row, qualification)
            if row.get("certified") is True and not derived:
                unsafe_commits.append([cell[0], cell[1], tier])
            if derived:
                cascade_cells.add(cell)
                terminal_rows[cell] = row
                break
    full_cells = {cell for cell, row in baseline.items() if row_certificate_pass(row, qualification)}
    missed = sorted(full_cells - cascade_cells)

    windows: dict[str, Any] = {}
    active_counts = tuple(dict.fromkeys(
        int(value) for value in (
            list(protocol["cost_accounting"]["active_utility_counts"])
            + list(protocol["cost_accounting"].get("secondary_active_utility_counts", []))
        )
    ))
    for metric_index, metric in enumerate(COST_METRICS):
        windows[metric] = {
            str(active_count): _window_summary(
                tier_maps, baseline, qualification,
                active_count=active_count,
                metric=metric,
                seed=int(qualification["bootstrap_seed"]) + metric_index * 100000,
                resamples=int(qualification["bootstrap_resamples"]),
            )
            for active_count in active_counts
        }

    energies = [float(row["total_energy"]) for row in terminal_rows.values()]
    nonfinite = sum(
        not all(_finite(row.get(field)) for field in (
            "total_energy", "fp32_robust_margin", "bf16_robust_margin",
            "candidate_permutation_error", "per_utility_normalized_cost",
            "per_utility_paired_replay_count",
        ))
        for mapping in (*tier_maps.values(), baseline)
        for row in mapping.values()
    )
    hard_violations = sum(row.get("hard_pass") is not True for mapping in tier_maps.values() for row in mapping.values())
    information_violations = sum(row.get("information_path_pass", True) is not True for mapping in tier_maps.values() for row in mapping.values())
    primary = tuple(map(str, protocol["cost_accounting"]["active_utility_counts"][:2]))
    # The protocol fixes [1,4] as primary even though secondary counts are also reported.
    primary = ("1", "4")
    normalized_primary = [windows["normalized_total_cost"][value] for value in primary]
    replay_primary = [windows["paired_replay_count"][value] for value in primary]

    invalid = bool(nonfinite or hard_violations or information_violations)
    unsafe = bool(unsafe_commits)
    coverage_fail = bool(missed)
    committed_energy_p99 = (
        float(np.quantile(energies, 0.99, method="higher")) if energies else None
    )
    committed_energy_max = max(energies) if energies else None
    energy_fail = bool(
        energies
        and (
            committed_energy_p99 > float(qualification["energy_p99_max"])
            or committed_energy_max > float(qualification["energy_hard_max"])
        )
    )
    economic_fail = not (
        all(row["median_reduction"] >= float(qualification["primary_median_cost_reduction_minimum"])
            and row["median_reduction_lcb95"] > float(qualification["bootstrap_lcb95_cost_reduction_minimum"])
            and row["p95_cost_ratio"] <= float(qualification["tail_cost_ratio_p95_maximum"])
            for row in normalized_primary)
        and all(row["median_reduction"] >= float(qualification["primary_median_paired_replay_reduction_minimum"])
                for row in replay_primary)
    )
    if invalid:
        branch = "CASCADE_INVALID"
    elif unsafe:
        branch = "CASCADE_UNSAFE_FAIL"
    elif coverage_fail:
        branch = "CASCADE_COVERAGE_FAIL"
    elif energy_fail:
        branch = "CASCADE_ENERGY_FAIL"
    elif economic_fail:
        branch = "CASCADE_ECONOMIC_FAIL"
    else:
        branch = "CASCADE_PASS"

    result = {
        "schema": "NCO_EFA2_CADC_V1_ADJUDICATION",
        "status": branch,
        "population_cells": 256,
        "cascade_certified_count": len(cascade_cells),
        "full_baseline_certified_count": len(full_cells),
        "full_cells_missed_by_cascade": [list(cell) for cell in missed],
        "cascade_extra_cells": [list(cell) for cell in sorted(cascade_cells - full_cells)],
        "unsafe_commits": unsafe_commits,
        "hard_violation_count": hard_violations,
        "information_path_violation_count": information_violations,
        "nonfinite_count": nonfinite,
        "committed_energy_count": len(energies),
        "committed_energy_p99": committed_energy_p99,
        "committed_energy_max": committed_energy_max,
        "energy_p99_gate_pass": not energy_fail,
        "window_metrics": windows,
        "model_training_performed": False,
        "qwen_parameters_updated": False,
        "frozen_fresh_confirm_read": False,
        "unseen_utility_read": False,
    }
    result["semantic_sha256"] = canonical_sha256(result)
    return result
