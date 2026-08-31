from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .physical_abi import (
    PhysicalContract,
    active_set_summary,
    build_pullback_frame,
    curvature_ratio,
    solve_certificate_aligned_start,
)


Array = np.ndarray


@dataclass(frozen=True)
class SearchOutcome:
    action: Array
    certified: bool
    hard_pass: bool
    exact_zero_pass: bool
    total_energy: float
    paired_replay_count: int
    normalized_total_cost: float
    first_certified_step: int | None
    terminal_stage: str
    metrics: dict[str, Any]


class IDTSRuntime(Protocol):
    """Capability-blind runtime interface bound by the final execution package."""

    context_rank: int

    def cost_snapshot(self) -> dict[str, float]: ...

    def zero_scores(self) -> tuple[Array, Array]: ...

    def effect_vjp(self, effect_probes: Array) -> Array: ...

    def replay_transactional(self, action: Array, precision: str) -> Array: ...

    def typed_effect_request(self, utility_ordinal: int) -> Array: ...

    def run_full_reference(self) -> list[dict[str, Any]]: ...

    def search_from_hint(
        self,
        *,
        utility_ordinal: int,
        warm_action: Array,
        trust_radius: float,
        force_relinearize: bool,
    ) -> SearchOutcome: ...


def run_context_full_reference_arm(runtime: IDTSRuntime) -> list[dict[str, Any]]:
    """Execute the byte-bound WitnessSearch-v2 parent without reinterpretation."""
    rows = runtime.run_full_reference()
    if len(rows) != 64 or {int(row["utility_ordinal"]) for row in rows} != set(range(64)):
        raise RuntimeError("full reference must close exactly 64 Utilities")
    return rows


def directional_responses(
    runtime: IDTSRuntime,
    frame: Array,
    *,
    probe_energy: float,
    contract: PhysicalContract = PhysicalContract(),
) -> tuple[Array, Array]:
    k = frame.shape[1]
    responses = {"fp32": [], "bf16": []}
    for column in range(k):
        direction = frame[:, column]
        amplitude = np.sqrt(min(probe_energy, 0.25 * contract.hard_total_budget))
        plus = direction * amplitude
        minus = -direction * amplitude
        for precision in ("fp32", "bf16"):
            positive = runtime.replay_transactional(plus, precision)
            negative = runtime.replay_transactional(minus, precision)
            responses[precision].append((positive - negative) / (2.0 * amplitude))
    return np.stack(responses["fp32"], axis=1), np.stack(responses["bf16"], axis=1)


def utility_aligned_effect_probes(
    zero_fp32: Array,
    zero_bf16: Array,
    effect_request: Array,
    maximum_probes: int = 4,
) -> Array:
    """Typed-predicate plus worst-margin covectors; no capability/action lookup."""
    request = np.asarray(effect_request, dtype=np.float64).reshape(64)
    target = int(np.argmax(request))
    columns = [request - request.mean()]
    competitors = []
    for zero in (np.asarray(zero_fp32), np.asarray(zero_bf16)):
        order = np.argsort(-(zero - np.eye(64)[target] * 1e30))
        competitors.extend(int(value) for value in order if int(value) != target)
    for competitor in dict.fromkeys(competitors):
        vector = np.zeros(64, dtype=np.float64)
        vector[target] = 1.0
        vector[competitor] = -1.0
        columns.append(vector)
        if len(columns) >= maximum_probes:
            break
    raw = np.stack(columns, axis=1)
    q, r = np.linalg.qr(raw, mode="reduced")
    keep = np.abs(np.diag(r)) > 1e-10
    return q[:, keep]


def run_context_arm(
    runtime: IDTSRuntime,
    effect_frame: Array,
    *,
    augmented: bool,
    probe_energy: float = 0.0005,
    contract: PhysicalContract = PhysicalContract(),
) -> list[dict[str, Any]]:
    cost_before = runtime.cost_snapshot()
    zero32, zerobf = runtime.zero_scores()
    frame = build_pullback_frame(runtime.effect_vjp, effect_frame)
    response32, responsebf = directional_responses(
        runtime, frame, probe_energy=probe_energy, contract=contract
    )
    cost_after = runtime.cost_snapshot()
    shared_cost = {
        key: float(cost_after.get(key, 0.0) - cost_before.get(key, 0.0))
        for key in set(cost_before) | set(cost_after)
    }
    if any(value < -1e-9 for value in shared_cost.values()):
        raise RuntimeError("runtime cost counters must be monotone")
    rows: list[dict[str, Any]] = []
    for utility in range(64):
        request = runtime.typed_effect_request(utility)
        start = solve_certificate_aligned_start(
            zero_fp32=zero32,
            zero_bf16=zerobf,
            response_fp32=response32,
            response_bf16=responsebf,
            effect_request=request,
            pullback_frame=frame,
            contract=contract,
        )
        warm = np.asarray(start["action"], dtype=np.float32)
        active = active_set_summary(warm, contract)
        trust_radius = 0.01
        force_relinearize = False
        curvature = {"fp32": 0.0, "bf16": 0.0}
        if augmented:
            amplitude = max(float(np.linalg.norm(warm)), 1e-12)
            coefficients = np.asarray(start["coefficients"], dtype=np.float64)
            predicted32 = response32 @ coefficients
            predictedbf = responsebf @ coefficients
            replay32 = runtime.replay_transactional(warm, "fp32")
            replaybf = runtime.replay_transactional(warm, "bf16")
            curvature = {
                "fp32": curvature_ratio(zero32, predicted32 / amplitude, amplitude, replay32),
                "bf16": curvature_ratio(zerobf, predictedbf / amplitude, amplitude, replaybf),
            }
            worst = max(curvature.values())
            trust_radius = float(np.clip(0.01 / (1.0 + worst), 0.0025, 0.02))
            force_relinearize = bool(worst > 0.5)
        outcome = runtime.search_from_hint(
            utility_ordinal=utility,
            warm_action=warm,
            trust_radius=trust_radius,
            force_relinearize=force_relinearize,
        )
        rows.append({
            "context_rank": int(runtime.context_rank),
            "utility_ordinal": utility,
            "certified": bool(outcome.certified),
            "hard_pass": bool(outcome.hard_pass),
            "exact_zero_pass": bool(outcome.exact_zero_pass),
            "total_energy": float(outcome.total_energy),
            "paired_replay_count": float(outcome.paired_replay_count) + shared_cost.get("paired_replay_count", 0.0) / 64.0,
            "normalized_total_cost": float(outcome.normalized_total_cost) + shared_cost.get("normalized_total_cost", 0.0) / 64.0,
            "shared_context_acquisition_cost": {key: value / 64.0 for key, value in sorted(shared_cost.items())},
            "shared_context_normalized_cost_total": shared_cost.get("normalized_total_cost", 0.0),
            "shared_context_paired_replay_total": shared_cost.get("paired_replay_count", 0.0),
            "per_utility_normalized_cost": float(outcome.normalized_total_cost),
            "per_utility_paired_replay_count": float(outcome.paired_replay_count),
            "fp32_certificate_pass": bool(outcome.metrics.get("fp32_certificate_pass", False)),
            "bf16_certificate_pass": bool(outcome.metrics.get("bf16_certificate_pass", False)),
            "fp32_top1_pass": bool(outcome.metrics.get("fp32_top1_pass", False)),
            "fp32_effect_peak_pass": bool(outcome.metrics.get("fp32_effect_peak_pass", False)),
            "fp32_robust_margin": float(outcome.metrics.get("fp32_robust_margin", float("nan"))),
            "bf16_top1_pass": bool(outcome.metrics.get("bf16_top1_pass", False)),
            "bf16_effect_peak_pass": bool(outcome.metrics.get("bf16_effect_peak_pass", False)),
            "bf16_robust_margin": float(outcome.metrics.get("bf16_robust_margin", float("nan"))),
            "candidate_permutation_error": float(outcome.metrics.get("candidate_permutation_error", float("nan"))),
            "first_certified_step": outcome.first_certified_step,
            "terminal_stage": outcome.terminal_stage,
            "reduced_solver_success": bool(start["solver_success"]),
            "reduced_solver_status": str(start["solver_status"]),
            "reduced_solver_maximum_constraint_violation": float(start["maximum_constraint_violation"]),
            "response_unit": str(start["response_unit"]),
            "active_set": active,
            "curvature": curvature,
            "search_metrics": outcome.metrics,
        })
    return rows


def run_context_utility_aligned_arm(
    runtime: IDTSRuntime,
    *,
    augmented: bool = True,
    probe_energy: float = 0.0005,
    contract: PhysicalContract = PhysicalContract(),
) -> list[dict[str, Any]]:
    context_cost_before = runtime.cost_snapshot()
    zero32, zerobf = runtime.zero_scores()
    context_cost_after = runtime.cost_snapshot()
    shared_cost = {
        key: float(context_cost_after.get(key, 0.0) - context_cost_before.get(key, 0.0))
        for key in set(context_cost_before) | set(context_cost_after)
    }
    if any(value < -1e-9 for value in shared_cost.values()):
        raise RuntimeError("runtime cost counters must be monotone")
    rows: list[dict[str, Any]] = []
    for utility in range(64):
        cost_before = runtime.cost_snapshot()
        request = runtime.typed_effect_request(utility)
        probes = utility_aligned_effect_probes(zero32, zerobf, request)
        frame = build_pullback_frame(runtime.effect_vjp, probes)
        response32, responsebf = directional_responses(
            runtime, frame, probe_energy=probe_energy, contract=contract
        )
        start = solve_certificate_aligned_start(
            zero_fp32=zero32, zero_bf16=zerobf,
            response_fp32=response32, response_bf16=responsebf,
            effect_request=request, pullback_frame=frame, contract=contract,
        )
        warm = np.asarray(start["action"], dtype=np.float32)
        active = active_set_summary(warm, contract)
        trust_radius = 0.01
        force_relinearize = False
        curvature = {"fp32": 0.0, "bf16": 0.0}
        if augmented:
            amplitude = max(float(np.linalg.norm(warm)), 1e-12)
            coefficients = np.asarray(start["coefficients"], dtype=np.float64)
            predicted32 = response32 @ coefficients
            predictedbf = responsebf @ coefficients
            replay32 = runtime.replay_transactional(warm, "fp32")
            replaybf = runtime.replay_transactional(warm, "bf16")
            curvature = {
                "fp32": curvature_ratio(zero32, predicted32 / amplitude, amplitude, replay32),
                "bf16": curvature_ratio(zerobf, predictedbf / amplitude, amplitude, replaybf),
            }
            worst = max(curvature.values())
            trust_radius = float(np.clip(0.01 / (1.0 + worst), 0.0025, 0.02))
            force_relinearize = bool(worst > 0.5)
        cost_before_search = runtime.cost_snapshot()
        outcome = runtime.search_from_hint(
            utility_ordinal=utility, warm_action=warm,
            trust_radius=trust_radius, force_relinearize=force_relinearize,
        )
        acquisition = {
            key: float(cost_before_search.get(key, 0.0) - cost_before.get(key, 0.0))
            for key in set(cost_before) | set(cost_before_search)
        }
        rows.append({
            "context_rank": int(runtime.context_rank),
            "utility_ordinal": utility,
            "certified": bool(outcome.certified),
            "hard_pass": bool(outcome.hard_pass),
            "exact_zero_pass": bool(outcome.exact_zero_pass),
            "total_energy": float(outcome.total_energy),
            "paired_replay_count": float(outcome.paired_replay_count) + acquisition.get("paired_replay_count", 0.0) + shared_cost.get("paired_replay_count", 0.0) / 64.0,
            "normalized_total_cost": float(outcome.normalized_total_cost) + acquisition.get("normalized_total_cost", 0.0) + shared_cost.get("normalized_total_cost", 0.0) / 64.0,
            "shared_context_acquisition_cost": {key: value / 64.0 for key, value in sorted(shared_cost.items())},
            "shared_context_normalized_cost_total": shared_cost.get("normalized_total_cost", 0.0),
            "shared_context_paired_replay_total": shared_cost.get("paired_replay_count", 0.0),
            "per_utility_normalized_cost": float(outcome.normalized_total_cost) + acquisition.get("normalized_total_cost", 0.0),
            "per_utility_paired_replay_count": float(outcome.paired_replay_count) + acquisition.get("paired_replay_count", 0.0),
            "fp32_certificate_pass": bool(outcome.metrics.get("fp32_certificate_pass", False)),
            "bf16_certificate_pass": bool(outcome.metrics.get("bf16_certificate_pass", False)),
            "fp32_top1_pass": bool(outcome.metrics.get("fp32_top1_pass", False)),
            "fp32_effect_peak_pass": bool(outcome.metrics.get("fp32_effect_peak_pass", False)),
            "fp32_robust_margin": float(outcome.metrics.get("fp32_robust_margin", float("nan"))),
            "bf16_top1_pass": bool(outcome.metrics.get("bf16_top1_pass", False)),
            "bf16_effect_peak_pass": bool(outcome.metrics.get("bf16_effect_peak_pass", False)),
            "bf16_robust_margin": float(outcome.metrics.get("bf16_robust_margin", float("nan"))),
            "candidate_permutation_error": float(outcome.metrics.get("candidate_permutation_error", float("nan"))),
            "utility_aligned_probe_count": int(probes.shape[1]),
            "per_utility_acquisition_cost": acquisition,
            "first_certified_step": outcome.first_certified_step,
            "terminal_stage": outcome.terminal_stage,
            "reduced_solver_success": bool(start["solver_success"]),
            "reduced_solver_status": str(start["solver_status"]),
            "reduced_solver_maximum_constraint_violation": float(start["maximum_constraint_violation"]),
            "response_unit": str(start["response_unit"]),
            "active_set": active,
            "curvature": curvature,
            "search_metrics": outcome.metrics,
        })
    return rows
