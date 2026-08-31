from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class PhysicalContract:
    bridge_budget: float = 0.006
    answer_budget: float = 0.015
    p99_budget: float = 0.018
    hard_total_budget: float = 0.021
    maximum_write_norm: float = 0.05
    action_dimension: int = 9216
    layers: tuple[int, int, int] = (7, 14, 21)


def contract_metrics(action: Array, contract: PhysicalContract = PhysicalContract()) -> dict[str, float | bool]:
    value = np.asarray(action, dtype=np.float64).reshape(-1)
    if value.shape != (contract.action_dimension,) or not np.isfinite(value).all():
        raise ValueError("finite 9216-dimensional physical action required")
    shaped = value.reshape(3, 3, 1024)
    write_norms = np.linalg.norm(shaped, axis=-1)
    bridge = float(np.square(write_norms[0]).sum())
    answer = float(np.square(write_norms[1:]).sum())
    total = bridge + answer
    maximum_write = float(write_norms.max())
    passed = (
        bridge <= contract.bridge_budget + 1e-10
        and answer <= contract.answer_budget + 1e-10
        and total <= contract.hard_total_budget + 1e-10
        and maximum_write <= contract.maximum_write_norm + 1e-10
    )
    return {
        "bridge": bridge,
        "answer": answer,
        "total": total,
        "maximum_write": maximum_write,
        "hard_pass": bool(passed),
    }


def legal_radial_energy_cap(direction: Array, contract: PhysicalContract = PhysicalContract()) -> float:
    value = np.asarray(direction, dtype=np.float64).reshape(-1)
    value = value / max(float(np.linalg.norm(value)), 1e-15)
    shaped = value.reshape(3, 3, 1024)
    write_energy = np.square(np.linalg.norm(shaped, axis=-1))
    caps = [
        contract.hard_total_budget,
        contract.bridge_budget / max(float(write_energy[0].sum()), 1e-15),
        contract.answer_budget / max(float(write_energy[1:].sum()), 1e-15),
        contract.maximum_write_norm**2 / max(float(write_energy.max()), 1e-15),
    ]
    return float(min(caps))


def inward_radial_action(direction: Array, energy: float, contract: PhysicalContract = PhysicalContract()) -> Array:
    value = np.asarray(direction, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 0:
        return np.zeros(contract.action_dimension, dtype=np.float32)
    unit = value / norm
    cap = legal_radial_energy_cap(unit, contract)
    chosen = min(max(float(energy), 0.0), cap, contract.hard_total_budget) * (1.0 - 1e-6)
    return (unit * np.sqrt(chosen)).astype(np.float32)


def _sylvester_hadamard(order: int) -> Array:
    if order < 1 or order & (order - 1):
        raise ValueError("Sylvester Hadamard order must be a positive power of two")
    value = np.ones((1, 1), dtype=np.float64)
    while value.shape[0] < order:
        value = np.block([[value, value], [value, -value]])
    return value


def fixed_effect_basis(dimension: int = 63) -> Array:
    """Deterministic centered Hadamard frame; no random-frame stop-rule risk."""
    if not 1 <= dimension <= 63:
        raise ValueError("effect probe dimension must lie in [1,63]")
    # The first Sylvester column is the constant mode and is removed.
    return (_sylvester_hadamard(64)[:, 1:] / 8.0)[:, :dimension]


def build_pullback_frame(
    effect_vjp: Callable[[Array], Array],
    effect_probes: Array,
) -> Array:
    """Build B_tau=J_tau^T Q without exposing a target/candidate identifier."""
    q = np.asarray(effect_probes, dtype=np.float64)
    if q.ndim != 2 or q.shape[0] != 64 or not np.isfinite(q).all():
        raise ValueError("effect probes must be finite [64,k]")
    pulled = np.asarray(effect_vjp(q.T), dtype=np.float64)
    if pulled.shape != (q.shape[1], 9216) or not np.isfinite(pulled).all():
        raise ValueError("VJP callback must return [k,9216]")
    basis = pulled.T
    norms = np.linalg.norm(basis, axis=0)
    if np.any(norms <= 1e-12):
        raise RuntimeError("inverse-dynamics pullback contains an unobservable probe")
    return basis / norms[None, :]


def radial_feasible_interval(
    *,
    zero_fp32: Array,
    zero_bf16: Array,
    response_fp32: Array,
    response_bf16: Array,
    effect_request: Array,
    qualification_margin: float = 0.2,
) -> dict[str, float | bool]:
    """Return the exact non-negative radial interval for all dual-precision margins."""
    z32 = np.asarray(zero_fp32, dtype=np.float64).reshape(64)
    zbf = np.asarray(zero_bf16, dtype=np.float64).reshape(64)
    r32 = np.asarray(response_fp32, dtype=np.float64).reshape(64)
    rbf = np.asarray(response_bf16, dtype=np.float64).reshape(64)
    request = np.asarray(effect_request, dtype=np.float64).reshape(64)
    target = int(np.argmax(request))
    lower = 0.0
    upper = float("inf")
    for zero, response in ((z32, r32), (zbf, rbf)):
        for competitor in range(64):
            if competitor == target:
                continue
            slope = float(response[target] - response[competitor])
            deficit = qualification_margin - float(zero[target] - zero[competitor])
            if slope > 1e-12:
                lower = max(lower, deficit / slope)
            elif slope < -1e-12:
                upper = min(upper, deficit / slope)
            elif deficit > 0:
                return {"feasible": False, "alpha_min": float("inf"), "alpha_max": float("-inf")}
    lower = max(lower, 0.0)
    return {
        "feasible": bool(lower <= upper and upper >= 0.0),
        "alpha_min": float(lower),
        "alpha_max": float(upper),
    }


def _certificate_linear_constraints(
    zero_fp32: Array,
    zero_bf16: Array,
    response_fp32: Array,
    response_bf16: Array,
    effect_request: Array,
    qualification_margin: float,
    effect_margin_floor: float,
) -> tuple[Array, Array, int]:
    z32 = np.asarray(zero_fp32, dtype=np.float64).reshape(64)
    zbf = np.asarray(zero_bf16, dtype=np.float64).reshape(64)
    r32 = np.asarray(response_fp32, dtype=np.float64)
    rbf = np.asarray(response_bf16, dtype=np.float64)
    request = np.asarray(effect_request, dtype=np.float64).reshape(64)
    if r32.shape != rbf.shape or r32.shape[0] != 64:
        raise ValueError("dual-precision derivative-normalized responses must be [64,k]")
    target = int(np.argmax(request))
    rows, rhs = [], []
    for zero, response in ((z32, r32), (zbf, rbf)):
        for competitor in range(64):
            if competitor == target:
                continue
            rows.append(response[target] - response[competitor])
            rhs.append(qualification_margin - float(zero[target] - zero[competitor]))
            rows.append(response[target] - response[competitor])
            rhs.append(effect_margin_floor)
    return np.asarray(rows), np.asarray(rhs), target


def solve_certificate_aligned_start(
    *,
    zero_fp32: Array,
    zero_bf16: Array,
    response_fp32: Array,
    response_bf16: Array,
    effect_request: Array,
    pullback_frame: Array,
    qualification_margin: float = 0.2,
    effect_margin_floor: float = 1e-5,
    interior_energy: float = 0.0175,
    contract: PhysicalContract = PhysicalContract(),
) -> dict[str, Array | float | bool | str]:
    """Minimum-physical-energy reduced QCQP aligned with the final certificate."""
    from scipy.optimize import minimize

    frame = np.asarray(pullback_frame, dtype=np.float64)
    if frame.ndim != 2 or frame.shape[0] != contract.action_dimension:
        raise ValueError("pullback frame must be [9216,k]")
    if not np.isfinite(frame).all():
        raise FloatingPointError("non-finite pullback frame")
    a, b, _target = _certificate_linear_constraints(
        zero_fp32, zero_bf16, response_fp32, response_bf16,
        effect_request, qualification_margin, effect_margin_floor,
    )
    k = frame.shape[1]
    gram = frame.T @ frame
    shaped = frame.reshape(3, 3, 1024, k)
    group_grams = {
        "bridge": shaped[0].reshape(-1, k).T @ shaped[0].reshape(-1, k),
        "answer": shaped[1:].reshape(-1, k).T @ shaped[1:].reshape(-1, k),
        "total": gram,
    }
    write_grams = [shaped[layer, write].T @ shaped[layer, write]
                   for layer in range(3) for write in range(3)]

    def objective(c):
        return 0.5 * float(c @ gram @ c)

    def objective_jac(c):
        return gram @ c

    def inequalities(c):
        values = [*(a @ c - b),
                  contract.bridge_budget - float(c @ group_grams["bridge"] @ c),
                  contract.answer_budget - float(c @ group_grams["answer"] @ c),
                  min(interior_energy, contract.hard_total_budget) - float(c @ gram @ c)]
        values.extend(contract.maximum_write_norm**2 - float(c @ value @ c) for value in write_grams)
        return np.asarray(values)

    def inequalities_jac(c):
        rows = [*a,
                -2.0 * group_grams["bridge"] @ c,
                -2.0 * group_grams["answer"] @ c,
                -2.0 * gram @ c]
        rows.extend(-2.0 * value @ c for value in write_grams)
        return np.asarray(rows)

    ridge = 1e-8 * np.eye(k)
    least_squares = np.linalg.lstsq(a.T @ a + ridge, a.T @ np.maximum(b, 0.0), rcond=None)[0]
    starts = (np.zeros(k, dtype=np.float64), least_squares)
    candidates = []
    for ordinal, start in enumerate(starts):
        solved = minimize(
            objective, start, jac=objective_jac, method="SLSQP",
            constraints={"type": "ineq", "fun": inequalities, "jac": inequalities_jac},
            options={"maxiter": 2000, "ftol": 1e-12, "disp": False},
        )
        c = np.asarray(solved.x, dtype=np.float64)
        residual = inequalities(c)
        candidates.append((float(np.maximum(-residual, 0.0).max()), objective(c), ordinal, solved, c))
    maximum_violation, _energy, start_ordinal, solved, coefficients = min(
        candidates, key=lambda row: (row[0], row[1], row[2])
    )
    action = (frame @ coefficients).astype(np.float32)
    metrics = contract_metrics(action, contract)
    success = bool(solved.success and maximum_violation <= 2e-6 and metrics["hard_pass"])
    return {
        "action": action,
        "coefficients": coefficients.astype(np.float32),
        "solver_success": success,
        "solver_status": str(solved.message),
        "solver_iterations": int(solved.nit),
        "maximum_constraint_violation": float(maximum_violation),
        "selected_start_ordinal": int(start_ordinal),
        "requested_energy": float(metrics["total"]),
        "response_unit": "effect_delta_per_unit_action_amplitude",
    }


def active_set_summary(action: Array, contract: PhysicalContract = PhysicalContract(), tolerance: float = 5e-4) -> dict:
    metrics = contract_metrics(action, contract)
    shaped = np.asarray(action, dtype=np.float64).reshape(3, 3, 1024)
    write_energy = np.square(np.linalg.norm(shaped, axis=-1))
    mass = write_energy / max(float(write_energy.sum()), 1e-15)
    entropy = -float(np.sum(np.where(mass > 0, mass * np.log(mass + 1e-30), 0.0)))
    slacks = {
        "bridge": contract.bridge_budget - float(write_energy[0].sum()),
        "answer": contract.answer_budget - float(write_energy[1:].sum()),
        "total": contract.hard_total_budget - float(write_energy.sum()),
        "per_write": contract.maximum_write_norm**2 - write_energy,
    }
    active = {
        "bridge": slacks["bridge"] <= tolerance,
        "answer": slacks["answer"] <= tolerance,
        "total": slacks["total"] <= tolerance,
        "per_write_indices": np.flatnonzero(slacks["per_write"] <= tolerance).astype(int).tolist(),
    }
    return {
        "metrics": metrics,
        "slack_bridge": float(slacks["bridge"]),
        "slack_answer": float(slacks["answer"]),
        "slack_total": float(slacks["total"]),
        "minimum_per_write_slack": float(np.min(slacks["per_write"])),
        "allocation_entropy": entropy,
        "active": active,
    }


def curvature_ratio(zero: Array, response: Array, delta_amplitude: float, replayed: Array, epsilon: float = 1e-8) -> float:
    predicted = np.asarray(zero, dtype=np.float64) + float(delta_amplitude) * np.asarray(response, dtype=np.float64)
    actual = np.asarray(replayed, dtype=np.float64)
    linear_delta = predicted - np.asarray(zero, dtype=np.float64)
    return float(np.linalg.norm(actual - predicted) / (np.linalg.norm(linear_delta) + epsilon))


def normalized_cost(*, cuda_milliseconds: float, paired_replay_median_ms: float) -> float:
    if paired_replay_median_ms <= 0 or cuda_milliseconds < 0:
        raise ValueError("positive calibration and non-negative measured time required")
    return float(cuda_milliseconds / paired_replay_median_ms)


def protocol_contract_dict() -> dict:
    return asdict(PhysicalContract())
