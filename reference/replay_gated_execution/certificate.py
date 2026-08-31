from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .authorization import ItemCertificate
from .physical_abi import PhysicalContract, contract_metrics


@dataclass(frozen=True)
class PrecisionBehavior:
    top1: bool
    effect_peak: bool
    robust_margin: float
    target_effect_margin: float


def behavior_metrics(scores: np.ndarray, zero_scores: np.ndarray, target: int) -> PrecisionBehavior:
    scores = np.asarray(scores, dtype=np.float64).reshape(64)
    zero_scores = np.asarray(zero_scores, dtype=np.float64).reshape(64)
    if not np.isfinite(scores).all() or not np.isfinite(zero_scores).all():
        raise FloatingPointError("non-finite replay scores")
    if isinstance(target, bool) or not 0 <= int(target) < 64:
        raise ValueError("target must index the public Z64 registry")
    target = int(target)
    competitors = np.arange(64) != target
    effect = scores - zero_scores
    return PrecisionBehavior(
        top1=bool(scores[target] > np.max(scores[competitors])),
        effect_peak=bool(effect[target] > np.max(effect[competitors])),
        robust_margin=float(scores[target] - np.max(scores[competitors])),
        target_effect_margin=float(effect[target] - np.max(effect[competitors])),
    )


def build_certificate(
    *,
    action: np.ndarray,
    scores_fp32: np.ndarray,
    scores_bf16: np.ndarray,
    zero_fp32: np.ndarray,
    zero_bf16: np.ndarray,
    target: int,
    candidate_permutation_error: float,
    exact_zero_pass: bool,
    contract: PhysicalContract = PhysicalContract(),
) -> ItemCertificate:
    physical = contract_metrics(action, contract)
    fp32 = behavior_metrics(scores_fp32, zero_fp32, target)
    bf16 = behavior_metrics(scores_bf16, zero_bf16, target)
    finite = bool(
        np.isfinite(np.asarray(action)).all()
        and np.isfinite(candidate_permutation_error)
    )
    return ItemCertificate(
        finite=finite,
        physical_contract_pass=bool(physical["hard_pass"]),
        fp32_top1=fp32.top1,
        bf16_top1=bf16.top1,
        fp32_effect_peak=fp32.effect_peak,
        bf16_effect_peak=bf16.effect_peak,
        fp32_margin=fp32.robust_margin,
        bf16_margin=bf16.robust_margin,
        total_energy=float(physical["total"]),
        candidate_permutation_error=float(candidate_permutation_error),
        exact_zero_pass=bool(exact_zero_pass),
    )
