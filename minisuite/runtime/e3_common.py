from __future__ import annotations

import hashlib
import json
import math
import pathlib
from typing import Any, Iterable

import numpy as np


HIDDEN_SIZE = 960
LAYER_INDICES = (8, 16, 24)
PHASE_COUNT = 3
ACTION_DIMENSION = PHASE_COUNT * len(LAYER_INDICES) * HIDDEN_SIZE

BRIDGE_BUDGET = 0.006
ANSWER_BUDGET = 0.015
TOTAL_BUDGET = 0.021
QUALIFICATION_P99_BUDGET = 0.018
INTERNAL_SEARCH_CAP = 0.0175
PROJECTED_SEARCH_CAP = 0.017749
MAXIMUM_PER_WRITE_RATIO = 0.05
ROBUST_MARGIN = 0.2


def canonical_sha256(value: Any) -> str:
    body = dict(value) if isinstance(value, dict) else value
    if isinstance(body, dict):
        body.pop("semantic_sha256", None)
    encoded = (json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump_semantic(path: pathlib.Path, value: dict[str, Any]) -> None:
    payload = dict(value)
    payload["semantic_sha256"] = canonical_sha256(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))


def physical_metrics(action: np.ndarray) -> dict[str, Any]:
    value = np.asarray(action, dtype=np.float64).reshape(PHASE_COUNT, len(LAYER_INDICES), HIDDEN_SIZE)
    if not np.isfinite(value).all():
        raise FloatingPointError("physical action contains non-finite values")
    writes = np.linalg.norm(value, axis=-1)
    bridge = float(np.square(writes[0]).sum())
    answer = float(np.square(writes[1:]).sum())
    total = bridge + answer
    passed = bool(
        writes.max() <= MAXIMUM_PER_WRITE_RATIO + 1e-7
        and bridge <= BRIDGE_BUDGET + 1e-7
        and answer <= ANSWER_BUDGET + 1e-7
        and total <= TOTAL_BUDGET + 1e-7
    )
    return {
        "write_norms": writes,
        "bridge": bridge,
        "answer": answer,
        "total": total,
        "maximum_write": float(writes.max()),
        "pass": passed,
    }


def legal_radial_energy_cap(direction: np.ndarray, internal_cap: float = INTERNAL_SEARCH_CAP) -> float:
    value = np.asarray(direction, dtype=np.float64).reshape(PHASE_COUNT, len(LAYER_INDICES), HIDDEN_SIZE)
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("finite nonzero direction required")
    unit = value / norm
    write_fraction = np.square(np.linalg.norm(unit, axis=-1))
    bridge_fraction = float(write_fraction[0].sum())
    answer_fraction = float(write_fraction[1:].sum())
    caps = [float(internal_cap)]
    if bridge_fraction > 0:
        caps.append(BRIDGE_BUDGET / bridge_fraction)
    if answer_fraction > 0:
        caps.append(ANSWER_BUDGET / answer_fraction)
    positive = write_fraction[write_fraction > 0]
    if positive.size:
        caps.append(float((MAXIMUM_PER_WRITE_RATIO**2 / positive).min()))
    return max(0.0, min(caps))


def behavior_metrics(scores: Iterable[float], zero_scores: Iterable[float], target: int) -> dict[str, Any]:
    value = np.asarray(list(scores), dtype=np.float64)
    zero = np.asarray(list(zero_scores), dtype=np.float64)
    if value.shape != (64,) or zero.shape != (64,) or not np.isfinite(value).all() or not np.isfinite(zero).all():
        raise ValueError("finite aligned Z64 scores required")
    mask = np.arange(64) != int(target)
    margin = float(value[target] - value[mask].max())
    effect = value - zero
    effect_margin = float(effect[target] - effect[mask].max())
    rank = int(1 + np.sum(value > value[target]))
    return {
        "rank": rank,
        "mrr": 1.0 / rank,
        "margin": margin,
        "effect_margin": effect_margin,
        "effect_peak": bool(effect_margin >= 0.0),
    }


def full_behavior_certificate(
    action: np.ndarray,
    scores_fp32: Iterable[float],
    scores_bf16: Iterable[float],
    zero_fp32: Iterable[float],
    zero_bf16: Iterable[float],
    target: int,
) -> dict[str, Any]:
    fp32 = behavior_metrics(scores_fp32, zero_fp32, target)
    bf16 = behavior_metrics(scores_bf16, zero_bf16, target)
    physical = physical_metrics(action)
    passed = bool(
        fp32["rank"] == 1
        and bf16["rank"] == 1
        and fp32["margin"] >= ROBUST_MARGIN
        and bf16["margin"] >= ROBUST_MARGIN
        and fp32["effect_peak"]
        and bf16["effect_peak"]
        and physical["total"] <= QUALIFICATION_P99_BUDGET + 1e-7
        and physical["pass"]
    )
    return {
        "pass": passed,
        "fp32": fp32,
        "bf16": bf16,
        "physical": {key: value for key, value in physical.items() if key != "write_norms"},
    }
