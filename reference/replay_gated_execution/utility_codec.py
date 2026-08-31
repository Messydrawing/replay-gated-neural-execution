"""Closed JSON codec for the finite Z64 UtilityIR.

This module is deliberately label-free.  A runtime request already contains a
fully compiled Utility; it never contains a benchmark target, candidate index,
capability name, effect id, or qualification outcome.
"""

from __future__ import annotations

from typing import Any

from .semantic import UtilityCell, UtilityIR, UtilityPredicate


REGISTRY_VERSION = "Z64-sequence-score-v1"
REGISTRY_MEMBERS = tuple(f" {index:02d}" for index in range(64))


def utility_from_json(value: dict[str, Any]) -> UtilityIR:
    if set(value) != {"registry_version", "cells"}:
        raise ValueError("closed UtilityIR fields required")
    if value["registry_version"] != REGISTRY_VERSION:
        raise ValueError("unknown Z64 registry identity")
    cells = []
    for raw_cell in value["cells"]:
        if set(raw_cell) != {"predicates"}:
            raise ValueError("closed Utility cell fields required")
        predicates = []
        for raw in raw_cell["predicates"]:
            if set(raw) != {"coefficients", "threshold", "observation_form"}:
                raise ValueError("closed Utility predicate fields required")
            predicates.append(UtilityPredicate(
                tuple(float(item) for item in raw["coefficients"]),
                float(raw["threshold"]),
                str(raw["observation_form"]),
            ))
        cells.append(UtilityCell(tuple(predicates)))
    result = UtilityIR(REGISTRY_VERSION, tuple(cells))
    if result.registry_width != 64:
        raise ValueError("Z64 Utility width mismatch")
    return result


def utility_to_json(value: UtilityIR) -> dict[str, Any]:
    if value.registry_version != REGISTRY_VERSION or value.registry_width != 64:
        raise ValueError("Z64 Utility required")
    return {
        "registry_version": value.registry_version,
        "cells": [{
            "predicates": [{
                "coefficients": list(predicate.coefficients),
                "threshold": float(predicate.threshold),
                "observation_form": predicate.observation_form,
            } for predicate in cell.predicates],
        } for cell in value.cells],
    }


def build_margin_utility(semantic_slot: int, margin: float = 0.2) -> UtilityIR:
    """Freeze-time helper; never imported by the K1 numerical runner."""
    if isinstance(semantic_slot, bool) or not 0 <= int(semantic_slot) < 64:
        raise ValueError("semantic slot outside Z64")
    predicates = []
    for other in range(64):
        if other == int(semantic_slot):
            continue
        coefficients = [0.0] * 64
        coefficients[int(semantic_slot)] = 1.0
        coefficients[other] = -1.0
        predicates.append(UtilityPredicate(tuple(coefficients), float(margin), "absolute"))
    return UtilityIR(REGISTRY_VERSION, (UtilityCell(tuple(predicates)),))
