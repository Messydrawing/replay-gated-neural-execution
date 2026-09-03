"""Typed Utility semantics.  Benchmark qualification concepts are absent here."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .canonical import canonical_sha256

FloatArray = NDArray[np.float64]


def _finite_vector(name: str, value: FloatArray, width: int | None = None) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or (width is not None and result.shape != (width,)):
        raise ValueError(f"{name} has an invalid shape")
    if not np.isfinite(result).all():
        raise FloatingPointError(f"{name} is non-finite")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class UtilityPredicate:
    """One public affine predicate over a versioned semantic registry."""

    coefficients: tuple[float, ...]
    threshold: float
    observation_form: str = "absolute"

    def __post_init__(self) -> None:
        values = np.asarray(self.coefficients, dtype=np.float64)
        if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
            raise ValueError("Utility predicate coefficients must be finite and nonempty")
        if abs(float(values.sum())) > 1e-12:
            raise ValueError("registry-affine coefficients must sum to zero")
        if self.observation_form not in {"absolute", "delta"}:
            raise ValueError("unknown observation form")
        if not np.isfinite(float(self.threshold)):
            raise ValueError("predicate threshold must be finite")

    @property
    def key(self) -> str:
        return canonical_sha256({
            "coefficients": self.coefficients,
            "threshold": self.threshold,
            "observation_form": self.observation_form,
        })


@dataclass(frozen=True)
class UtilityCell:
    predicates: tuple[UtilityPredicate, ...]

    def __post_init__(self) -> None:
        if not self.predicates:
            raise ValueError("a Utility conjunction cell cannot be empty")
        if len({len(item.coefficients) for item in self.predicates}) != 1:
            raise ValueError("predicate registry widths differ")


@dataclass(frozen=True)
class UtilityIR:
    registry_version: str
    cells: tuple[UtilityCell, ...]

    def __post_init__(self) -> None:
        if not self.registry_version or not self.cells:
            raise ValueError("UtilityIR requires a registry identity and cells")
        widths = {len(predicate.coefficients) for cell in self.cells for predicate in cell.predicates}
        if len(widths) != 1:
            raise ValueError("UtilityIR registry widths differ")

    @property
    def registry_width(self) -> int:
        return len(self.cells[0].predicates[0].coefficients)

    @property
    def semantic_sha256(self) -> str:
        return canonical_sha256(self)


@dataclass(frozen=True)
class RuntimeTelemetry:
    values: tuple[float, ...]
    provenance_sha256: str

    def __post_init__(self) -> None:
        if not self.provenance_sha256 or not np.isfinite(np.asarray(self.values, dtype=float)).all():
            raise ValueError("invalid telemetry")


@dataclass(frozen=True)
class ExecutionAuthority:
    lease_active: bool
    scope_authorized: bool
    lease_sha256: str
    scope_sha256: str

    def __post_init__(self) -> None:
        if not self.lease_sha256 or not self.scope_sha256:
            raise ValueError("authority identities are required")


@dataclass(frozen=True)
class RuntimeSemanticObservation:
    registry_version: str
    precision: str
    values: FloatArray

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _finite_vector("semantic observation", self.values))
        if self.precision not in {"fp32", "bf16"}:
            raise ValueError("precision must be fp32 or bf16")


@dataclass(frozen=True)
class RuntimeConstraintVector:
    precision: str
    cell_index: int
    keys: tuple[str, ...]
    values: FloatArray

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _finite_vector("constraint vector", self.values))
        if len(self.keys) != len(self.values) or len(set(self.keys)) != len(self.keys):
            raise ValueError("constraint keys and values are inconsistent")


class RuntimeSemanticOracle(Protocol):
    """Normative execution interface visible to the solver."""

    def observe(self, physical_action: FloatArray, precision: str) -> RuntimeSemanticObservation:
        ...


def compile_cell_constraints(utility: UtilityIR, cell_index: int,
                             observation: RuntimeSemanticObservation,
                             exact_zero: RuntimeSemanticObservation) -> RuntimeConstraintVector:
    if observation.registry_version != utility.registry_version or exact_zero.registry_version != utility.registry_version:
        raise ValueError("semantic registry identity mismatch")
    if observation.precision != exact_zero.precision:
        raise ValueError("precision mismatch")
    if observation.values.shape != (utility.registry_width,) or exact_zero.values.shape != observation.values.shape:
        raise ValueError("semantic observation width mismatch")
    cell = utility.cells[cell_index]
    values = []
    keys = []
    for predicate in cell.predicates:
        coeff = np.asarray(predicate.coefficients, dtype=np.float64)
        source = observation.values if predicate.observation_form == "absolute" else observation.values - exact_zero.values
        values.append(float(coeff @ source - predicate.threshold))
        keys.append(predicate.key)
    return RuntimeConstraintVector(observation.precision, int(cell_index), tuple(keys), np.asarray(values))
