"""Candidate-blind adapter from a canonical semantic registry to Utility constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
from numpy.typing import NDArray

from .semantic import RuntimeConstraintVector, UtilityIR

FloatArray = NDArray[np.float64]
RegistryReplay = Callable[[FloatArray], FloatArray]
RegistryJacobian = Callable[[FloatArray], tuple[FloatArray, FloatArray]]
ExactZeroCheck = Callable[[], bool]


@dataclass(frozen=True)
class RegistryExecutionCallbacks:
    registry_version: str
    registry_width: int
    singleton_replays: Mapping[str, RegistryReplay]
    fp32_registry_jacobian: RegistryJacobian
    exact_zero_check: ExactZeroCheck


class RegistryUtilityOracle:
    """Translate fixed registry observations into metadata-free constraints.

    The callbacks and registry order are execution-bound before a Utility is
    supplied.  No evaluation-plane object is accepted by this adapter.
    """

    def __init__(self, callbacks: RegistryExecutionCallbacks, action_width: int) -> None:
        if set(callbacks.singleton_replays) != {"fp32", "bf16"}:
            raise ValueError("exact dual singleton registry replays are required")
        self.callbacks = callbacks
        self.action_width = int(action_width)
        zero = np.zeros(self.action_width, dtype=np.float64)
        self._zero = {name: self._observe(zero, name)
                      for name in ("fp32", "bf16")}

    def exact_zero_is_bitwise(self) -> bool:
        return bool(self.callbacks.exact_zero_check())

    def _observe(self, physical_action: FloatArray, precision: str) -> FloatArray:
        values = np.asarray(self.callbacks.singleton_replays[precision](physical_action),
                            dtype=np.float64).reshape(-1)
        if values.shape != (self.callbacks.registry_width,) or not np.isfinite(values).all():
            raise FloatingPointError("invalid canonical registry observation")
        return values - values.mean()

    def replay(self, utility: UtilityIR, cell_index: int, physical_action: FloatArray,
               precision: str) -> RuntimeConstraintVector:
        if utility.registry_version != self.callbacks.registry_version:
            raise ValueError("Utility/runtime registry mismatch")
        observation = self._observe(physical_action, precision)
        values = []
        keys = []
        for predicate in utility.cells[cell_index].predicates:
            coefficient = np.asarray(predicate.coefficients, dtype=np.float64)
            source = observation if predicate.observation_form == "absolute" else observation - self._zero[precision]
            values.append(float(coefficient @ source - predicate.threshold))
            keys.append(predicate.key)
        return RuntimeConstraintVector(precision, cell_index, tuple(keys), np.asarray(values))

    def fp32_action_jacobian(self, utility: UtilityIR, cell_index: int,
                             physical_action: FloatArray) -> FloatArray:
        observation, registry_jacobian = self.callbacks.fp32_registry_jacobian(physical_action)
        observation = np.asarray(observation, dtype=np.float64).reshape(-1)
        jacobian = np.asarray(registry_jacobian, dtype=np.float64)
        if observation.shape != (self.callbacks.registry_width,) or jacobian.shape != (
                self.callbacks.registry_width, self.action_width):
            raise ValueError("canonical registry Jacobian shape mismatch")
        if not np.isfinite(observation).all() or not np.isfinite(jacobian).all():
            raise FloatingPointError("canonical registry Jacobian is invalid")
        cell = utility.cells[cell_index]
        return np.stack([np.asarray(item.coefficients, dtype=np.float64) @ jacobian
                         for item in cell.predicates])
