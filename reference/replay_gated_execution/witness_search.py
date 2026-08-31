from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch


class WitnessMachine(Protocol):
    device: torch.device
    fp32_model: Any
    bf16_model: Any
    contract: Any

    def new_schedule(self, batch_size: int, seed: int) -> Any: ...
    def hook_manager(self, model: Any, schedule: Any) -> Any: ...
    def scores(self, schedule: Any, model: Any, hooks: Any, batch_size: int) -> torch.Tensor: ...


class WitnessObjective(Protocol):
    def raw_to_state(self, action: np.ndarray, template: Any) -> Any: ...
    def state_to_raw(self, state: Any) -> np.ndarray: ...
    def project(self, schedule: Any, contract: Any) -> None: ...
    def torch_metrics(self, scores: torch.Tensor, zero: torch.Tensor, gold: torch.Tensor) -> dict[str, torch.Tensor]: ...
    def loss_from(self, metrics32: dict[str, torch.Tensor], metricsbf: dict[str, torch.Tensor], scores32: torch.Tensor, scoresbf: torch.Tensor, gold: torch.Tensor) -> torch.Tensor: ...
    def certificate(self, action: np.ndarray, scores32: np.ndarray, scoresbf: np.ndarray, zero32: np.ndarray, zerobf: np.ndarray, target: int) -> dict[str, Any]: ...
    def criterion_key(self, certificate: dict[str, Any], step: int) -> tuple[float, ...]: ...


@dataclass(frozen=True)
class WitnessSearchResult:
    normal_certificates: dict[int, dict[str, Any]]
    normal_actions: dict[int, np.ndarray]
    best_certificates: dict[int, dict[str, Any]]
    best_actions: dict[int, np.ndarray]
    first_certified_steps: dict[int, int | None]


def adaptive_witness_search(
    *,
    machine: WitnessMachine,
    objective: WitnessObjective,
    targets: list[int],
    starts: dict[int, np.ndarray],
    zero_fp32: np.ndarray,
    zero_bf16: np.ndarray,
    seed: int,
    learning_rate: float = 0.004,
    normal_steps: int = 97,
    maximum_steps: int = 388,
) -> WitnessSearchResult:
    """Frozen analytic -> 97 -> 388 lifecycle used by the full slow path.

    The function contains no capability/action table and owns a fresh optimizer.
    A caller may pass a singleton or a small batch, but every target starts from
    an explicitly supplied public initializer rather than a stored witness.
    """
    schedule = machine.new_schedule(len(targets), seed)
    for local, target in enumerate(targets):
        schedule.import_utility(
            local,
            objective.raw_to_state(starts[target], schedule.export_utility(local)),
        )
    objective.project(schedule, machine.contract)
    optimizer = torch.optim.Adam(schedule.parameters(), lr=learning_rate)
    gold = torch.tensor(targets, dtype=torch.long, device=machine.device)
    zero32_t = torch.from_numpy(np.repeat(zero_fp32[None], len(targets), axis=0)).to(machine.device)
    zerobf_t = torch.from_numpy(np.repeat(zero_bf16[None], len(targets), axis=0)).to(machine.device)
    best_actions = {target: starts[target].copy() for target in targets}
    best_certificates: dict[int, dict[str, Any]] = {}
    first_pass = {target: None for target in targets}
    normal_actions: dict[int, np.ndarray] = {}
    normal_certificates: dict[int, dict[str, Any]] = {}
    hooks32 = machine.hook_manager(machine.fp32_model, schedule)
    hooksbf = machine.hook_manager(machine.bf16_model, schedule)
    hooks32.install()
    hooksbf.install()
    try:
        for step in range(maximum_steps):
            optimizer.zero_grad(set_to_none=True)
            scoresbf = machine.scores(schedule, machine.bf16_model, hooksbf, len(targets))
            scores32 = machine.scores(schedule, machine.fp32_model, hooks32, len(targets))
            metrics32 = objective.torch_metrics(scores32, zero32_t, gold)
            metricsbf = objective.torch_metrics(scoresbf, zerobf_t, gold)
            loss = objective.loss_from(metrics32, metricsbf, scores32, scoresbf, gold)
            for local, target in enumerate(targets):
                action = objective.state_to_raw(schedule.export_utility(local))
                cert = objective.certificate(
                    action,
                    scores32[local].detach().float().cpu().numpy(),
                    scoresbf[local].detach().float().cpu().numpy(),
                    zero_fp32,
                    zero_bf16,
                    target,
                )
                if target not in best_certificates or objective.criterion_key(cert, step) > objective.criterion_key(best_certificates[target], step):
                    best_certificates[target] = cert
                    best_actions[target] = action.copy()
                if cert["pass"] and first_pass[target] is None:
                    first_pass[target] = step
            if step == normal_steps - 1:
                normal_actions = {target: action.copy() for target, action in best_actions.items()}
                normal_certificates = {target: dict(cert) for target, cert in best_certificates.items()}
            if step + 1 < maximum_steps:
                loss.backward()
                optimizer.step()
                objective.project(schedule, machine.contract)
            if step >= normal_steps - 1 and all(cert["pass"] for cert in best_certificates.values()):
                break
    finally:
        hooks32.uninstall()
        hooksbf.uninstall()
    if not normal_certificates:
        normal_actions = {target: action.copy() for target, action in best_actions.items()}
        normal_certificates = {target: dict(cert) for target, cert in best_certificates.items()}
    return WitnessSearchResult(
        normal_certificates,
        normal_actions,
        best_certificates,
        best_actions,
        first_pass,
    )
