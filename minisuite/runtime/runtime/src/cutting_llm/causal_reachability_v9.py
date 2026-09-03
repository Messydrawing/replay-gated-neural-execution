from __future__ import annotations

from dataclasses import dataclass
import math
import types
from typing import Any, Iterable, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .scoring import resolve_decoder_and_output_head


@dataclass(frozen=True)
class ReachabilityEnergyContract:
    """Exact Single-Boundary Causal Lease energy contract for E-S0a."""

    bridge_budget: float = 0.006
    answer_budget: float = 0.024
    total_budget: float = 0.03
    maximum_per_write_ratio: float = 0.05
    epsilon: float = 1e-12

    def effective_answer_budget(
        self, *, scored_answer_transitions: int, receiver_layer_count: int
    ) -> float:
        support_limit = (
            int(scored_answer_transitions)
            * int(receiver_layer_count)
            * self.maximum_per_write_ratio**2
        )
        return min(self.answer_budget, support_limit)


def token_boundary(
    tokenizer: Any, prompt: str, address_prefix: str
) -> tuple[list[int], int]:
    encoded = tokenizer(
        prompt,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = [tuple(map(int, pair)) for pair in encoded["offset_mapping"]]
    boundary_character = len(address_prefix)
    if any(start < boundary_character < end for start, end in offsets):
        raise ValueError("E-S0a found a token crossing the causal boundary")
    address = [index for index, (_start, end) in enumerate(offsets) if end <= boundary_character]
    bridge = [index for index, (start, _end) in enumerate(offsets) if start >= boundary_character]
    if not address or not bridge or bridge[0] != address[-1] + 1:
        raise ValueError("E-S0a found an invalid address/bridge boundary")
    return list(map(int, encoded["input_ids"])), address[-1]


def candidate_token_ids(
    tokenizer: Any, answers: Sequence[str], device: torch.device
) -> Tensor:
    encoded = [tokenizer(answer, add_special_tokens=False)["input_ids"] for answer in answers]
    lengths = {len(value) for value in encoded}
    if not encoded or len(lengths) != 1 or next(iter(lengths)) <= 0:
        raise ValueError("E-S0a requires equal, non-empty candidate token sequences")
    if len({tuple(value) for value in encoded}) != len(encoded):
        raise ValueError("E-S0a candidate token sequences must be unique")
    return torch.tensor(encoded, dtype=torch.long, device=device)


def _parameter_name(
    utility_index: int,
    phase: str,
    position: int,
    layer_index: int,
    prefix_index: int = 0,
) -> str:
    return (
        f"u{utility_index:03d}__{phase}__p{position:03d}"
        f"__x{prefix_index:03d}__l{layer_index:03d}"
    )


class CausalRatioActionSchedule(nn.Module):
    """Dense oracle actions expressed directly in the audited ratio units.

    Answer parameters are indexed by the *consumed* candidate prefix.  All
    branches with the same consumed prefix therefore share one action.  This
    is the structural next-token constraint that prevents candidate identity
    from leaking through branch-specific parameters.
    """

    def __init__(
        self,
        *,
        utility_count: int,
        hidden_size: int,
        layer_indices: Sequence[int],
        bridge_token_count: int,
        candidate_ids: Tensor,
        seed: int = 1701,
        perturbation_scale: float = 0.0,
        candidate_independent_answer: bool = False,
    ) -> None:
        super().__init__()
        if utility_count <= 0 or hidden_size <= 0 or bridge_token_count <= 0:
            raise ValueError("E-S0a schedule dimensions must be positive")
        if candidate_ids.ndim != 2 or candidate_ids.shape[0] < 2:
            raise ValueError("E-S0a schedule requires [candidate,token] IDs")
        self.utility_count = int(utility_count)
        self.hidden_size = int(hidden_size)
        self.layer_indices = tuple(map(int, layer_indices))
        self.bridge_token_count = int(bridge_token_count)
        self.candidate_ids_cpu = candidate_ids.detach().long().cpu()
        self.candidate_independent_answer = bool(candidate_independent_answer)
        self.ratios = nn.ParameterDict()
        self._bridge_names: dict[tuple[int, int, int], str] = {}
        self._answer_names: dict[tuple[int, int, tuple[int, ...], int], str] = {}
        self._answer_prefix_indices: dict[tuple[int, tuple[int, ...]], int] = {}

        generator = torch.Generator().manual_seed(int(seed))

        def initial() -> Tensor:
            if perturbation_scale == 0:
                return torch.zeros(self.hidden_size, dtype=torch.float32)
            value = torch.randn(self.hidden_size, generator=generator, dtype=torch.float32)
            value = value / value.norm().clamp_min(1e-12)
            return float(perturbation_scale) * value

        for utility in range(self.utility_count):
            for position in range(self.bridge_token_count):
                for layer in self.layer_indices:
                    name = _parameter_name(utility, "bridge", position, layer)
                    self._bridge_names[(utility, position, layer)] = name
                    self.ratios[name] = nn.Parameter(initial())

            for position in range(int(self.candidate_ids_cpu.shape[1]) - 1):
                prefixes = (
                    [()]
                    if self.candidate_independent_answer
                    else sorted(
                        {
                            tuple(map(int, row[: position + 1].tolist()))
                            for row in self.candidate_ids_cpu
                        }
                    )
                )
                for prefix_index, prefix in enumerate(prefixes):
                    self._answer_prefix_indices[(position, prefix)] = prefix_index
                    for layer in self.layer_indices:
                        name = _parameter_name(
                            utility,
                            "answer",
                            position,
                            layer,
                            prefix_index,
                        )
                        self._answer_names[(utility, position, prefix, layer)] = name
                        self.ratios[name] = nn.Parameter(initial())

    def bridge_ratio(self, utility: int, position: int, layer: int) -> Tensor:
        return self.ratios[self._bridge_names[(int(utility), int(position), int(layer))]]

    def answer_ratio(
        self,
        utility: int,
        position: int,
        prefix: Sequence[int],
        layer: int,
    ) -> Tensor:
        normalized_prefix = (
            () if self.candidate_independent_answer else tuple(map(int, prefix))
        )
        key = (int(utility), int(position), normalized_prefix, int(layer))
        return self.ratios[self._answer_names[key]]

    def utility_parameters(self, utilities: Iterable[int]) -> list[nn.Parameter]:
        wanted = {int(value) for value in utilities}
        result: list[nn.Parameter] = []
        for name, parameter in self.ratios.items():
            utility = int(name[1:4])
            if utility in wanted:
                result.append(parameter)
        return result

    def export_utility(self, utility: int) -> dict[str, Tensor]:
        prefix = f"u{int(utility):03d}__"
        return {
            name[len(prefix) :]: value.detach().cpu().clone()
            for name, value in self.ratios.items()
            if name.startswith(prefix)
        }

    def import_utility(self, utility: int, values: dict[str, Tensor]) -> None:
        prefix = f"u{int(utility):03d}__"
        expected = {
            name[len(prefix) :]
            for name in self.ratios
            if name.startswith(prefix)
        }
        if set(values) != expected:
            raise ValueError("E-S0a utility action sidecar has a different support")
        with torch.no_grad():
            for suffix, value in values.items():
                self.ratios[prefix + suffix].copy_(
                    value.to(
                        device=self.ratios[prefix + suffix].device,
                        dtype=self.ratios[prefix + suffix].dtype,
                    )
                )

    def multiply_(self, factor: float) -> None:
        if not math.isfinite(factor) or factor < 0:
            raise ValueError("E-S0a action scale must be finite and non-negative")
        with torch.no_grad():
            for parameter in self.ratios.values():
                parameter.mul_(float(factor))

    def energy_summary(self, utility: int) -> dict[str, Any]:
        utility = int(utility)
        bridge = sum(
            float(self.bridge_ratio(utility, position, layer).float().square().sum().item())
            for position in range(self.bridge_token_count)
            for layer in self.layer_indices
        )
        answer_paths: list[float] = []
        for candidate in self.candidate_ids_cpu:
            energy = 0.0
            for position in range(int(candidate.shape[0]) - 1):
                prefix = (
                    ()
                    if self.candidate_independent_answer
                    else tuple(map(int, candidate[: position + 1].tolist()))
                )
                for layer in self.layer_indices:
                    energy += float(
                        self.answer_ratio(
                            utility, position, prefix, layer
                        ).float().square().sum().item()
                    )
            answer_paths.append(energy)
        maximum_write = max(
            (
                float(parameter.float().norm().item())
                for name, parameter in self.ratios.items()
                if name.startswith(f"u{utility:03d}__")
            ),
            default=0.0,
        )
        maximum_answer = max(answer_paths, default=0.0)
        return {
            "bridge_energy": bridge,
            "maximum_answer_path_energy": maximum_answer,
            "maximum_total_path_energy": bridge + maximum_answer,
            "minimum_answer_path_energy": min(answer_paths, default=0.0),
            "maximum_per_write_ratio": maximum_write,
            "answer_path_energies": answer_paths,
        }

    def project_(
        self,
        *,
        contract: ReachabilityEnergyContract,
        bridge_cap: float,
        answer_cap: float,
    ) -> None:
        if not 0 <= bridge_cap <= contract.bridge_budget + 1e-15:
            raise ValueError("E-S0a bridge cap exceeds the formal lease")
        if not 0 <= answer_cap <= contract.answer_budget + 1e-15:
            raise ValueError("E-S0a answer cap exceeds the formal lease")
        with torch.no_grad():
            for parameter in self.ratios.values():
                norm = parameter.float().norm()
                if float(norm.item()) > contract.maximum_per_write_ratio:
                    parameter.mul_(contract.maximum_per_write_ratio / norm)

            for utility in range(self.utility_count):
                bridge_parameters = [
                    self.bridge_ratio(utility, position, layer)
                    for position in range(self.bridge_token_count)
                    for layer in self.layer_indices
                ]
                bridge_energy = sum(
                    float(value.float().square().sum().item())
                    for value in bridge_parameters
                )
                if bridge_energy > bridge_cap and bridge_energy > 0:
                    scale = math.sqrt(bridge_cap / bridge_energy)
                    for value in bridge_parameters:
                        value.mul_(scale)

                answer_parameters = [
                    parameter
                    for name, parameter in self.ratios.items()
                    if name.startswith(f"u{utility:03d}__answer__")
                ]
                maximum_path = float(
                    self.energy_summary(utility)["maximum_answer_path_energy"]
                )
                if maximum_path > answer_cap and maximum_path > 0:
                    scale = math.sqrt(answer_cap / maximum_path)
                    for value in answer_parameters:
                        value.mul_(scale)


class CausalDirectActionHookManager:
    """Inject a ratio schedule at layer inputs without a receiver or state."""

    def __init__(self, model: nn.Module, schedule: CausalRatioActionSchedule) -> None:
        decoder, _ = resolve_decoder_and_output_head(model)
        layers = getattr(decoder, "layers", None)
        if not isinstance(layers, nn.ModuleList):
            raise ValueError("E-S0a requires a decoder ModuleList")
        if schedule.layer_indices[-1] >= len(layers):
            raise ValueError("E-S0a receiver layer is outside the decoder")
        self.model = model
        self.layers = layers
        self.schedule = schedule
        self._original_forwards: dict[int, Any] = {}
        self._installed = False
        self._active = False
        self._phase = ""
        self._utilities: list[int] = []
        self._position = 0
        self._prefixes: list[tuple[int, ...]] = []
        self.last_realized_write_ratios: list[float] = []

    @property
    def installed(self) -> bool:
        return self._installed

    @property
    def active(self) -> bool:
        return self._active

    def install(self) -> None:
        if self._installed:
            return
        for layer_index in self.schedule.layer_indices:
            layer = self.layers[layer_index]
            original = layer.forward
            self._original_forwards[layer_index] = original

            def patched(
                module: nn.Module,
                hidden_states: Tensor,
                *args: Any,
                _layer_index: int = layer_index,
                _original: Any = original,
                **kwargs: Any,
            ) -> Any:
                if not self._active:
                    return _original(hidden_states, *args, **kwargs)
                return self._active_layer_forward(
                    _layer_index, _original, hidden_states, *args, **kwargs
                )

            layer.forward = types.MethodType(patched, layer)
        self._installed = True

    def uninstall(self) -> None:
        self.deactivate()
        if not self._installed:
            return
        for layer_index, original in self._original_forwards.items():
            self.layers[layer_index].forward = original
        self._original_forwards.clear()
        self._installed = False

    def activate_bridge(self, utilities: Sequence[int], position: int) -> None:
        self._activate("bridge", utilities, position, ())

    def activate_answer(
        self,
        utilities: Sequence[int],
        position: int,
        prefixes: Sequence[Sequence[int]],
    ) -> None:
        self._activate("answer", utilities, position, prefixes)

    def _activate(
        self,
        phase: str,
        utilities: Sequence[int],
        position: int,
        prefixes: Sequence[Sequence[int]],
    ) -> None:
        if not self._installed or self._active:
            raise RuntimeError("E-S0a hooks are not ready for activation")
        if phase not in {"bridge", "answer"} or not utilities:
            raise ValueError("E-S0a activation is invalid")
        self._phase = phase
        self._utilities = list(map(int, utilities))
        self._position = int(position)
        self._prefixes = [tuple(map(int, value)) for value in prefixes]
        if phase == "answer" and len(self._prefixes) != len(self._utilities):
            raise ValueError("E-S0a answer prefixes do not align with branches")
        self.last_realized_write_ratios = []
        self._active = True

    def deactivate(self) -> None:
        self._active = False
        self._phase = ""
        self._utilities = []
        self._position = 0
        self._prefixes = []

    def _active_layer_forward(
        self,
        layer_index: int,
        original: Any,
        hidden_states: Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if hidden_states.ndim != 3 or hidden_states.shape[1] != 1:
            raise RuntimeError("E-S0a control requires one-token KV continuation")
        if hidden_states.shape[0] != len(self._utilities):
            raise RuntimeError("E-S0a active batch differs from action schedule")
        values = []
        for index, utility in enumerate(self._utilities):
            if self._phase == "bridge":
                value = self.schedule.bridge_ratio(
                    utility, self._position, layer_index
                )
            else:
                value = self.schedule.answer_ratio(
                    utility,
                    self._position,
                    self._prefixes[index],
                    layer_index,
                )
            values.append(value)
        ratios = torch.stack(values, dim=0)
        prewrite = hidden_states[:, 0, :].float()
        scale = torch.sqrt(
            prewrite.square().sum(dim=-1, keepdim=True)
            + ReachabilityEnergyContract().epsilon
        )
        delta = (scale * ratios.float()).to(hidden_states.dtype)
        realized = delta.float().norm(dim=-1) / scale.squeeze(-1)
        self.last_realized_write_ratios.extend(
            map(float, realized.detach().cpu().tolist())
        )
        return original(hidden_states + delta.unsqueeze(1), *args, **kwargs)


def causal_direct_candidate_scores(
    *,
    model: nn.Module,
    tokenizer: Any,
    schedule: CausalRatioActionSchedule,
    hooks: CausalDirectActionHookManager,
    row: dict[str, Any],
    utility_indices: Sequence[int],
    control_enabled: bool = True,
) -> Tensor:
    """Full-candidate causal scoring for one same-input counterfactual group."""

    if not hooks.installed or hooks.active:
        raise RuntimeError("E-S0a scoring requires installed, inactive hooks")
    if not utility_indices:
        raise ValueError("E-S0a scoring requires at least one utility")
    device = next(model.parameters()).device
    prompt_ids, address_end = token_boundary(
        tokenizer, str(row["prompt"]), str(row["address_prefix"])
    )
    prefix_ids = torch.tensor(
        [prompt_ids[: address_end + 1]], dtype=torch.long, device=device
    )
    bridge_ids = prompt_ids[address_end + 1 :]
    if len(bridge_ids) != schedule.bridge_token_count:
        raise ValueError("E-S0a schedule bridge support differs from the prompt")
    candidates = candidate_token_ids(tokenizer, row["candidate_answers"], device)
    if not torch.equal(candidates.detach().cpu(), schedule.candidate_ids_cpu):
        raise ValueError("E-S0a schedule candidate support differs from scoring")
    decoder, output_head = resolve_decoder_and_output_head(model)
    prefix_attention = torch.ones_like(prefix_ids)
    prefix_outputs = decoder(
        input_ids=prefix_ids,
        attention_mask=prefix_attention,
        use_cache=True,
        return_dict=True,
    )
    cache = prefix_outputs.past_key_values
    batch_size = len(utility_indices)
    if batch_size > 1:
        cache.batch_repeat_interleave(batch_size)
    last_hidden = prefix_outputs.last_hidden_state[:, -1, :].repeat(
        batch_size, 1
    )
    try:
        for offset, token_id in enumerate(bridge_ids):
            if control_enabled:
                hooks.activate_bridge(utility_indices, offset)
            attention = torch.ones(
                batch_size,
                prefix_ids.shape[1] + offset + 1,
                dtype=prefix_attention.dtype,
                device=device,
            )
            outputs = decoder(
                input_ids=torch.full(
                    (batch_size, 1), int(token_id), dtype=torch.long, device=device
                ),
                attention_mask=attention,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            if control_enabled:
                hooks.deactivate()
            cache = outputs.past_key_values
            last_hidden = outputs.last_hidden_state[:, -1, :]
    finally:
        hooks.deactivate()

    first_logits = output_head(last_hidden).float()
    first_log_probabilities = F.log_softmax(first_logits, dim=-1)
    scores = first_log_probabilities[:, candidates[:, 0]]
    candidate_count, candidate_length = map(int, candidates.shape)
    cache.batch_repeat_interleave(candidate_count)
    flat_candidates = candidates.unsqueeze(0).expand(
        batch_size, candidate_count, candidate_length
    ).reshape(batch_size * candidate_count, candidate_length)
    branch_utilities = [
        utility for utility in utility_indices for _ in range(candidate_count)
    ]
    try:
        for position in range(candidate_length - 1):
            prefixes = [
                tuple(map(int, row_ids[: position + 1].tolist()))
                for row_ids in flat_candidates
            ]
            if control_enabled:
                hooks.activate_answer(
                    branch_utilities,
                    position,
                    prefixes,
                )
            attention = torch.ones(
                batch_size * candidate_count,
                len(prompt_ids) + position + 1,
                dtype=prefix_attention.dtype,
                device=device,
            )
            outputs = decoder(
                input_ids=flat_candidates[:, position : position + 1],
                attention_mask=attention,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            if control_enabled:
                hooks.deactivate()
            cache = outputs.past_key_values
            hidden = outputs.last_hidden_state[:, -1, :]
            logits = output_head(hidden).float()
            token_log_probabilities = F.log_softmax(logits, dim=-1).gather(
                1, flat_candidates[:, position + 1 : position + 2]
            ).reshape(batch_size, candidate_count)
            scores = scores + token_log_probabilities
    finally:
        hooks.deactivate()
    return scores / candidate_length


def candidate_rank_metrics(scores: Tensor, gold_indices: Tensor) -> dict[str, Tensor]:
    if scores.ndim != 2 or gold_indices.shape != (scores.shape[0],):
        raise ValueError("E-S0a rank metrics require [batch,candidate] scores")
    gold = scores.gather(1, gold_indices.unsqueeze(1)).squeeze(1)
    mask = F.one_hot(gold_indices, num_classes=scores.shape[1]).bool()
    other = scores.masked_fill(mask, -torch.inf).max(dim=-1).values
    ranks = 1 + (scores > gold.unsqueeze(1)).sum(dim=-1)
    return {
        "rank": ranks,
        "mrr": 1.0 / ranks.float(),
        "margin": gold - other,
        "top1": ranks.eq(1),
    }


def score_effect_binding(
    controlled_scores: Tensor,
    zero_scores: Tensor,
    gold_indices: Tensor,
) -> dict[str, Tensor]:
    """Zero-relative full-64 effect matrix and symmetric double centering."""

    if controlled_scores.ndim != 2 or zero_scores.shape != controlled_scores.shape:
        raise ValueError("E-S0a score effects require aligned score matrices")
    effects = controlled_scores - zero_scores
    effects = effects - effects.mean(dim=-1, keepdim=True)
    targets = F.one_hot(
        gold_indices, num_classes=controlled_scores.shape[1]
    ).float()
    targets = targets - targets.mean(dim=-1, keepdim=True)
    targets = targets / targets.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    binding = effects @ targets.T
    centered = (
        binding
        - binding.mean(dim=0, keepdim=True)
        - binding.mean(dim=1, keepdim=True)
        + binding.mean()
    )
    diagonal = torch.diagonal(centered)
    off_diagonal = centered.masked_fill(
        torch.eye(centered.shape[0], dtype=torch.bool, device=centered.device),
        -torch.inf,
    )
    advantage = diagonal - off_diagonal.max(dim=1).values
    cosine = (effects * targets).sum(dim=-1) / effects.norm(
        dim=-1
    ).clamp_min(1e-12)
    residual = torch.sqrt(torch.clamp(1 - cosine.square(), min=0))
    return {
        "effects": effects,
        "targets": targets,
        "binding": binding,
        "double_centered_binding": centered,
        "diagonal_advantage": advantage,
        "target_cosine": cosine,
        "target_projection_residual": residual,
    }
