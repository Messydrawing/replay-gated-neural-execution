"""Bound singleton SmolLM2 runtime for the canonical Z64 semantic registry.

The numerical interface exposes only the frozen registry observations and
Utility-native derivatives.  It has no qualification-plane input and no
example-level answer/candidate metadata.
"""

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.nn import functional as F

from .canonical import canonical_sha256
from .registry_adapter import RegistryExecutionCallbacks
from .semantic import RuntimeTelemetry
from .utility_codec import REGISTRY_MEMBERS, REGISTRY_VERSION


LAYERS = (8, 16, 24)
PHASES = 3
HIDDEN = 960
ACTION_DIMENSION = 8640
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SmolLMContext:
    context_id: str
    prompt: str
    address_prefix: str

    def __post_init__(self) -> None:
        if not self.context_id or not self.prompt or not self.address_prefix:
            raise ValueError("complete public context required")
        if not self.prompt.startswith(self.address_prefix):
            raise ValueError("address prefix must be a literal prompt prefix")


def resolve(model: nn.Module) -> tuple[nn.Module, nn.Module]:
    decoder = getattr(model, "model", None)
    head = getattr(model, "lm_head", None)
    if decoder is None or head is None or not isinstance(getattr(decoder, "layers", None), nn.ModuleList):
        raise ValueError("frozen SmolLM2 decoder/output-head layout mismatch")
    return decoder, head


def token_boundary(tokenizer: Any, prompt: str, address_prefix: str) -> tuple[list[int], int]:
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    boundary = len(address_prefix)
    offsets = [tuple(map(int, row)) for row in encoded["offset_mapping"]]
    if any(left < boundary < right for left, right in offsets):
        raise ValueError("token crosses causal boundary")
    address = [index for index, (_left, right) in enumerate(offsets) if right <= boundary]
    bridge = [index for index, (left, _right) in enumerate(offsets) if left >= boundary]
    if not address or len(bridge) != 1 or bridge[0] != address[-1] + 1:
        raise ValueError("K1 requires exactly one bridge token")
    return list(map(int, encoded["input_ids"])), address[-1]


class PhysicalWriteHooks:
    def __init__(self, model: nn.Module) -> None:
        decoder, _ = resolve(model)
        self.layers = decoder.layers
        self.originals: dict[int, Any] = {}
        self.active_action: Tensor | None = None
        self.phase = 0

    def install(self) -> None:
        if self.originals:
            raise RuntimeError("hooks already installed")
        for layer_index in LAYERS:
            layer = self.layers[layer_index]
            original = layer.forward
            self.originals[layer_index] = original

            def patched(module, hidden_states, *args, _index=layer_index,
                        _original=original, **kwargs):
                if self.active_action is None:
                    return _original(hidden_states, *args, **kwargs)
                if hidden_states.ndim != 3 or hidden_states.shape[1] != 1:
                    raise RuntimeError("physical write requires singleton continuation")
                shaped = self.active_action.reshape(-1, PHASES, len(LAYERS), HIDDEN)
                ratio = shaped[:, self.phase, LAYERS.index(_index)]
                if ratio.shape[0] != hidden_states.shape[0]:
                    raise RuntimeError("singleton action batch mismatch")
                prewrite = hidden_states[:, 0].float()
                scale = torch.sqrt(prewrite.square().sum(dim=-1, keepdim=True) + 1e-12)
                delta = (scale * ratio.float()).to(hidden_states.dtype)
                return _original(hidden_states + delta.unsqueeze(1), *args, **kwargs)

            layer.forward = types.MethodType(patched, layer)

    def activate(self, action: Tensor, phase: int) -> None:
        if not self.originals or self.active_action is not None or not 0 <= phase < PHASES:
            raise RuntimeError("invalid physical hook activation")
        if action.ndim != 2 or action.shape[1] != ACTION_DIMENSION or not torch.isfinite(action).all():
            raise ValueError("finite physical action batch required")
        if action.shape[0] > 1 and not torch.equal(action, action[:1].expand_as(action)):
            raise ValueError("registry branches must share one singleton action")
        self.active_action = action
        self.phase = int(phase)

    def deactivate(self) -> None:
        self.active_action = None
        self.phase = 0

    def uninstall(self) -> None:
        self.deactivate()
        for index, original in self.originals.items():
            self.layers[index].forward = original
        self.originals.clear()


def registry_token_ids(tokenizer: Any, device: torch.device) -> Tensor:
    rows = [tokenizer.encode(member, add_special_tokens=False) for member in REGISTRY_MEMBERS]
    if len(set(map(tuple, rows))) != 64 or len({len(row) for row in rows}) != 1:
        raise RuntimeError("frozen Z64 registry tokenization mismatch")
    if len(rows[0]) != PHASES:
        raise RuntimeError("Z64 registry must occupy the frozen three physical phases")
    return torch.tensor(rows, dtype=torch.long, device=device)


def _score_registry(model: nn.Module, tokenizer: Any, context: SmolLMContext,
                    ids: Tensor, action: Tensor | None,
                    hooks: PhysicalWriteHooks) -> Tensor:
    """Normative registry scorer.  `None` is the true no-hook path."""
    decoder, head = resolve(model)
    prompt_ids, boundary = token_boundary(tokenizer, context.prompt, context.address_prefix)
    device = ids.device
    prefix = torch.tensor([prompt_ids[:boundary + 1]], dtype=torch.long, device=device)
    prefix_out = decoder(input_ids=prefix, attention_mask=torch.ones_like(prefix),
                         use_cache=True, return_dict=True)
    cache = prefix_out.past_key_values
    bridge_id = int(prompt_ids[boundary + 1])
    if action is not None:
        hooks.activate(action, 0)
    try:
        bridge = decoder(
            input_ids=torch.full((1, 1), bridge_id, dtype=torch.long, device=device),
            attention_mask=torch.ones(1, prefix.shape[1] + 1, dtype=torch.long, device=device),
            past_key_values=cache, use_cache=True, return_dict=True,
        )
    finally:
        if action is not None:
            hooks.deactivate()
    values = F.log_softmax(head(bridge.last_hidden_state[:, -1]).float(), dim=-1)[:, ids[:, 0]]
    cache = bridge.past_key_values
    cache.batch_repeat_interleave(64)
    for position in range(ids.shape[1] - 1):
        if action is not None:
            hooks.activate(action.expand(64, -1), position + 1)
        try:
            output = decoder(
                input_ids=ids[:, position:position + 1],
                attention_mask=torch.ones(64, len(prompt_ids) + position + 1,
                                            dtype=torch.long, device=device),
                past_key_values=cache, use_cache=True, return_dict=True,
            )
        finally:
            if action is not None:
                hooks.deactivate()
        cache = output.past_key_values
        token_lp = F.log_softmax(head(output.last_hidden_state[:, -1]).float(), dim=-1)
        values = values + token_lp.gather(1, ids[:, position + 1:position + 2]).reshape(1, 64)
    values = values / float(ids.shape[1])
    return values - values.mean(dim=-1, keepdim=True)


def _score_registry_independent(model: nn.Module, tokenizer: Any, context: SmolLMContext,
                                ids: Tensor, action: Tensor | None,
                                hooks: PhysicalWriteHooks) -> Tensor:
    """Second candidate-blind implementation used only for final replay."""
    decoder, head = resolve(model)
    tokens, endpoint = token_boundary(tokenizer, context.prompt, context.address_prefix)
    device = ids.device
    prefix_ids = torch.as_tensor(tokens[:endpoint + 1], device=device).reshape(1, -1)
    first = decoder(input_ids=prefix_ids, attention_mask=torch.ones_like(prefix_ids),
                    use_cache=True, return_dict=True)
    if action is not None:
        hooks.activate(action, 0)
    try:
        second = decoder(
            input_ids=torch.tensor([[tokens[endpoint + 1]]], dtype=torch.long, device=device),
            attention_mask=torch.ones(1, prefix_ids.shape[1] + 1, dtype=torch.long, device=device),
            past_key_values=first.past_key_values, use_cache=True, return_dict=True,
        )
    finally:
        if action is not None:
            hooks.deactivate()
    result = F.log_softmax(head(second.last_hidden_state[:, -1]).float(), dim=-1)[:, ids[:, 0]]
    branch_cache = second.past_key_values
    branch_cache.batch_repeat_interleave(64)
    for offset in range(1, ids.shape[1]):
        if action is not None:
            hooks.activate(action.expand(64, -1), offset)
        try:
            branch = decoder(
                input_ids=ids[:, offset - 1:offset],
                attention_mask=torch.ones(64, len(tokens) + offset, dtype=torch.long, device=device),
                past_key_values=branch_cache, use_cache=True, return_dict=True,
            )
        finally:
            if action is not None:
                hooks.deactivate()
        branch_cache = branch.past_key_values
        result = result + F.log_softmax(head(branch.last_hidden_state[:, -1]).float(), dim=-1).gather(
            1, ids[:, offset:offset + 1]).reshape(1, 64)
    result = result / float(ids.shape[1])
    return result - result.mean(dim=-1, keepdim=True)


class BoundSmolLMZ64Runtime:
    """Own the two frozen models and expose bound registry callbacks."""

    def __init__(self, *, fp32_model: nn.Module, bf16_model: nn.Module,
                 tokenizer: Any, context: SmolLMContext, vjp_chunk: int = 8) -> None:
        if vjp_chunk <= 0 or 64 % vjp_chunk:
            raise ValueError("VJP chunk must divide 64")
        if any(parameter.requires_grad for model in (fp32_model, bf16_model)
               for parameter in model.parameters()):
            raise ValueError("SmolLM2 parameters must be frozen")
        self.models = {"fp32": fp32_model, "bf16": bf16_model}
        self.tokenizer = tokenizer
        self.context = context
        self.vjp_chunk = int(vjp_chunk)
        self.hooks = {name: PhysicalWriteHooks(model) for name, model in self.models.items()}
        for hook in self.hooks.values():
            hook.install()
        self.ids = {name: registry_token_ids(tokenizer, next(model.parameters()).device)
                    for name, model in self.models.items()}
        self.replay_counts = {"fp32": 0, "bf16": 0, "fp32_vjp": 0}
        self._zero_cache: dict[tuple[str, str], np.ndarray] = {}
        self._exact_zero_cache: dict[str, bool] = {}

    def close(self) -> None:
        for hook in self.hooks.values():
            hook.uninstall()

    def _score(self, precision: str, action: FloatArray, implementation: str) -> np.ndarray:
        value = np.asarray(action, dtype=np.float64).reshape(-1)
        if value.shape != (ACTION_DIMENSION,) or not np.isfinite(value).all():
            raise ValueError("finite physical action required")
        model = self.models[precision]
        tensor = torch.from_numpy(value.astype(np.float32)).to(next(model.parameters()).device).reshape(1, -1)
        scorer: Callable[..., Tensor] = (_score_registry if implementation == "registry"
                                         else _score_registry_independent)
        with torch.no_grad():
            result = scorer(model, self.tokenizer, self.context, self.ids[precision],
                            tensor, self.hooks[precision])
        self.replay_counts[precision] += 1
        return result[0].detach().float().cpu().numpy().astype(np.float64)

    def replay(self, precision: str, action: FloatArray, implementation: str = "registry") -> np.ndarray:
        if precision not in self.models or implementation not in {"registry", "independent"}:
            raise ValueError("unknown bound replay identity")
        return self._score(precision, action, implementation)

    def replay_with_fixed_candidate_permutation(
        self, precision: str, action: FloatArray, implementation: str = "independent"
    ) -> np.ndarray:
        """Replay a fixed bijective candidate order and restore canonical Z64 order."""
        if precision not in self.models or implementation not in {"registry", "independent"}:
            raise ValueError("unknown bound replay identity")
        value = np.asarray(action, dtype=np.float64).reshape(-1)
        if value.shape != (ACTION_DIMENSION,) or not np.isfinite(value).all():
            raise ValueError("finite physical action required")
        model = self.models[precision]
        device = next(model.parameters()).device
        tensor = torch.from_numpy(value.astype(np.float32)).to(device).reshape(1, -1)
        permutation = torch.tensor([(17 * index + 3) % 64 for index in range(64)], device=device)
        scorer: Callable[..., Tensor] = (_score_registry if implementation == "registry"
                                         else _score_registry_independent)
        with torch.no_grad():
            permuted = scorer(model, self.tokenizer, self.context,
                              self.ids[precision][permutation], tensor,
                              self.hooks[precision])[0]
        restored = torch.empty_like(permuted)
        restored[permutation] = permuted
        self.replay_counts[precision] += 1
        return restored.detach().float().cpu().numpy().astype(np.float64)

    def fp32_registry_jacobian(self, action: FloatArray) -> tuple[np.ndarray, np.ndarray]:
        value = np.asarray(action, dtype=np.float64).reshape(-1)
        model = self.models["fp32"]
        tensor = torch.from_numpy(value.astype(np.float32)).to(next(model.parameters()).device).reshape(1, -1)
        tensor = tensor.detach().requires_grad_(True)
        score = _score_registry(model, self.tokenizer, self.context, self.ids["fp32"],
                                tensor, self.hooks["fp32"])
        identity = torch.eye(64, dtype=score.dtype, device=score.device)
        rows = []
        for left in range(0, 64, self.vjp_chunk):
            right = left + self.vjp_chunk
            gradient = torch.autograd.grad(
                score, tensor, grad_outputs=identity[left:right].unsqueeze(1),
                is_grads_batched=True, retain_graph=right < 64,
            )[0][:, 0]
            rows.append(gradient.detach().float().cpu())
        self.replay_counts["fp32_vjp"] += 1
        return (score[0].detach().float().cpu().numpy().astype(np.float64),
                torch.cat(rows).numpy().astype(np.float64))

    def exact_zero_is_bitwise(self, implementation: str = "registry") -> bool:
        if implementation not in {"registry", "independent"}:
            raise ValueError("unknown zero-check implementation")
        if implementation in self._exact_zero_cache:
            return self._exact_zero_cache[implementation]
        scorer = _score_registry if implementation == "registry" else _score_registry_independent
        passed = True
        for precision, model in self.models.items():
            device = next(model.parameters()).device
            zero = torch.zeros(1, ACTION_DIMENSION, dtype=torch.float32, device=device)
            with torch.no_grad():
                absent = scorer(model, self.tokenizer, self.context, self.ids[precision],
                                None, self.hooks[precision])
                explicit = scorer(model, self.tokenizer, self.context, self.ids[precision],
                                  zero, self.hooks[precision])
            passed = passed and torch.equal(absent, explicit)
        self._exact_zero_cache[implementation] = bool(passed)
        return bool(passed)

    def capture_telemetry(self) -> RuntimeTelemetry:
        """Capture candidate-free pre-write bridge states once from FP32 SmolLM2."""
        model = self.models["fp32"]
        decoder, _ = resolve(model)
        ids, boundary = token_boundary(self.tokenizer, self.context.prompt,
                                       self.context.address_prefix)
        device = next(model.parameters()).device
        prefix = torch.tensor([ids[:boundary + 1]], dtype=torch.long, device=device)
        with torch.no_grad():
            prefix_out = decoder(input_ids=prefix, attention_mask=torch.ones_like(prefix),
                                 use_cache=True, return_dict=True)
        captured: dict[int, Tensor] = {}
        handles = []
        for layer_index in LAYERS:
            def pre_hook(_module, args, _index=layer_index):
                state = args[0]
                if state.shape != (1, 1, HIDDEN):
                    raise RuntimeError("telemetry capture must be singleton pre-write")
                captured[_index] = state[0, 0].detach().float().cpu()
            handles.append(decoder.layers[layer_index].register_forward_pre_hook(pre_hook))
        try:
            with torch.no_grad():
                decoder(
                    input_ids=torch.tensor([[ids[boundary + 1]]], dtype=torch.long, device=device),
                    attention_mask=torch.ones(1, prefix.shape[1] + 1, dtype=torch.long, device=device),
                    past_key_values=prefix_out.past_key_values, use_cache=True, return_dict=True,
                )
        finally:
            for handle in handles:
                handle.remove()
        if set(captured) != set(LAYERS):
            raise RuntimeError("candidate-free telemetry capture incomplete")
        vector = torch.cat([captured[index] for index in LAYERS]).numpy().astype(np.float64)
        identity = canonical_sha256({
            "schema": "K1_CANDIDATE_FREE_PREWRITE_TELEMETRY_V1",
            "context_id": self.context.context_id,
            "layers": LAYERS,
            "values_sha256": __import__("hashlib").sha256(vector.astype("<f8").tobytes()).hexdigest(),
        })
        return RuntimeTelemetry(tuple(float(item) for item in vector), identity)

    def callbacks(self, implementation: str = "registry") -> RegistryExecutionCallbacks:
        if implementation not in {"registry", "independent"}:
            raise ValueError("unknown callback implementation")
        return RegistryExecutionCallbacks(
            REGISTRY_VERSION,
            64,
            {precision: (lambda action, p=precision, i=implementation: self.replay(p, action, i))
             for precision in ("fp32", "bf16")},
            self.fp32_registry_jacobian,
            lambda i=implementation: self.exact_zero_is_bitwise(i),
        )
