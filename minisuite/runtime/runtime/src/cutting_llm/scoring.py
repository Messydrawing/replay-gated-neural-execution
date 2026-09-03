from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def resolve_decoder_and_output_head(model: nn.Module) -> tuple[nn.Module, nn.Module]:
    """Resolve the decoder body and frozen vocabulary projection.

    Scoring only answer positions avoids constructing prompt-position logits,
    while preserving the exact frozen LM-head probabilities used by generation.
    """

    root = model
    get_base_model = getattr(model, "get_base_model", None)
    if callable(get_base_model):
        candidate = get_base_model()
        if isinstance(candidate, nn.Module):
            root = candidate
    for name in ("model", "gpt_neox", "transformer"):
        decoder = getattr(root, name, None)
        if isinstance(decoder, nn.Module):
            break
    else:
        raise ValueError("unable to resolve causal decoder body")
    output_head = root.get_output_embeddings() if hasattr(root, "get_output_embeddings") else None
    if not isinstance(output_head, nn.Module):
        raise ValueError("unable to resolve frozen output embedding head")
    return decoder, output_head


def candidate_sequence_log_scores(
    model: nn.Module,
    batch: dict[str, Tensor],
    *,
    length_normalize: bool = True,
) -> Tensor:
    """Score candidate suffixes with the frozen LM head.

    ``labels`` uses -100 outside candidate tokens. Scores are differentiable
    through hidden interventions but never add a capsule-owned vocabulary head.
    """

    if "labels" not in batch:
        raise ValueError("candidate batch requires labels")
    decoder, output_head = resolve_decoder_and_output_head(model)
    decoder_arguments: dict[str, Any] = {
        key: value for key, value in batch.items() if key in {"input_ids", "attention_mask"}
    }
    outputs = decoder(**decoder_arguments, use_cache=False, return_dict=True)
    hidden = outputs.last_hidden_state[:, :-1, :]
    labels = batch["labels"][:, 1:]
    mask = labels.ne(-100)
    if not bool(mask.any().item()):
        raise ValueError("candidate batch contains no scored tokens")
    selected_hidden = hidden[mask]
    selected_labels = labels[mask]
    token_logits = output_head(selected_hidden).float()
    token_log_probabilities = F.log_softmax(token_logits, dim=-1).gather(
        1, selected_labels.unsqueeze(1)
    ).squeeze(1)
    batch_indices = torch.arange(labels.shape[0], device=labels.device).unsqueeze(1)
    selected_batch_indices = batch_indices.expand_as(labels)[mask]
    scores = torch.zeros(labels.shape[0], device=hidden.device, dtype=torch.float32)
    scores.scatter_add_(0, selected_batch_indices, token_log_probabilities)
    if length_normalize:
        scores = scores / mask.sum(dim=1).clamp_min(1)
    return scores


def shared_prefix_candidate_log_scores(
    model: nn.Module,
    prompt_batch: dict[str, Tensor],
    candidate_ids: Tensor,
    *,
    after_prompt: Callable[[], None] | None = None,
) -> Tensor:
    """Length-normalized candidate scores with exact prompt/prefix KV reuse.

    Candidates must have equal token length. ``after_prompt`` can structurally
    disable an intervention before candidate tokens are processed, enforcing a
    prompt-only compiler objective without detaching the differentiable cache.
    """

    if candidate_ids.ndim != 2 or candidate_ids.shape[0] < 2:
        raise ValueError("candidate_ids must contain at least two token sequences")
    decoder, output_head = resolve_decoder_and_output_head(model)
    prompt_outputs = decoder(
        **prompt_batch,
        use_cache=True,
        return_dict=True,
    )
    if after_prompt is not None:
        after_prompt()
    cache = prompt_outputs.past_key_values
    divergence = None
    for index in range(candidate_ids.shape[1]):
        if not bool(torch.all(candidate_ids[:, index] == candidate_ids[0, index]).item()):
            divergence = index
            break
    if divergence is None:
        raise ValueError("candidate set contains no token-level divergence")
    last_hidden = prompt_outputs.last_hidden_state[:, -1, :]
    if divergence > 0:
        common_prefix = candidate_ids[:1, :divergence]
        full_attention = torch.ones(
            1,
            prompt_batch["input_ids"].shape[1] + divergence,
            dtype=prompt_batch["attention_mask"].dtype,
            device=candidate_ids.device,
        )
        common_outputs = decoder(
            input_ids=common_prefix,
            attention_mask=full_attention,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        cache = common_outputs.past_key_values
        last_hidden = common_outputs.last_hidden_state[:, -1, :]
    branch_logits = output_head(last_hidden).float()
    branch_log_probabilities = F.log_softmax(branch_logits, dim=-1)[
        0, candidate_ids[:, divergence]
    ]
    remaining_targets = candidate_ids[:, divergence + 1 :]
    if remaining_targets.shape[1] == 0:
        return branch_log_probabilities / candidate_ids.shape[1]
    cache.batch_repeat_interleave(candidate_ids.shape[0])
    suffix_inputs = candidate_ids[:, divergence:-1]
    suffix_attention = torch.ones(
        candidate_ids.shape[0],
        prompt_batch["input_ids"].shape[1] + divergence + suffix_inputs.shape[1],
        dtype=prompt_batch["attention_mask"].dtype,
        device=candidate_ids.device,
    )
    suffix_outputs = decoder(
        input_ids=suffix_inputs,
        attention_mask=suffix_attention,
        past_key_values=cache,
        use_cache=False,
        return_dict=True,
    )
    suffix_hidden = suffix_outputs.last_hidden_state
    suffix_logits = output_head(suffix_hidden.reshape(-1, suffix_hidden.shape[-1])).float()
    suffix_log_probabilities = F.log_softmax(suffix_logits, dim=-1).gather(
        1, remaining_targets.reshape(-1, 1)
    ).reshape(candidate_ids.shape[0], -1)
    return (
        branch_log_probabilities + suffix_log_probabilities.sum(dim=1)
    ) / candidate_ids.shape[1]


def split_address_candidate_log_scores(
    model: nn.Module,
    prompt_batch: dict[str, Tensor],
    candidate_ids: Tensor,
    *,
    address_end_token_index: int,
    after_address: Callable[[Tensor, Any], None] | None = None,
) -> Tensor:
    """Score candidates after a causal two-stage prompt prefill.

    The frozen decoder first consumes the prompt through the token completing
    the address.  ``after_address`` may then create a request-scoped carrier
    from the public address representation.  Remaining prompt tokens and answer
    candidates read that carrier without rewriting prefix history, adding cache
    positions, or moving answer positions.

    Calling this function without ``after_address`` provides the matched split-
    prefill base path used by every Latch comparison.
    """

    if candidate_ids.ndim != 2 or candidate_ids.shape[0] < 2:
        raise ValueError("candidate_ids must contain at least two token sequences")
    if set(prompt_batch) < {"input_ids", "attention_mask"}:
        raise ValueError("split prefill requires input_ids and attention_mask")
    prompt_length = int(prompt_batch["input_ids"].shape[1])
    if not 0 <= address_end_token_index < prompt_length:
        raise ValueError("address-end token index lies outside the prompt")
    decoder, output_head = resolve_decoder_and_output_head(model)
    prefix_length = address_end_token_index + 1
    prefix_inputs = {
        "input_ids": prompt_batch["input_ids"][:, :prefix_length],
        "attention_mask": prompt_batch["attention_mask"][:, :prefix_length],
    }
    prefix_outputs = decoder(
        **prefix_inputs,
        use_cache=True,
        return_dict=True,
    )
    cache = prefix_outputs.past_key_values
    last_hidden = prefix_outputs.last_hidden_state[:, -1, :]
    if after_address is not None:
        after_address(last_hidden, cache)
    if prefix_length < prompt_length:
        tail_ids = prompt_batch["input_ids"][:, prefix_length:]
        tail_outputs = decoder(
            input_ids=tail_ids,
            attention_mask=prompt_batch["attention_mask"],
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        cache = tail_outputs.past_key_values
        last_hidden = tail_outputs.last_hidden_state[:, -1, :]

    divergence = None
    for index in range(candidate_ids.shape[1]):
        if not bool(torch.all(candidate_ids[:, index] == candidate_ids[0, index]).item()):
            divergence = index
            break
    if divergence is None:
        raise ValueError("candidate set contains no token-level divergence")
    if divergence > 0:
        common_prefix = candidate_ids[:1, :divergence]
        full_attention = torch.ones(
            1,
            prompt_length + divergence,
            dtype=prompt_batch["attention_mask"].dtype,
            device=candidate_ids.device,
        )
        common_outputs = decoder(
            input_ids=common_prefix,
            attention_mask=full_attention,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        cache = common_outputs.past_key_values
        last_hidden = common_outputs.last_hidden_state[:, -1, :]
    branch_logits = output_head(last_hidden).float()
    branch_log_probabilities = F.log_softmax(branch_logits, dim=-1)[
        0, candidate_ids[:, divergence]
    ]
    remaining_targets = candidate_ids[:, divergence + 1 :]
    if remaining_targets.shape[1] == 0:
        return branch_log_probabilities / candidate_ids.shape[1]
    cache.batch_repeat_interleave(candidate_ids.shape[0])
    suffix_inputs = candidate_ids[:, divergence:-1]
    suffix_attention = torch.ones(
        candidate_ids.shape[0],
        prompt_length + divergence + suffix_inputs.shape[1],
        dtype=prompt_batch["attention_mask"].dtype,
        device=candidate_ids.device,
    )
    suffix_outputs = decoder(
        input_ids=suffix_inputs,
        attention_mask=suffix_attention,
        past_key_values=cache,
        use_cache=False,
        return_dict=True,
    )
    suffix_hidden = suffix_outputs.last_hidden_state
    suffix_logits = output_head(suffix_hidden.reshape(-1, suffix_hidden.shape[-1])).float()
    suffix_log_probabilities = F.log_softmax(suffix_logits, dim=-1).gather(
        1, remaining_targets.reshape(-1, 1)
    ).reshape(candidate_ids.shape[0], -1)
    return (
        branch_log_probabilities + suffix_log_probabilities.sum(dim=1)
    ) / candidate_ids.shape[1]
