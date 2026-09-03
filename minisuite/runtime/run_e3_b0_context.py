from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from e3_binding import load_semantic, verify_e3_b0_binding
from e3_common import (
    ACTION_DIMENSION,
    ANSWER_BUDGET,
    BRIDGE_BUDGET,
    HIDDEN_SIZE,
    LAYER_INDICES,
    MAXIMUM_PER_WRITE_RATIO,
    PROJECTED_SEARCH_CAP,
    TOTAL_BUDGET,
    canonical_sha256,
    dump_semantic,
    file_sha256,
    full_behavior_certificate,
    legal_radial_energy_cap,
    physical_metrics,
)


CHART_WEIGHTS = ((1.0, 0.0), (1.0, 1.0), (1.0, 2.0), (1.0, 4.0))
RADIAL_FRACTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)
FALLBACK_STEPS = 97
FALLBACK_LR = 0.004


def select_context(commitment_path: pathlib.Path, pool_path: pathlib.Path, rank: int) -> tuple[dict[str, Any], list[int]]:
    commitment = load_semantic(commitment_path)
    pool = json.loads(pool_path.read_text(encoding="utf-8"))
    if commitment.get("role") != "DESIGN_PUBLIC" or commitment.get("context_count") != 2:
        raise RuntimeError("E3 B0 requires the two-context DesignPublic commitment")
    if not 0 <= rank < 2:
        raise ValueError("E3 B0 context rank must be 0 or 1")
    chosen = commitment["selected_contexts"][rank]
    matches = [row for row in pool["contexts"] if row["context_id"] == chosen["context_id"]]
    if len(matches) != 1 or matches[0]["family"] != chosen["family"]:
        raise RuntimeError("E3 DesignPublic commitment does not match public pool")
    utilities = list(map(int, commitment["utility_ordinals"]))
    if len(utilities) != 32 or len(set(utilities)) != 32 or not all(0 <= value < 64 for value in utilities):
        raise RuntimeError("E3 utility subset is invalid")
    return matches[0], utilities


def target_vector(target: int) -> np.ndarray:
    value = np.full(64, -1.0 / 64.0, dtype=np.float64)
    value[int(target)] += 1.0
    return value / np.linalg.norm(value)


def raw_to_state(action: np.ndarray, template: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    shaped = np.asarray(action, dtype=np.float32).reshape(3, 3, HIDDEN_SIZE)
    result = {}
    for name, tensor in template.items():
        parts = name.split("__")
        phase = 0 if parts[0] == "bridge" else 1 + int(parts[1][1:])
        layer = LAYER_INDICES.index(int(parts[-1][1:]))
        result[name] = torch.from_numpy(shaped[phase, layer]).to(device=tensor.device, dtype=tensor.dtype)
    return result


def state_to_raw(state: dict[str, torch.Tensor]) -> np.ndarray:
    shaped = np.zeros((3, 3, HIDDEN_SIZE), dtype=np.float32)
    for name, tensor in state.items():
        parts = name.split("__")
        phase = 0 if parts[0] == "bridge" else 1 + int(parts[1][1:])
        layer = LAYER_INDICES.index(int(parts[-1][1:]))
        shaped[phase, layer] = tensor.detach().float().cpu().numpy()
    return shaped.reshape(-1)


def torch_metrics(scores: torch.Tensor, zero: torch.Tensor, gold: torch.Tensor) -> dict[str, torch.Tensor]:
    target = scores.gather(1, gold[:, None]).squeeze(1)
    mask = F.one_hot(gold, 64).bool()
    other = scores.masked_fill(mask, -torch.inf).max(dim=-1).values
    effect = scores - zero
    target_effect = effect.gather(1, gold[:, None]).squeeze(1)
    other_effect = effect.masked_fill(mask, -torch.inf).max(dim=-1).values
    ranks = 1 + (scores > target[:, None]).sum(dim=-1)
    return {
        "rank": ranks,
        "margin": target - other,
        "effect_margin": target_effect - other_effect,
        "mrr": 1.0 / ranks.float(),
    }


def criterion(
    metrics32: dict[str, torch.Tensor],
    metricsbf: dict[str, torch.Tensor],
    local: int,
    energy: float,
    step: int,
) -> tuple[float, ...]:
    margin = min(
        metrics32["margin"][local].detach().float().item(),
        metricsbf["margin"][local].detach().float().item(),
    )
    effect = min(
        metrics32["effect_margin"][local].detach().float().item(),
        metricsbf["effect_margin"][local].detach().float().item(),
    )
    mrr = min(
        metrics32["mrr"][local].detach().float().item(),
        metricsbf["mrr"][local].detach().float().item(),
    )
    return (float(margin >= 0.2 and effect >= 0.0), margin, effect, mrr, -float(energy), -float(step))


def project(schedule: Any, contract: Any) -> None:
    schedule.project_(contract=contract, bridge_cap=contract.bridge_budget, answer_cap=contract.answer_budget)
    for local in range(schedule.utility_count):
        energy = float(schedule.energy_summary(local)["maximum_total_path_energy"])
        if energy > PROJECTED_SEARCH_CAP and energy > 0.0:
            factor = math.sqrt(PROJECTED_SEARCH_CAP / energy)
            state = schedule.export_utility(local)
            schedule.import_utility(local, {name: value * factor for name, value in state.items()})


class Machine:
    def __init__(self, model_path: pathlib.Path, runtime_root: pathlib.Path, row: dict[str, Any]) -> None:
        sys.path.insert(0, str(runtime_root / "src"))
        from cutting_llm import causal_reachability_v9 as abi
        from fibertrust.smollm_z64 import BoundSmolLMZ64Runtime, SmolLMContext

        self.abi = abi
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.fp32_model = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True, dtype=torch.float32, attn_implementation="eager"
        ).to("cuda").eval()
        self.bf16_model = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager"
        ).to("cuda").eval()
        for model in (self.fp32_model, self.bf16_model):
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        self.device = next(self.fp32_model.parameters()).device
        self.runtime = BoundSmolLMZ64Runtime(
            fp32_model=self.fp32_model,
            bf16_model=self.bf16_model,
            tokenizer=self.tokenizer,
            context=SmolLMContext(row["context_id"], row["prompt"], row["address_prefix"]),
            vjp_chunk=8,
        )
        self.candidate_ids = abi.candidate_token_ids(
            self.tokenizer, [f" {value:02d}" for value in range(64)], self.device
        )
        prompt_ids, boundary = abi.token_boundary(self.tokenizer, row["prompt"], row["address_prefix"])
        if len(prompt_ids) - boundary - 1 != 1:
            raise RuntimeError("E3 B0 requires exactly one bridge token")
        self.score_row = {
            "prompt": row["prompt"],
            "address_prefix": row["address_prefix"],
            "candidate_answers": [f" {value:02d}" for value in range(64)],
        }
        self.contract = abi.ReachabilityEnergyContract(
            bridge_budget=BRIDGE_BUDGET,
            answer_budget=ANSWER_BUDGET,
            total_budget=TOTAL_BUDGET,
            maximum_per_write_ratio=MAXIMUM_PER_WRITE_RATIO,
            epsilon=1e-12,
        )

    def new_schedule(self, count: int, seed: int):
        return self.abi.CausalRatioActionSchedule(
            utility_count=count,
            hidden_size=HIDDEN_SIZE,
            layer_indices=LAYER_INDICES,
            bridge_token_count=1,
            candidate_ids=self.candidate_ids,
            seed=seed,
            perturbation_scale=0.0,
            candidate_independent_answer=True,
        ).to(self.device)

    def scores(self, schedule: Any, model: Any, hooks: Any, count: int) -> torch.Tensor:
        return self.abi.causal_direct_candidate_scores(
            model=model,
            tokenizer=self.tokenizer,
            schedule=schedule,
            hooks=hooks,
            row=self.score_row,
            utility_indices=list(range(count)),
        )

    def close(self) -> None:
        self.runtime.close()


def analytic_search(machine: Machine, utilities: list[int]) -> tuple[dict[int, np.ndarray], set[int], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    zero = np.zeros(ACTION_DIMENSION, dtype=np.float64)
    zero32, jacobian = machine.runtime.fp32_registry_jacobian(zero)
    zerobf = machine.runtime.replay("bf16", zero)
    _, singular, vh = np.linalg.svd(jacobian, full_matrices=False)
    basis = vh[:63]
    responsebf = []
    for direction in basis:
        cap = legal_radial_energy_cap(direction)
        amplitude = math.sqrt(min(0.0005, cap * 0.25))
        unit = direction / max(float(np.linalg.norm(direction)), 1e-12)
        responsebf.append(
            (machine.runtime.replay("bf16", unit * amplitude) - machine.runtime.replay("bf16", -unit * amplitude))
            / (2.0 * amplitude)
        )
    responsebf_matrix = np.stack(responsebf, axis=1)
    warm: dict[int, np.ndarray] = {}
    passed: set[int] = set()
    for target in utilities:
        best = None
        desired = target_vector(target)
        for w32, wbf in CHART_WEIGHTS:
            blocks, rhs = [], []
            if w32:
                blocks.append(math.sqrt(w32) * (jacobian @ basis.T))
                rhs.append(math.sqrt(w32) * desired)
            if wbf:
                blocks.append(math.sqrt(wbf) * responsebf_matrix)
                rhs.append(math.sqrt(wbf) * desired)
            coefficients = np.linalg.lstsq(np.concatenate(blocks), np.concatenate(rhs), rcond=1e-8)[0]
            direction = coefficients @ basis
            direction /= max(float(np.linalg.norm(direction)), 1e-12)
            cap = legal_radial_energy_cap(direction)
            for fraction in RADIAL_FRACTIONS:
                action = (direction * math.sqrt(cap * fraction)).astype(np.float32)
                score32 = machine.runtime.replay("fp32", action)
                scorebf = machine.runtime.replay("bf16", action)
                cert = full_behavior_certificate(action, score32, scorebf, zero32, zerobf, target)
                violation = max(
                    0.0,
                    0.2 - cert["fp32"]["margin"],
                    0.2 - cert["bf16"]["margin"],
                    -cert["fp32"]["effect_margin"],
                    -cert["bf16"]["effect_margin"],
                )
                key = (float(cert["pass"]), -violation, -cert["physical"]["total"])
                if best is None or key > best[0]:
                    best = (key, action)
        if best is None:
            raise RuntimeError("analytic search produced no candidate")
        warm[target] = best[1]
        if best[0][0] == 1.0:
            passed.add(target)
    diagnostics = {
        "jacobian_shape": list(jacobian.shape),
        "jacobian_effective_rank": int(np.sum(singular > singular.max() * 1e-4)),
        "singular_values": singular.astype(np.float32),
    }
    return warm, passed, zero32, zerobf, basis, diagnostics


def fallback_search(
    machine: Machine,
    utilities: list[int],
    warm: dict[int, np.ndarray],
    zero32: np.ndarray,
    zerobf: np.ndarray,
    trace_handle: Any,
) -> dict[int, np.ndarray]:
    optimized: dict[int, np.ndarray] = {}
    for left in range(0, len(utilities), 2):
        targets = utilities[left : left + 2]
        count = len(targets)
        schedule = machine.new_schedule(count, 53087 + left)
        for local, target in enumerate(targets):
            schedule.import_utility(local, raw_to_state(warm[target], schedule.export_utility(local)))
        project(schedule, machine.contract)
        gold = torch.tensor(targets, dtype=torch.long, device=machine.device)
        zero32_t = torch.from_numpy(zero32[None]).to(machine.device).expand(count, -1)
        zerobf_t = torch.from_numpy(zerobf[None]).to(machine.device).expand(count, -1)
        hooks32 = machine.abi.CausalDirectActionHookManager(machine.fp32_model, schedule)
        hooksbf = machine.abi.CausalDirectActionHookManager(machine.bf16_model, schedule)
        hooks32.install()
        hooksbf.install()
        optimizer = torch.optim.Adam(schedule.parameters(), lr=FALLBACK_LR)
        best_states = [schedule.export_utility(local) for local in range(count)]
        best_keys: list[tuple[float, ...] | None] = [None] * count
        first_steps: list[int | None] = [None] * count
        try:
            for step in range(FALLBACK_STEPS):
                optimizer.zero_grad(set_to_none=True)
                scoresbf = machine.scores(schedule, machine.bf16_model, hooksbf, count)
                metricsbf = torch_metrics(scoresbf, zerobf_t, gold)
                scores32 = machine.scores(schedule, machine.fp32_model, hooks32, count)
                metrics32 = torch_metrics(scores32, zero32_t, gold)
                all_pass = True
                for local, target in enumerate(targets):
                    action = state_to_raw(schedule.export_utility(local))
                    energy = physical_metrics(action)["total"]
                    key = criterion(metrics32, metricsbf, local, energy, step)
                    if best_keys[local] is None or key > best_keys[local]:
                        best_keys[local] = key
                        best_states[local] = schedule.export_utility(local)
                    passed = key[0] == 1.0
                    all_pass = all_pass and passed
                    if passed and first_steps[local] is None:
                        first_steps[local] = step
                if step == 0 or step % 16 == 0 or all_pass or step == FALLBACK_STEPS - 1:
                    trace_handle.write(
                        json.dumps(
                            {
                                "targets": targets,
                                "step": step,
                                "best_keys": best_keys,
                                "first_certified_steps": first_steps,
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    trace_handle.flush()
                if all_pass:
                    break
                lossbf = (
                    F.cross_entropy(scoresbf, gold)
                    + 5.0 * F.softplus(0.25 - metricsbf["margin"]).mean()
                    + 2.0 * F.softplus(0.02 - metricsbf["effect_margin"]).mean()
                )
                loss32 = (
                    F.cross_entropy(scores32, gold)
                    + 5.0 * F.softplus(0.25 - metrics32["margin"]).mean()
                    + 2.0 * F.softplus(0.02 - metrics32["effect_margin"]).mean()
                )
                (0.5 * lossbf).backward()
                (0.5 * loss32).backward()
                optimizer.step()
                project(schedule, machine.contract)
        finally:
            hooks32.uninstall()
            hooksbf.uninstall()
        for local, target in enumerate(targets):
            optimized[target] = state_to_raw(best_states[local])
    return optimized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", type=pathlib.Path, required=True)
    parser.add_argument("--execution-root", type=pathlib.Path, required=True)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--public-pool", type=pathlib.Path, required=True)
    parser.add_argument("--design-commitment", type=pathlib.Path, required=True)
    parser.add_argument("--context-rank", type=int, required=True)
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    args = parser.parse_args()
    verify_e3_b0_binding(
        binding_path=args.binding,
        execution_root=args.execution_root,
        model_root=args.model,
        public_pool=args.public_pool,
        design_commitment=args.design_commitment,
    )
    if args.output_root.exists():
        raise FileExistsError("fresh E3 B0 context output required")
    row, utilities = select_context(args.design_commitment, args.public_pool, args.context_rank)
    args.output_root.mkdir(parents=True)
    started = time.time()
    machine = Machine(args.model, args.execution_root / "runtime", row)
    trace_path = args.output_root / "E3_B0_OPTIMIZATION_TRACE.jsonl"
    try:
        warm, analytic_pass, zero32, zerobf, _basis, diagnostics = analytic_search(machine, utilities)
        zero_action = np.zeros(ACTION_DIMENSION, dtype=np.float64)
        independent_zero32 = machine.runtime.replay("fp32", zero_action, "independent")
        independent_zerobf = machine.runtime.replay("bf16", zero_action, "independent")
        missing = [target for target in utilities if target not in analytic_pass]
        with trace_path.open("w", encoding="utf-8") as trace_handle:
            optimized = fallback_search(machine, missing, warm, zero32, zerobf, trace_handle) if missing else {}
        final_actions = {target: optimized.get(target, warm[target]) for target in utilities}
        rows = []
        for target in utilities:
            action = final_actions[target]
            registry32 = machine.runtime.replay("fp32", action, "registry")
            registrybf = machine.runtime.replay("bf16", action, "registry")
            independent32 = machine.runtime.replay("fp32", action, "independent")
            independentbf = machine.runtime.replay("bf16", action, "independent")
            permuted32 = machine.runtime.replay_with_fixed_candidate_permutation(
                "fp32", action, "independent"
            )
            permutedbf = machine.runtime.replay_with_fixed_candidate_permutation(
                "bf16", action, "independent"
            )
            certificate = full_behavior_certificate(
                action, independent32, independentbf, independent_zero32, independent_zerobf, target
            )
            scorer_error = max(
                float(np.max(np.abs(registry32 - independent32))),
                float(np.max(np.abs(registrybf - independentbf))),
            )
            permutation_error = max(
                float(np.max(np.abs(independent32 - permuted32))),
                float(np.max(np.abs(independentbf - permutedbf))),
            )
            rows.append(
                {
                    "context_rank": args.context_rank,
                    "utility_ordinal": target,
                    "certified": bool(certificate["pass"]),
                    "analytic_certified": target in analytic_pass,
                    "fallback_used": target in optimized,
                    "fp32": certificate["fp32"],
                    "bf16": certificate["bf16"],
                    "physical": certificate["physical"],
                    "independent_scorer_error": scorer_error,
                    "candidate_permutation_error": permutation_error,
                }
            )
        exact_zero = bool(
            machine.runtime.exact_zero_is_bitwise("registry")
            and machine.runtime.exact_zero_is_bitwise("independent")
        )
    finally:
        machine.close()

    actions = torch.stack([torch.from_numpy(final_actions[target].astype(np.float32)) for target in utilities])
    sidecar_path = args.output_root / "E3_B0_ACTIONS.pt"
    torch.save(
        {
            "schema": "NCO_EFA2_PAPER1_TMLR_E3_B0_ACTIONS_V1",
            "context_rank": args.context_rank,
            "family": row["family"],
            "utility_ordinals": utilities,
            "actions": actions,
            "zero_scores_fp32": torch.from_numpy(zero32.astype(np.float32)),
            "zero_scores_bf16": torch.from_numpy(zerobf.astype(np.float32)),
            "independent_zero_scores_fp32": torch.from_numpy(independent_zero32.astype(np.float32)),
            "independent_zero_scores_bf16": torch.from_numpy(independent_zerobf.astype(np.float32)),
            "singular_values": torch.from_numpy(diagnostics["singular_values"]),
            "model_training_performed": False,
            "backbone_parameters_updated": False,
            "qwen_action_or_checkpoint_reuse": False,
            "pilot_or_fresh_read": False,
            "unseen_utility_read": False,
        },
        sidecar_path,
    )
    certified = sum(item["certified"] for item in rows)
    energies = np.asarray([item["physical"]["total"] for item in rows], dtype=np.float64)
    result = {
        "schema": "NCO_EFA2_PAPER1_TMLR_E3_B0_CONTEXT_RESULT_V1",
        "status": "E3_B0_CONTEXT_COMPLETE",
        "context_rank": args.context_rank,
        "family": row["family"],
        "utility_ordinals": utilities,
        "cell_count": len(rows),
        "certified_count": certified,
        "analytic_certified_count": len(analytic_pass),
        "fallback_target_count": len(missing),
        "fallback_recovered_count": sum(item["certified"] and item["fallback_used"] for item in rows),
        "jacobian_effective_rank": diagnostics["jacobian_effective_rank"],
        "energy_p99": float(np.quantile(energies, 0.99)),
        "energy_max": float(energies.max()),
        "exact_zero_pass": exact_zero,
        "maximum_independent_scorer_error": max(item["independent_scorer_error"] for item in rows),
        "maximum_candidate_permutation_error": max(item["candidate_permutation_error"] for item in rows),
        "rows": rows,
        "actions_file_sha256": file_sha256(sidecar_path),
        "trace_file_sha256": file_sha256(trace_path),
        "elapsed_seconds": time.time() - started,
        "model_training_performed": False,
        "backbone_parameters_updated": False,
        "qwen_action_or_checkpoint_reuse": False,
        "pilot_or_fresh_read": False,
        "unseen_utility_read": False,
    }
    dump_semantic(args.output_root / "E3_B0_RESULT.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "context_rank": args.context_rank,
                "family": row["family"],
                "certified": certified,
                "cells": len(rows),
                "analytic": len(analytic_pass),
                "fallback_recovered": result["fallback_recovered_count"],
                "elapsed_seconds": result["elapsed_seconds"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
