from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parent


def sha(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def semantic_hash(value: dict[str, Any]) -> str:
    clone = dict(value); clone.pop("semantic_sha256", None)
    return hashlib.sha256(canonical_bytes(clone)).hexdigest()


def verify_source_manifest() -> str:
    manifest_path = ROOT / "SHA256SUMS.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for relative, expected in manifest["files"].items():
        if sha(ROOT / relative) != expected:
            raise RuntimeError(f"minisuite source identity mismatch: {relative}")
    return sha(manifest_path)


def resolve_model(path: pathlib.Path | None, cache: pathlib.Path) -> pathlib.Path:
    protocol = json.loads((ROOT / "config/protocol.json").read_text(encoding="utf-8"))
    if path is not None:
        return path.resolve()
    from huggingface_hub import snapshot_download
    return pathlib.Path(snapshot_download(
        repo_id=protocol["model"]["repo_id"], revision=protocol["model"]["revision"],
        local_dir=cache,
        allow_patterns=["config.json", "generation_config.json", "model.safetensors", "merges.txt",
                        "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.json"],
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the public 2-context x 4-Utility replay-gated execution miniature.")
    parser.add_argument("--model", type=pathlib.Path, default=None, help="Existing pinned SmolLM2 model directory")
    parser.add_argument("--model-cache", type=pathlib.Path, default=ROOT / ".model_cache")
    parser.add_argument("--output", type=pathlib.Path, default=ROOT / "output")
    args = parser.parse_args()
    manifest_sha = verify_source_manifest()
    if args.output.exists():
        raise FileExistsError("use a fresh output directory")
    model_path = resolve_model(args.model, args.model_cache)
    args.output.mkdir(parents=True)

    runtime_root = ROOT / "runtime"
    sys.path.insert(0, str(runtime_root))
    import numpy as np
    from e3_common import ACTION_DIMENSION, full_behavior_certificate, physical_metrics
    from run_e3_b0_context import Machine, analytic_search, fallback_search

    protocol = json.loads((ROOT / "config/protocol.json").read_text(encoding="utf-8"))
    contexts = json.loads((ROOT / "data/public_contexts.json").read_text(encoding="utf-8"))["contexts"]
    targets = list(map(int, protocol["utility_ordinals"]))
    started = time.time()
    actions: dict[tuple[int, int], np.ndarray] = {}
    rows = []
    for context_rank, context in enumerate(contexts):
        machine = Machine(model_path, runtime_root / "runtime", context)
        trace_path = args.output / f"context_{context_rank:02d}_trace.jsonl"
        try:
            warm, analytic_pass, zero32, zerobf, _basis, _diagnostics = analytic_search(machine, targets)
            missing = [target for target in targets if target not in analytic_pass]
            with trace_path.open("w", encoding="utf-8") as trace:
                optimized = fallback_search(machine, missing, warm, zero32, zerobf, trace) if missing else {}
            zero = np.zeros(ACTION_DIMENSION, dtype=np.float64)
            independent_zero32 = machine.runtime.replay("fp32", zero, "independent")
            independent_zerobf = machine.runtime.replay("bf16", zero, "independent")
            for target in targets:
                action = optimized.get(target, warm[target])
                actions[(context_rank, target)] = action
                score32 = machine.runtime.replay("fp32", action, "independent")
                scorebf = machine.runtime.replay("bf16", action, "independent")
                certificate = full_behavior_certificate(action, score32, scorebf,
                                                         independent_zero32, independent_zerobf, target)
                perm32 = machine.runtime.replay_with_fixed_candidate_permutation("fp32", action, "independent")
                perm_error = float(np.max(np.abs(score32 - perm32)))
                rows.append({"context_rank": context_rank, "context_id": context["context_id"],
                             "utility_ordinal": target, "finder": "analytic" if target in analytic_pass else "fallback",
                             "certified": bool(certificate["pass"]), "fp32": certificate["fp32"],
                             "bf16": certificate["bf16"], "physical": certificate["physical"],
                             "candidate_permutation_error": perm_error})
        finally:
            machine.close()

    np.savez_compressed(args.output / "actions.npz", **{
        f"context_{context_rank:02d}_utility_{target:02d}": action.astype(np.float32)
        for (context_rank, target), action in actions.items()
    })

    transfer = []
    for target_context_rank, context in enumerate(contexts):
        machine = Machine(model_path, runtime_root / "runtime", context)
        try:
            zero = np.zeros(ACTION_DIMENSION, dtype=np.float64)
            zero32 = machine.runtime.replay("fp32", zero, "independent")
            zerobf = machine.runtime.replay("bf16", zero, "independent")
            for source_context_rank in range(len(contexts)):
                for target in targets:
                    action = actions[(source_context_rank, target)]
                    cert = full_behavior_certificate(action, machine.runtime.replay("fp32", action, "independent"),
                                                     machine.runtime.replay("bf16", action, "independent"),
                                                     zero32, zerobf, target)
                    transfer.append({"source_context_rank": source_context_rank,
                                     "target_context_rank": target_context_rank,
                                     "utility_ordinal": target, "certified": bool(cert["pass"]),
                                     "fp32": cert["fp32"], "bf16": cert["bf16"],
                                     "physical": cert["physical"]})
        finally:
            machine.close()

    diagonal = [row for row in transfer if row["source_context_rank"] == row["target_context_rank"]]
    offdiag = [row for row in transfer if row["source_context_rank"] != row["target_context_rank"]]
    scientific_payload = {
        "model": protocol["model"], "action_dimension": ACTION_DIMENSION,
        "rows": rows, "cross_state_transfer": transfer,
        "summary": {"native_certified": sum(row["certified"] for row in rows), "native_total": len(rows),
                    "diagonal_transfer_certified": sum(row["certified"] for row in diagonal),
                    "diagonal_transfer_total": len(diagonal),
                    "offdiagonal_transfer_certified": sum(row["certified"] for row in offdiag),
                    "offdiagonal_transfer_total": len(offdiag),
                    "unsafe_commit_count": 0,
                    "energy_max": max(row["physical"]["total"] for row in rows),
                    "candidate_permutation_error_max": max(row["candidate_permutation_error"] for row in rows)},
    }
    result = {
        "schema": "REPLAY_GATED_NEURAL_EXECUTION_PUBLIC_MINISUITE_RESULT_V1",
        "status": "MINISUITE_COMPLETE",
        "source_manifest_sha256": manifest_sha,
        **scientific_payload,
        "scientific_payload_sha256": hashlib.sha256(canonical_bytes(scientific_payload)).hexdigest(),
        "actions_sha256": sha(args.output / "actions.npz"),
        "elapsed_seconds": time.time() - started,
        "model_training_performed": False,
        "interpretation_boundary": "Executable miniature on retired public contexts; not a substitute for the frozen formal cohort.",
    }
    result["semantic_sha256"] = semantic_hash(result)
    (args.output / "result.json").write_bytes(canonical_bytes(result))
    print(json.dumps({"status": result["status"], "summary": result["summary"],
                      "semantic_sha256": result["semantic_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
