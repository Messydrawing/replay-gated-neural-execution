from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "frozen" / "aggregate_results"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("semantic_sha256", None)
    encoded = (json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n").encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load(name: str) -> dict[str, Any]:
    path = RESULTS / name
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = value.get("semantic_sha256")
    require(expected is None or canonical_sha256(value) == expected, f"semantic hash mismatch: {name}")
    return value


def verify_manifest() -> None:
    seen: set[str] = set()
    for line in (ROOT / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        require(relative not in seen and len(digest) == 64, f"invalid manifest entry: {relative}")
        seen.add(relative)
        path = (ROOT / relative).resolve()
        require(ROOT.resolve() in path.parents and path.is_file(), f"unsafe or missing manifest path: {relative}")
        require(file_sha256(path) == digest, f"file hash mismatch: {relative}")
    excluded_parts = {".git", "__pycache__", ".pytest_cache", ".venv", "build"}
    actual = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file()
        and path.name != "SHA256SUMS.txt"
        and path.suffix != ".pyc"
        and not excluded_parts.intersection(path.relative_to(ROOT).parts)
    }
    require(seen == actual, f"manifest closure mismatch: missing={sorted(actual-seen)}, extra={sorted(seen-actual)}")


def verify_source_provenance() -> None:
    provenance = json.loads((ROOT / "frozen" / "manifests" / "source_provenance.json").read_text(encoding="utf-8"))
    source = ROOT / "reference" / "replay_gated_execution"
    for name, record in provenance["modules"].items():
        path = source / name
        require(path.is_file(), f"missing provenance module: {name}")
        if record["public_copy_exact"]:
            require(file_sha256(path) == record["frozen_sha256"], f"exact-copy provenance mismatch: {name}")
    for name in provenance["excluded_historical_modules"]:
        require(not (source / name).exists(), f"obsolete module is still present: {name}")
    for name in provenance["reference_reimplementations"]:
        require((source / name).is_file(), f"missing reference implementation: {name}")

    frozen_identity = json.loads(
        (ROOT / "frozen" / "manifests" / "frozen_source_identity.json").read_text(encoding="utf-8")
    )
    require(frozen_identity["formal_runner_imports_cascade_core"] is False, "formal runner/core audit changed")
    require(frozen_identity["formal_runner_uses_row_certificate_pass"] is True, "formal admission audit missing")
    require(frozen_identity["row_certificate_pass_requires_exact_zero"] is True, "exact-zero audit missing")
    require(
        frozen_identity["final_selected_action_dual_precision_replay_present"] is True,
        "final replay audit missing",
    )


def main() -> None:
    verify_manifest()
    verify_source_provenance()
    reference = load("reference_stability_closeout.json")
    pilot = load("cascade_pilot_adjudication.json")
    pilot_closeout = load("cascade_pilot_closeout.json")
    fresh = load("frozen_fresh_adjudication.json")
    fresh_closeout = load("frozen_fresh_closeout.json")

    require(reference["status"] == "REFERENCE_STABILITY_VALID_CLOSEOUT", "bad Reference Stability status")
    require(reference["semantic_reference"]["cell_count"] == 767, "bad semantic-reference count")
    require(reference["semantic_reference"]["failed_cells"] == [], "semantic-reference failures present")
    require(reference["rediscovery"]["rediscovered_count"] == 66, "bad rediscovery count")
    require(reference["rediscovery"]["repeat_count"] == 69, "bad repeat count")

    for result, status in ((pilot, "CASCADE_PASS"), (fresh, "FROZEN_FRESH_PASS")):
        require(result["status"] == status, f"bad result status: {status}")
        require(result["population_cells"] == 256, "bad population count")
        require(result["cascade_certified_count"] == 256, "bad cascade coverage")
        require(result["full_baseline_certified_count"] == 256, "bad full coverage")
        require(result["unsafe_commits"] == [], "observed authorization bypass")
        require(result["hard_violation_count"] == 0, "hard violation present")
        require(result["information_path_violation_count"] == 0, "information-path violation present")
        require(result["nonfinite_count"] == 0, "non-finite result present")
        require(result["committed_energy_p99"] <= 0.018, "P99 energy gate failed")
        require(result["committed_energy_max"] <= 0.021, "hard energy gate failed")
        require(result["model_training_performed"] is False, "model training was performed")
        require(result["qwen_parameters_updated"] is False, "Qwen parameters were updated")

    require(pilot_closeout["status"] == "CASCADE_VALID_CLOSEOUT", "bad Pilot closeout")
    require(fresh_closeout["status"] == "FROZEN_FRESH_VALID_CLOSEOUT", "bad Fresh closeout")
    require(pilot_closeout["result_semantic_sha256"] == pilot["semantic_sha256"], "Pilot closeout mismatch")
    require(fresh_closeout["result_semantic_sha256"] == fresh["semantic_sha256"], "Fresh closeout mismatch")
    print("release verification: PASS")


if __name__ == "__main__":
    main()
