from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parent


def sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("output", type=pathlib.Path); a = p.parse_args()
    path = a.output / "result.json"; value = json.loads(path.read_text(encoding="utf-8")); failures = []
    clone = dict(value); observed_semantic = clone.pop("semantic_sha256", None)
    if hashlib.sha256(canonical(clone)).hexdigest() != observed_semantic: failures.append("result_semantic")
    if value["source_manifest_sha256"] != sha(ROOT / "SHA256SUMS.json"): failures.append("source_manifest")
    if value["actions_sha256"] != sha(a.output / "actions.npz"): failures.append("actions")
    payload = {key: value[key] for key in ("model", "action_dimension", "rows", "cross_state_transfer", "summary")}
    if hashlib.sha256(canonical(payload)).hexdigest() != value["scientific_payload_sha256"]: failures.append("scientific_payload")
    rows = value["rows"]; transfer = value["cross_state_transfer"]; summary = value["summary"]
    if len(rows) != 8 or not all(row["certified"] for row in rows): failures.append("native_certificate")
    if len(transfer) != 16: failures.append("transfer_population")
    diagonal = [row for row in transfer if row["source_context_rank"] == row["target_context_rank"]]
    if len(diagonal) != 8 or not all(row["certified"] for row in diagonal): failures.append("diagonal_replay")
    if any(row["physical"]["total"] > .021 or not row["physical"]["pass"] for row in rows): failures.append("physical_contract")
    if summary["candidate_permutation_error_max"] > 1e-6 or summary["unsafe_commit_count"] != 0: failures.append("audit")
    audit = {"schema": "REPLAY_GATED_NEURAL_EXECUTION_PUBLIC_MINISUITE_AUDIT_V1",
             "status": "PASS" if not failures else "FAIL", "failures": failures,
             "result_file_sha256": sha(path), "result_semantic_sha256": observed_semantic,
             "scientific_payload_sha256": value["scientific_payload_sha256"],
             "source_manifest_sha256": value["source_manifest_sha256"]}
    (a.output / "audit.json").write_text(json.dumps(audit, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if failures: raise SystemExit(1)


if __name__ == "__main__": main()
