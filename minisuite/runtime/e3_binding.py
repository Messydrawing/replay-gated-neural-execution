from __future__ import annotations

import json
import pathlib
from typing import Any

from e3_common import canonical_sha256, file_sha256


MINIMUM_MODEL_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def load_semantic(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("semantic_sha256") != canonical_sha256(value):
        raise RuntimeError(f"semantic SHA-256 mismatch: {path}")
    return value


def verify_e3_b0_binding(
    *,
    binding_path: pathlib.Path,
    execution_root: pathlib.Path,
    model_root: pathlib.Path,
    public_pool: pathlib.Path,
    design_commitment: pathlib.Path,
) -> dict[str, Any]:
    binding = load_semantic(binding_path)
    if binding.get("schema") != "NCO_EFA2_PAPER1_TMLR_E3_B0_BINDING_V1":
        raise RuntimeError("unknown E3 B0 binding schema")
    if binding.get("stage") != "B0_DESIGN_PUBLIC_REACHABILITY":
        raise RuntimeError("E3 B0 binding stage mismatch")
    if binding.get("execution_authorized") is not True:
        raise RuntimeError("E3 B0 execution is not authorized")
    if binding.get("pilot_execution_authorized") is not False or binding.get("fresh_execution_authorized") is not False:
        raise RuntimeError("E3 B0 binding illegally authorizes sealed stages")
    for relative, expected in binding["files"].items():
        path = execution_root / relative
        if not path.is_file() or file_sha256(path) != expected:
            raise RuntimeError(f"E3 bound source mismatch: {relative}")
    if file_sha256(public_pool) != binding["public_pool_file_sha256"]:
        raise RuntimeError("E3 public pool identity mismatch")
    if file_sha256(design_commitment) != binding["design_commitment_file_sha256"]:
        raise RuntimeError("E3 DesignPublic commitment identity mismatch")
    design = load_semantic(design_commitment)
    if design.get("role") != "DESIGN_PUBLIC" or design.get("scientific_evidence_eligible") is not False:
        raise RuntimeError("E3 B0 received a non-DesignPublic population")
    protocol_path = execution_root / "E3_SECOND_BACKBONE_PROTOCOL.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if file_sha256(protocol_path) != binding["protocol_file_sha256"]:
        raise RuntimeError("E3 protocol file identity mismatch")
    if canonical_sha256(protocol) != binding["protocol_semantic_sha256"]:
        raise RuntimeError("E3 protocol semantic identity mismatch")
    contract_path = execution_root / "E3_PHYSICAL_CONTRACT.json"
    if file_sha256(contract_path) != binding["physical_contract_file_sha256"]:
        raise RuntimeError("E3 physical contract identity mismatch")
    contract = load_semantic(contract_path)
    if contract.get("execution_authorized") is not False:
        raise RuntimeError("E3 physical contract must remain independently execution-blocked")
    environment = load_semantic(execution_root / "E3_TARGET_ENVIRONMENT.json")
    cleanroom = load_semantic(execution_root / "E3_B0_CLEANROOM.json")
    authorization = load_semantic(execution_root / "E3_B0_USER_AUTHORIZATION.json")
    if file_sha256(execution_root / "E3_TARGET_ENVIRONMENT.json") != binding["target_environment_file_sha256"]:
        raise RuntimeError("E3 target environment identity mismatch")
    if file_sha256(execution_root / "E3_B0_CLEANROOM.json") != binding["cleanroom_result_file_sha256"]:
        raise RuntimeError("E3 clean-room identity mismatch")
    if file_sha256(execution_root / "E3_B0_USER_AUTHORIZATION.json") != binding["user_authorization_file_sha256"]:
        raise RuntimeError("E3 user authorization identity mismatch")
    if cleanroom.get("status") != "PASS" or cleanroom.get("failures") != []:
        raise RuntimeError("E3 clean-room did not pass")
    if authorization.get("B0_execution_authorized") is not True:
        raise RuntimeError("E3 B0 lacks explicit user authorization")
    if authorization.get("Pilot_execution_authorized") is not False or authorization.get("FrozenFresh_execution_authorized") is not False:
        raise RuntimeError("E3 user authorization illegally opens a sealed stage")
    actual_sources = {
        str(path.relative_to(execution_root)).replace("\\", "/"): file_sha256(path)
        for path in sorted(execution_root.rglob("*.py"))
        if "__pycache__" not in path.parts
    }
    if environment.get("source_files") != actual_sources:
        raise RuntimeError("E3 live source closure differs from target environment")
    for relative, expected in binding["model_files"].items():
        path = model_root / relative
        if not path.is_file() or file_sha256(path) != expected:
            raise RuntimeError(f"E3 model identity mismatch: {relative}")
    if environment.get("model_files") != binding["model_files"]:
        raise RuntimeError("E3 environment model closure mismatch")
    return binding
