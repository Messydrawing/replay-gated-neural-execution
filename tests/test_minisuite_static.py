from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MINI = ROOT / "minisuite"


def test_minisuite_population_and_claim_boundary() -> None:
    data = json.loads((MINI / "data/public_contexts.json").read_text(encoding="utf-8"))
    protocol = json.loads((MINI / "config/protocol.json").read_text(encoding="utf-8"))
    assert data["population_role"] == "RETIRED_DESIGN_PUBLIC"
    assert len(data["contexts"]) == 2
    assert len({row["family"] for row in data["contexts"]}) == 2
    assert protocol["utility_ordinals"] == [0, 21, 42, 60]
    assert protocol["scientific_status"].startswith("PUBLIC_EXECUTABLE_MINIATURE")


def test_minisuite_contract_is_frozen() -> None:
    protocol = json.loads((MINI / "config/protocol.json").read_text(encoding="utf-8"))
    cert = protocol["certificate"]
    assert cert["robust_margin_minimum"] == 0.2
    assert cert["population_energy_p99_maximum"] == 0.018
    assert cert["hard_energy_maximum"] == 0.021
    assert cert["precisions"] == ["fp32", "bf16"]
    assert protocol["model_training"] is False


def test_minisuite_manifest_closes_sources() -> None:
    manifest = json.loads((MINI / "SHA256SUMS.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["files"].items():
        assert hashlib.sha256((MINI / relative).read_bytes()).hexdigest() == expected
