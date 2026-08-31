from __future__ import annotations

import numpy as np

from replay_gated_execution.semantic import RuntimeSemanticObservation, compile_cell_constraints
from replay_gated_execution.utility_codec import build_margin_utility, utility_from_json, utility_to_json


def test_z64_codec_is_closed_and_label_free() -> None:
    utility = build_margin_utility(5, margin=0.2)
    encoded = utility_to_json(utility)
    assert utility_from_json(encoded).semantic_sha256 == utility.semantic_sha256
    assert "capability" not in str(encoded).lower()
    assert "context" not in str(encoded).lower()


def test_margin_constraints_compile_from_observations() -> None:
    utility = build_margin_utility(5, margin=0.2)
    values = np.zeros(64)
    values[5] = 0.3
    observation = RuntimeSemanticObservation("Z64-sequence-score-v1", "fp32", values)
    zero = RuntimeSemanticObservation("Z64-sequence-score-v1", "fp32", np.zeros(64))
    constraints = compile_cell_constraints(utility, 0, observation, zero)
    assert constraints.values.shape == (63,)
    assert np.min(constraints.values) >= 0.1 - 1e-12
