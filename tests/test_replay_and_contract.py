from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pytest

from replay_gated_execution.certificate import build_certificate
from replay_gated_execution.physical_abi import PhysicalContract, contract_metrics
from replay_gated_execution.replay import AuthorizationError, evaluate_candidates, select_and_commit


UTILITY_SHA = "1" * 64
PROTOCOL_SHA = "2" * 64


def legal_action(energy: float = 0.01) -> np.ndarray:
    return np.full(9216, np.sqrt(energy / 9216), dtype=np.float32)


class FakeBackend:
    def __init__(self) -> None:
        self.state = {"cache": 0, "rng": 7}
        self.replay_starts: list[tuple[int, str]] = []
        self.commit_actions: list[np.ndarray] = []
        self.fail_replay_after: int | None = None

    def snapshot(self):
        return copy.deepcopy(self.state)

    def restore(self, snapshot) -> None:
        self.state = copy.deepcopy(snapshot)

    def state_sha256(self) -> str:
        raw = json.dumps(self.state, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()

    def replay(self, action: np.ndarray, precision: str) -> np.ndarray:
        self.replay_starts.append((self.state["cache"], precision))
        self.state["cache"] += 1
        scores = np.zeros(64, dtype=np.float64)
        if self.fail_replay_after is None or len(self.replay_starts) <= self.fail_replay_after:
            scores[3] = 1.0 + float(np.linalg.norm(action))
        return scores

    def commit(self, action: np.ndarray):
        assert self.state["cache"] == 0
        self.commit_actions.append(action.copy())
        self.state["cache"] = 1
        return {"committed_norm": float(np.linalg.norm(action))}


def certifier(action, fp32, bf16):
    zero = np.zeros(64, dtype=np.float64)
    return build_certificate(
        action=action,
        scores_fp32=fp32,
        scores_bf16=bf16,
        zero_fp32=zero,
        zero_bf16=zero,
        target=3,
        candidate_permutation_error=0.0,
        exact_zero_pass=True,
    )


def evaluate(backend: FakeBackend):
    return evaluate_candidates(
        backend,
        [legal_action(0.009), legal_action(0.01)],
        certifier,
        utility_identity_sha256=UTILITY_SHA,
        protocol_identity_sha256=PROTOCOL_SHA,
        information_path_pass=True,
        evidence_binding_pass=True,
    )


def commit(backend: FakeBackend, batch):
    return select_and_commit(
        backend,
        batch,
        certifier,
        expected_utility_identity_sha256=UTILITY_SHA,
        expected_protocol_identity_sha256=PROTOCOL_SHA,
    )


def test_physical_contract_fixture() -> None:
    metrics = contract_metrics(legal_action(), PhysicalContract())
    assert metrics["hard_pass"]
    assert abs(metrics["total"] - 0.01) < 1e-7


def test_every_candidate_and_final_replay_start_from_original_state() -> None:
    backend = FakeBackend()
    batch = evaluate(backend)
    assert backend.state["cache"] == 0
    assert backend.replay_starts == [(0, "fp32"), (0, "bf16"), (0, "fp32"), (0, "bf16")]
    backend.state["cache"] = 99
    evidence = commit(backend, batch)
    assert evidence is not None
    assert evidence.selected.ordinal == 0
    assert backend.replay_starts[-2:] == [(0, "fp32"), (0, "bf16")]
    assert len(backend.commit_actions) == 1
    assert np.array_equal(backend.commit_actions[0], evidence.selected.action)
    assert backend.state["cache"] == 1


def test_final_replay_failure_never_commits() -> None:
    backend = FakeBackend()
    batch = evaluate(backend)
    backend.fail_replay_after = 4
    with pytest.raises(AuthorizationError, match="did not recertify"):
        commit(backend, batch)
    assert backend.commit_actions == []
    assert backend.state["cache"] == 0


def test_action_tampering_never_commits() -> None:
    backend = FakeBackend()
    batch = evaluate(backend)
    batch.rows[0].action[0] += 1.0
    with pytest.raises(AuthorizationError, match="changed after certification"):
        commit(backend, batch)
    assert backend.commit_actions == []


@pytest.mark.parametrize("kind", ["utility", "protocol"])
def test_identity_mismatch_never_replays_or_commits(kind: str) -> None:
    backend = FakeBackend()
    batch = evaluate(backend)
    replay_count = len(backend.replay_starts)
    kwargs = {
        "expected_utility_identity_sha256": UTILITY_SHA,
        "expected_protocol_identity_sha256": PROTOCOL_SHA,
    }
    kwargs[f"expected_{kind}_identity_sha256"] = "f" * 64
    with pytest.raises(AuthorizationError, match="identity mismatch"):
        select_and_commit(backend, batch, certifier, **kwargs)
    assert len(backend.replay_starts) == replay_count
    assert backend.commit_actions == []
