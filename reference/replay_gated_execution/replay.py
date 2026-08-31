from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence

import numpy as np

from .authorization import Authorization, ItemCertificate, RunAudit, action_sha256


class TransactionalBackend(Protocol):
    """Stateful backend required by the replay and commit authority."""

    def snapshot(self) -> Any: ...

    def restore(self, snapshot: Any) -> None: ...

    def state_sha256(self) -> str: ...

    def replay(self, action: np.ndarray, precision: str) -> np.ndarray: ...

    def commit(self, action: np.ndarray) -> Any: ...


class AuthorizationError(RuntimeError):
    """Raised before commit whenever replay authorization is incomplete."""


@dataclass(frozen=True)
class CandidateReplay:
    ordinal: int
    action: np.ndarray
    scores_fp32: np.ndarray
    scores_bf16: np.ndarray
    authorization: Authorization


@dataclass(frozen=True)
class ReplayBatch:
    original_snapshot: Any
    original_state_sha256: str
    utility_identity_sha256: str
    protocol_identity_sha256: str
    rows: tuple[CandidateReplay, ...]


@dataclass(frozen=True)
class CommitEvidence:
    selected: CandidateReplay
    final_authorization: Authorization
    committed: Any


def _require_original_state(
    backend: TransactionalBackend,
    original_snapshot: Any,
    expected_sha256: str,
) -> bool:
    backend.restore(original_snapshot)
    return backend.state_sha256() == expected_sha256


def isolated_dual_precision_replay(
    backend: TransactionalBackend,
    action: np.ndarray,
    original_snapshot: Any,
    original_state_sha256: str,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Evaluate one counterfactual action without retaining candidate state."""
    restored = False
    try:
        if not _require_original_state(backend, original_snapshot, original_state_sha256):
            raise AuthorizationError("failed to restore original state before FP32 replay")
        fp32 = np.asarray(backend.replay(action, "fp32"), dtype=np.float64)
        if not _require_original_state(backend, original_snapshot, original_state_sha256):
            raise AuthorizationError("failed to restore original state before BF16 replay")
        bf16 = np.asarray(backend.replay(action, "bf16"), dtype=np.float64)
        restored = _require_original_state(backend, original_snapshot, original_state_sha256)
        return fp32, bf16, restored
    finally:
        backend.restore(original_snapshot)


def evaluate_candidates(
    backend: TransactionalBackend,
    candidates: Sequence[np.ndarray],
    certifier: Callable[[np.ndarray, np.ndarray, np.ndarray], ItemCertificate],
    *,
    utility_identity_sha256: str,
    protocol_identity_sha256: str,
    information_path_pass: bool,
    evidence_binding_pass: bool,
) -> ReplayBatch:
    """Replay candidates transactionally and bind item and run evidence."""
    original = backend.snapshot()
    original_sha256 = backend.state_sha256()
    rows: list[CandidateReplay] = []
    try:
        for ordinal, candidate in enumerate(candidates):
            action = np.asarray(candidate, dtype=np.float32).reshape(-1).copy()
            fp32, bf16, restored = isolated_dual_precision_replay(
                backend, action, original, original_sha256
            )
            audit = RunAudit(
                snapshot_restore_pass=restored,
                no_state_leakage=backend.state_sha256() == original_sha256,
                information_path_pass=bool(information_path_pass),
                evidence_binding_pass=bool(evidence_binding_pass),
                action_sha256=action_sha256(action),
                utility_identity_sha256=utility_identity_sha256,
                protocol_identity_sha256=protocol_identity_sha256,
            )
            rows.append(CandidateReplay(
                ordinal=ordinal,
                action=action,
                scores_fp32=fp32,
                scores_bf16=bf16,
                authorization=Authorization(certifier(action, fp32, bf16), audit),
            ))
        return ReplayBatch(
            original_snapshot=original,
            original_state_sha256=original_sha256,
            utility_identity_sha256=utility_identity_sha256,
            protocol_identity_sha256=protocol_identity_sha256,
            rows=tuple(rows),
        )
    finally:
        backend.restore(original)


def select_and_commit(
    backend: TransactionalBackend,
    batch: ReplayBatch,
    certifier: Callable[[np.ndarray, np.ndarray, np.ndarray], ItemCertificate],
    *,
    expected_utility_identity_sha256: str,
    expected_protocol_identity_sha256: str,
) -> CommitEvidence | None:
    """Select, replay the exact action again, recertify, then commit.

    Every rejection happens before ``backend.commit``. The final replay starts
    from the snapshot captured before candidate evaluation; a caller cannot
    substitute the Utility, protocol, or action after certification.
    """
    if batch.utility_identity_sha256 != expected_utility_identity_sha256:
        raise AuthorizationError("Utility identity mismatch")
    if batch.protocol_identity_sha256 != expected_protocol_identity_sha256:
        raise AuthorizationError("protocol identity mismatch")
    if not _require_original_state(
        backend, batch.original_snapshot, batch.original_state_sha256
    ):
        raise AuthorizationError("original state is not restorable")

    authorized = [row for row in batch.rows if row.authorization.permits_commit()]
    if not authorized:
        return None
    selected = min(authorized, key=lambda row: (
        row.authorization.item_certificate.total_energy,
        -min(
            row.authorization.item_certificate.fp32_margin,
            row.authorization.item_certificate.bf16_margin,
        ),
        row.ordinal,
    ))

    exact_action = np.asarray(selected.action, dtype=np.float32).reshape(-1).copy()
    expected_action_sha256 = selected.authorization.run_audit.action_sha256
    if action_sha256(exact_action) != expected_action_sha256:
        raise AuthorizationError("selected action changed after certification")
    if selected.authorization.run_audit.utility_identity_sha256 != expected_utility_identity_sha256:
        raise AuthorizationError("selected authorization has wrong Utility identity")
    if selected.authorization.run_audit.protocol_identity_sha256 != expected_protocol_identity_sha256:
        raise AuthorizationError("selected authorization has wrong protocol identity")

    fp32, bf16, restored = isolated_dual_precision_replay(
        backend,
        exact_action,
        batch.original_snapshot,
        batch.original_state_sha256,
    )
    final_audit = RunAudit(
        snapshot_restore_pass=restored,
        no_state_leakage=backend.state_sha256() == batch.original_state_sha256,
        information_path_pass=selected.authorization.run_audit.information_path_pass,
        evidence_binding_pass=selected.authorization.run_audit.evidence_binding_pass,
        action_sha256=action_sha256(exact_action),
        utility_identity_sha256=expected_utility_identity_sha256,
        protocol_identity_sha256=expected_protocol_identity_sha256,
    )
    final_authorization = Authorization(
        certifier(exact_action, fp32, bf16),
        final_audit,
    )
    if final_audit.action_sha256 != expected_action_sha256:
        raise AuthorizationError("final replay action hash mismatch")
    if not final_authorization.permits_commit():
        raise AuthorizationError("final same-action replay did not recertify")
    if not _require_original_state(
        backend, batch.original_snapshot, batch.original_state_sha256
    ):
        raise AuthorizationError("final commit did not start from original state")

    committed = backend.commit(exact_action.copy())
    return CommitEvidence(selected, final_authorization, committed)
