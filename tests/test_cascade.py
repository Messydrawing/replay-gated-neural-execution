from __future__ import annotations

import numpy as np

from replay_gated_execution.authorization import (
    Authorization,
    ItemCertificate,
    RunAudit,
    action_sha256,
)
from replay_gated_execution.cascade import CascadeStatus, TierAttempt, run_certificate_gated_cascade


IDENTITY = "a" * 64


def authorization(*, qualified: bool = False, finite: bool = True, audit: bool = True) -> Authorization:
    action = np.zeros(9216, dtype=np.float32)
    item = ItemCertificate(
        finite=finite,
        physical_contract_pass=True,
        fp32_top1=qualified,
        bf16_top1=qualified,
        fp32_effect_peak=qualified,
        bf16_effect_peak=qualified,
        fp32_margin=0.25 if qualified else -0.1,
        bf16_margin=0.25 if qualified else -0.1,
        total_energy=0.017,
        candidate_permutation_error=0.0,
        exact_zero_pass=True,
    )
    run = RunAudit(
        snapshot_restore_pass=audit,
        no_state_leakage=audit,
        information_path_pass=audit,
        evidence_binding_pass=audit,
        action_sha256=action_sha256(action),
        utility_identity_sha256=IDENTITY,
        protocol_identity_sha256=IDENTITY,
    )
    return Authorization(item, run)


def attempt(name: str, auth: Authorization) -> TierAttempt:
    return TierAttempt(name, auth, np.zeros(9216), tuple(), 3.0, 2.0, 1, "fixture", {})


def test_finite_miss_escalates_and_costs_are_charged() -> None:
    calls: list[str] = []

    def tier1(_):
        calls.append("tier1")
        return attempt("tier1", authorization())

    def tier2(_):
        calls.append("tier2")
        return attempt("tier2", authorization(qualified=True))

    result = run_certificate_gated_cascade(object(), [tier1, tier2])
    assert result.status is CascadeStatus.AUTHORIZED_FOR_FINAL_REPLAY
    assert result.selected_tier == "tier2"
    assert result.normalized_cost == 6.0
    assert calls == ["tier1", "tier2"]


def test_invalid_run_audit_fails_closed_without_later_tier() -> None:
    called = False

    def invalid(_):
        return attempt("tier1", authorization(audit=False))

    def forbidden(_):
        nonlocal called
        called = True
        return attempt("tier2", authorization(qualified=True))

    result = run_certificate_gated_cascade(object(), [invalid, forbidden])
    assert result.status is CascadeStatus.INVALID
    assert not called


def test_exact_zero_failure_is_invalid() -> None:
    value = authorization(qualified=True)
    item = ItemCertificate(**{**value.item_certificate.__dict__, "exact_zero_pass": False})
    broken = Authorization(item, value.run_audit)
    result = run_certificate_gated_cascade(object(), [lambda _: attempt("tier1", broken)])
    assert result.status is CascadeStatus.INVALID


def test_action_identity_mismatch_is_invalid() -> None:
    auth = authorization(qualified=True)
    changed = np.ones(9216, dtype=np.float32)
    row = TierAttempt("tier1", auth, changed, tuple(), 1.0, 1.0, 0, "fixture", {})
    result = run_certificate_gated_cascade(object(), [lambda _: row])
    assert result.status is CascadeStatus.INVALID
    assert "ACTION_IDENTITY_MISMATCH" in result.terminal_reason
