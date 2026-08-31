from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np


def action_sha256(action: np.ndarray) -> str:
    """Hash the exact float32 action, including its public shape contract."""
    value = np.ascontiguousarray(np.asarray(action, dtype=np.float32).reshape(-1))
    header = json.dumps(
        {"dtype": "float32", "shape": list(value.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(header + b"\n" + value.tobytes(order="C")).hexdigest()


@dataclass(frozen=True)
class ItemCertificate:
    """A certificate about one action's observed behavior and physical legality.

    Run-level facts such as state restoration and information-path integrity are
    deliberately absent.  They belong to :class:`RunAudit`.
    """

    finite: bool
    physical_contract_pass: bool
    fp32_top1: bool
    bf16_top1: bool
    fp32_effect_peak: bool
    bf16_effect_peak: bool
    fp32_margin: float
    bf16_margin: float
    total_energy: float
    candidate_permutation_error: float
    exact_zero_pass: bool

    def valid(
        self,
        *,
        margin_minimum: float = 0.2,
        energy_maximum: float = 0.021,
        permutation_error_maximum: float = 1e-6,
    ) -> bool:
        return bool(
            self.finite
            and self.physical_contract_pass
            and self.fp32_top1
            and self.bf16_top1
            and self.fp32_effect_peak
            and self.bf16_effect_peak
            and self.fp32_margin >= margin_minimum
            and self.bf16_margin >= margin_minimum
            and self.total_energy <= energy_maximum
            and self.candidate_permutation_error <= permutation_error_maximum
            and self.exact_zero_pass
        )

    def invalid_contract(self, *, hard_energy_maximum: float = 0.021) -> bool:
        return bool(
            not self.finite
            or not self.physical_contract_pass
            or not self.exact_zero_pass
            or self.total_energy > hard_energy_maximum
        )


@dataclass(frozen=True)
class RunAudit:
    """Evidence that a counterfactual replay was isolated and correctly bound."""

    snapshot_restore_pass: bool
    no_state_leakage: bool
    information_path_pass: bool
    evidence_binding_pass: bool
    action_sha256: str
    utility_identity_sha256: str
    protocol_identity_sha256: str

    def valid(self) -> bool:
        identities = (
            self.action_sha256,
            self.utility_identity_sha256,
            self.protocol_identity_sha256,
        )
        return bool(
            self.snapshot_restore_pass
            and self.no_state_leakage
            and self.information_path_pass
            and self.evidence_binding_pass
            and all(len(value) == 64 and all(ch in "0123456789abcdef" for ch in value) for value in identities)
        )


@dataclass(frozen=True)
class Authorization:
    """The conjunction required to let the public runtime commit an action."""

    item_certificate: ItemCertificate
    run_audit: RunAudit

    def permits_commit(self) -> bool:
        return bool(self.item_certificate.valid() and self.run_audit.valid())

    def invalid(self) -> bool:
        return bool(self.item_certificate.invalid_contract() or not self.run_audit.valid())
