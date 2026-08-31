"""Core implementation for replay-gated neural execution.

This package is a cleaned reference implementation of the Paper 1 runtime
semantics. It did not generate the archived results and intentionally excludes
private populations, witness tensors, model weights, and future ABI stages.
"""

from .cascade import (
    CascadeResult,
    CascadeStatus,
    TierAttempt,
    run_certificate_gated_cascade,
)
from .authorization import Authorization, ItemCertificate, RunAudit
from .physical_abi import PhysicalContract, contract_metrics
from .semantic import UtilityCell, UtilityIR, UtilityPredicate

__all__ = [
    "CascadeResult",
    "CascadeStatus",
    "Authorization",
    "ItemCertificate",
    "PhysicalContract",
    "RunAudit",
    "TierAttempt",
    "UtilityCell",
    "UtilityIR",
    "UtilityPredicate",
    "contract_metrics",
    "run_certificate_gated_cascade",
]
