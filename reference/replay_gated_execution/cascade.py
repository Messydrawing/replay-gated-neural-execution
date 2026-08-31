from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterable, Protocol

from .authorization import Authorization, action_sha256


class CascadeStatus(str, Enum):
    AUTHORIZED_FOR_FINAL_REPLAY = "AUTHORIZED_FOR_FINAL_REPLAY"
    SEARCH_MISS_ABSTAIN = "SEARCH_MISS_ABSTAIN"
    INVALID = "INVALID"


@dataclass(frozen=True)
class TierAttempt:
    tier_name: str
    authorization: Authorization
    action: Any | None
    trajectory: tuple[dict[str, Any], ...]
    normalized_cost: float
    paired_replay_count: float
    optimizer_updates: int
    terminal_reason: str
    diagnostics: dict[str, Any]

    def __post_init__(self) -> None:
        if self.normalized_cost < 0 or self.paired_replay_count < 0:
            raise ValueError("tier costs must be non-negative")
        if self.optimizer_updates < 0:
            raise ValueError("optimizer updates must be non-negative")


class Tier(Protocol):
    name: str

    def __call__(self, semantic_request: Any) -> TierAttempt: ...


@dataclass(frozen=True)
class CascadeResult:
    status: CascadeStatus
    selected_action: Any | None
    selected_tier: str | None
    attempts: tuple[TierAttempt, ...]
    normalized_cost: float
    paired_replay_count: float
    terminal_reason: str


def run_certificate_gated_cascade(
    semantic_request: Any,
    tiers: Iterable[Callable[[Any], TierAttempt]],
) -> CascadeResult:
    """Run independent tiers and select an attempt for mandatory final replay.

    The controller never inspects capability/context identifiers and never forwards
    a failed tier's action or optimizer state into the next tier. This function
    never commits; :func:`replay.select_and_commit` is the commit authority.
    """

    attempts: list[TierAttempt] = []
    total_cost = 0.0
    total_replays = 0.0
    for tier in tiers:
        attempt = tier(semantic_request)
        attempts.append(attempt)
        total_cost += float(attempt.normalized_cost)
        total_replays += float(attempt.paired_replay_count)

        if attempt.authorization.invalid():
            return CascadeResult(
                status=CascadeStatus.INVALID,
                selected_action=None,
                selected_tier=None,
                attempts=tuple(attempts),
                normalized_cost=total_cost,
                paired_replay_count=total_replays,
                terminal_reason=f"{attempt.tier_name}:INVALID_CONTRACT_OR_IDENTITY",
            )

        if attempt.authorization.permits_commit():
            if attempt.action is None:
                return CascadeResult(
                    status=CascadeStatus.INVALID,
                    selected_action=None,
                    selected_tier=None,
                    attempts=tuple(attempts),
                    normalized_cost=total_cost,
                    paired_replay_count=total_replays,
                    terminal_reason=f"{attempt.tier_name}:CERTIFIED_WITHOUT_ACTION",
                )
            if action_sha256(attempt.action) != attempt.authorization.run_audit.action_sha256:
                return CascadeResult(
                    status=CascadeStatus.INVALID,
                    selected_action=None,
                    selected_tier=None,
                    attempts=tuple(attempts),
                    normalized_cost=total_cost,
                    paired_replay_count=total_replays,
                    terminal_reason=f"{attempt.tier_name}:ACTION_IDENTITY_MISMATCH",
                )
            return CascadeResult(
                status=CascadeStatus.AUTHORIZED_FOR_FINAL_REPLAY,
                selected_action=attempt.action,
                selected_tier=attempt.tier_name,
                attempts=tuple(attempts),
                normalized_cost=total_cost,
                paired_replay_count=total_replays,
                terminal_reason=f"{attempt.tier_name}:AUTHORIZED_PENDING_FINAL_REPLAY",
            )

    return CascadeResult(
        status=CascadeStatus.SEARCH_MISS_ABSTAIN,
        selected_action=None,
        selected_tier=None,
        attempts=tuple(attempts),
        normalized_cost=total_cost,
        paired_replay_count=total_replays,
        terminal_reason="ALL_TIERS_EXHAUSTED_WITHOUT_CERTIFICATE",
    )
