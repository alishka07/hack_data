"""Source-only candidate consolidation and deadline-aware model admission.

These helpers never inspect media paths, labels or expected verdicts. They do not
turn unverified candidates into negative results when the deadline is short.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, replace
import math
from typing import Iterable, Protocol, TypeVar


class CandidateLike(Protocol):
    type_id: str
    anchor: float
    start: float
    end: float
    observation: str
    episode_id: str


CandidateT = TypeVar("CandidateT", bound=CandidateLike)


def _same_event(left: CandidateLike, right: CandidateLike) -> bool:
    if left.type_id != right.type_id or left.episode_id != right.episode_id:
        return False
    overlap = min(left.end, right.end) - max(left.start, right.start)
    if abs(left.anchor - right.anchor) <= 3 and overlap >= -1:
        return True
    # An actual shared episode permits matching ongoing actions whose screening
    # anchors are farther apart. An unknown episode cannot justify that merge.
    shorter = min(left.end - left.start, right.end - right.start)
    return bool(left.episode_id) and shorter > 0 and overlap > 0 and overlap / shorter >= .5


def consolidate_candidates(candidates: Iterable[CandidateT]) -> list[CandidateT]:
    """Merge repeated hypotheses without mutating input dataclass instances.

    A distinct rule, known episode, or separated time interval remains separate.
    A wide interval alone is insufficient without substantial overlap. The first
    anchor stays authoritative; observations retain each distinct hypothesis.
    """
    pending = sorted(candidates, key=lambda item: (item.anchor, item.type_id, item.start, item.end))
    merged: list[CandidateT] = []
    for candidate in pending:
        values = (candidate.anchor, candidate.start, candidate.end)
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) for value in values) or candidate.start > candidate.end:
            raise ValueError("Candidates require finite, ordered source timestamps")
        matches = [item for item in merged if _same_event(item, candidate)]
        if not matches:
            merged.append(replace(candidate))
            continue
        representative = matches[0]
        observations = list(dict.fromkeys(item.observation.strip() for item in [*matches, candidate]
                                          if item.observation.strip()))
        combined = replace(representative,
                           start=min(item.start for item in [*matches, candidate]),
                           end=max(item.end for item in [*matches, candidate]),
                           observation=" | ".join(observations))
        matched_ids = {id(item) for item in matches}
        merged = [item for item in merged if id(item) not in matched_ids]
        merged.append(combined)
    return sorted(merged, key=lambda item: (item.anchor, item.type_id))


def schedule_candidates(candidates: Iterable[CandidateT]) -> list[CandidateT]:
    """Give every observed rule type a first pass before its repeated episodes.

    Within each rule, chronological order is retained. This changes only work
    ordering; every consolidated candidate remains in the returned schedule.
    """
    groups: dict[str, deque[CandidateT]] = defaultdict(deque)
    for candidate in consolidate_candidates(candidates):
        groups[candidate.type_id].append(candidate)
    order = sorted(groups, key=lambda key: (groups[key][0].anchor, key))
    scheduled: list[CandidateT] = []
    while any(groups.values()):
        for key in order:
            if groups[key]:
                scheduled.append(groups[key].popleft())
    return scheduled


def estimate_request_sec(observed_latencies: Iterable[float] = (), *, fallback_sec: float = 12.) -> float:
    """Use recent slow requests with 25% headroom, never below the fallback."""
    if not math.isfinite(fallback_sec) or fallback_sec <= 0:
        raise ValueError("The request fallback must be positive and finite")
    observed = [float(value) for value in observed_latencies
                if type(value) in (int, float) and math.isfinite(value) and value > 0][-8:]
    return max(fallback_sec, max(observed, default=0.) * 1.25)


@dataclass(frozen=True)
class VerificationAdmission:
    allowed: bool
    remaining_sec: float
    estimated_request_sec: float
    required_sec: float
    reserve_sec: float
    reason: str


def verification_admission(remaining_sec: float, observed_latencies: Iterable[float] = (), *,
                           reserve_sec: float = 12., fallback_sec: float = 12.,
                           calls_required: int = 1) -> VerificationAdmission:
    """Keep final JSON/evidence time available before starting another request."""
    if not math.isfinite(remaining_sec):
        raise ValueError("The remaining deadline must be finite")
    if not math.isfinite(reserve_sec) or reserve_sec < 0:
        raise ValueError("The finishing reserve must be finite and nonnegative")
    if type(calls_required) is not int or calls_required < 1:
        raise ValueError("At least one whole model call must be reserved")
    estimated = estimate_request_sec(observed_latencies, fallback_sec=fallback_sec)
    required = reserve_sec + estimated * calls_required
    allowed = remaining_sec >= required
    return VerificationAdmission(allowed, max(0., remaining_sec), estimated, required, reserve_sec,
                                 "within_budget" if allowed else "verification_budget_exhausted")


def verification_can_start(remaining_sec: float, observed_latencies: Iterable[float] = (), **kwargs) -> bool:
    """Boolean convenience wrapper; skipped work must be reported as unverified."""
    return verification_admission(remaining_sec, observed_latencies, **kwargs).allowed
