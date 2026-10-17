from dataclasses import dataclass

import pytest

from worker.scheduling import (consolidate_candidates, estimate_request_sec, schedule_candidates,
                               verification_admission, verification_can_start)


@dataclass
class Candidate:
    type_id: str
    anchor: float
    start: float
    end: float
    observation: str = "Visible hypothesis"
    episode_id: str = "episode-1"


def test_ongoing_episode_merges_overlapping_windows_with_distant_anchors():
    source = [Candidate("a", 5, 0, 12, "First cue"), Candidate("a", 13, 6, 18, "Second cue")]
    merged = consolidate_candidates(source)
    assert len(merged) == 1
    assert (merged[0].start, merged[0].end, merged[0].anchor) == (0, 18, 5)
    assert "First cue" in merged[0].observation and "Second cue" in merged[0].observation
    assert source[0].end == 12


@pytest.mark.parametrize("other", [Candidate("b", 5, 0, 12),
                                   Candidate("a", 5, 0, 12, episode_id="episode-2"),
                                   Candidate("a", 30, 28, 32),
                                   Candidate("a", 15, 11, 22)])
def test_distinct_rules_episodes_and_separated_events_remain(other):
    assert len(consolidate_candidates([Candidate("a", 5, 0, 12), other])) == 2


def test_unknown_episode_cannot_merge_distant_anchors_using_only_wide_overlap():
    source = [Candidate("a", 5, 0, 12, episode_id=""), Candidate("a", 13, 6, 18, episode_id="")]
    assert len(consolidate_candidates(source)) == 2


def test_existing_close_anchor_deduplication_and_duplicate_observation_retained_once():
    source = [Candidate("a", 3, 2, 4), Candidate("a", 5, 5, 6)]
    merged = consolidate_candidates(source)
    assert len(merged) == 1 and merged[0].observation == "Visible hypothesis"


def test_transitive_same_episode_windows_are_consolidated():
    source = [Candidate("a", 4, 0, 10), Candidate("a", 12, 5, 15), Candidate("a", 17, 10, 20)]
    merged = consolidate_candidates(source)
    assert len(merged) == 1 and (merged[0].start, merged[0].end) == (0, 20)


def test_types_get_first_pass_before_one_types_repeated_separated_episodes():
    source = [Candidate("a", 1, 0, 2), Candidate("a", 12, 11, 13), Candidate("b", 20, 19, 21),
              Candidate("c", 30, 29, 31), Candidate("b", 40, 39, 41)]
    assert [(row.type_id, row.anchor) for row in schedule_candidates(source)] == [
        ("a", 1), ("b", 20), ("c", 30), ("a", 12), ("b", 40)]


def test_admission_preserves_finish_reserve_and_adapts_to_slow_request():
    assert verification_can_start(24)
    assert not verification_can_start(23.9)
    admission = verification_admission(39, [5, 24], reserve_sec=10)
    assert not admission.allowed
    assert admission.required_sec == 40
    assert admission.reason == "verification_budget_exhausted"
    assert verification_can_start(70, [24], reserve_sec=10, calls_required=2)


def test_empty_and_invalid_timings_cannot_underestimate_request_fallback():
    assert schedule_candidates([]) == []
    assert estimate_request_sec([0, -1, float("nan"), float("inf"), True, "bad"]) == 12
    assert estimate_request_sec([40] + [1] * 8) == 12
    assert not verification_can_start(-5)


@pytest.mark.parametrize("kwargs", [{"remaining_sec": float("nan")}, {"remaining_sec": 100, "calls_required": True},
                                  {"remaining_sec": 100, "reserve_sec": -1}, {"remaining_sec": 100, "fallback_sec": 0}])
def test_invalid_budget_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        verification_admission(**kwargs)


def test_invalid_source_interval_rejected():
    with pytest.raises(ValueError):
        consolidate_candidates([Candidate("a", 2, 4, 1)])
