"""Catalog policy and episode separation use synthetic metadata only."""
from pathlib import Path

import pytest

from worker.catalog import clip_assessments, load_catalog
from worker.pipeline import load_rules


ROOT = Path(__file__).resolve().parents[1]


def event(identifier="active_intake_departure", episode="one", timestamp=1):
    return {"type_id": identifier, "episode_id": episode, "timecode_sec": timestamp,
            "evidence_frames": [{"filename": f"event-{episode}.jpg", "timestamp_sec": timestamp}]}


def test_catalog_automatic_scope_matches_the_enabled_legal_gate():
    catalog = load_catalog(ROOT)
    automatic = {item["id"] for item in catalog["items"] if item["automation"] == "automatic_when_supported"}
    enabled = {key for key, rule in load_rules(ROOT).items() if rule.get("auto_emit", True)}
    assert automatic == enabled
    assert len(catalog["items"]) == 33
    assert all(item["initial_score"] == (0 if item["scoring_kind"] == "event_occurrence" else 100)
               for item in catalog["items"])
    items = {item["id"]: item for item in catalog["items"]}
    for item in items.values():
        for related in item.get("related_ids", []):
            assert item["id"] in items[related]["related_ids"]


def test_unmapped_vault_review_does_not_invent_a_clause_or_automatic_finding():
    items = {item["id"]: item for item in load_catalog(ROOT)["items"]}
    item = items["vault_cash_segregation_storage"]
    assert item["automation"] == "manual" and item["legal_status"] == "unmapped_review_only"
    assert item["paragraph"] == "unmapped" and not item["source_url"]
    assert item["id"] not in load_rules(ROOT)


def test_review_proposals_do_not_change_initial_scores_or_confirm_findings():
    catalog = load_catalog(ROOT)
    rows, episodes = clip_assessments(catalog, [], [
        {"type_id": "active_intake_departure"}, {"type_id": "loose_cash_issued_without_observed_count"},
        {"type_id": "unmapped_visual_event"}], 10)
    assert not episodes
    assert all(row["score"] == row["initial_score"] and not row["confirmed_finding"] for row in rows)
    assert {row["item_id"] for row in rows if row["assessment"] == "unresolved"} == {
        "active_intake_departure", "loose_cash_issued_without_observed_count"}
    assert all(row["applicability"] == "unknown" and not row["evidence_refs"] for row in rows)


def test_verified_event_episodes_have_independent_evidence_and_do_not_clear_procedures():
    rows, episodes = clip_assessments(load_catalog(ROOT), [event(), event(episode="two", timestamp=8)], [], 10)
    observed = [row for row in rows if row["confirmed_finding"]]
    assert len(observed) == 1 and observed[0]["confirmed_occurrences"] == 2
    assert len(episodes) == 2 and [e["episode_id"] for e in episodes] == ["one", "two"]
    assert [e["evidence_refs"][0]["timestamp_sec"] for e in episodes] == [1, 8]
    assert all(row["score"] == 100 and row["assessment"] == "unassessed"
               for row in rows if row["scoring_kind"] == "required_procedure")
    fresh, _ = clip_assessments(load_catalog(ROOT), [], [], 10)
    assert all(row["score"] == row["initial_score"] and not row["confirmed_finding"] for row in fresh)


@pytest.mark.parametrize("identifier", ["unmapped_visual_event", "personal_belongings_in_cash_unit",
                                       "loose_cash_issued_without_observed_count"])
def test_broad_catalog_cannot_expand_automatic_finding_scope(identifier):
    with pytest.raises(ValueError, match="outside the automatic"):
        clip_assessments(load_catalog(ROOT), [event(identifier=identifier)], [], 10)


def observation(identifier, confidence, episode="one", timestamp=1, state=None,
                applicability="unknown", scope_certainty="observed_start"):
    return {"item_id": identifier, "confidence": confidence, "episode_id": episode,
            "assessment": state or ("event_evidence" if identifier == "active_intake_departure" else "completion_evidence"),
            "applicability": applicability, "observable": True, "scope_certainty": scope_certainty,
            "evidence_refs": [{"source_index": int(timestamp * 30), "timestamp_sec": timestamp}],
            "start_sec": timestamp, "end_sec": timestamp}


def test_event_and_right_protocol_confidence_update_without_any_jury_finding():
    rows, episodes = clip_assessments(load_catalog(ROOT), [], [], 10, confidence_observations=[
        observation("active_intake_departure", .68), observation("intake_banknotes_counted", .82)])
    by_id = {row["item_id"]: row for row in rows}
    assert by_id["active_intake_departure"]["score"] == 68
    assert by_id["intake_banknotes_counted"]["score"] == 18
    assert len(episodes) == 2 and not any(row["confirmed_finding"] for row in rows)


def test_one_protocol_completion_never_clears_a_different_pending_episode():
    rows, episodes = clip_assessments(load_catalog(ROOT), [], [], 10, confidence_observations=[
        observation("intake_banknotes_counted", .95),
        observation("intake_banknotes_counted", 0, episode="two", timestamp=8, state="unresolved",
                    applicability="applicable")])
    summary = next(row for row in rows if row["item_id"] == "intake_banknotes_counted")
    assert [episode["score"] for episode in episodes] == [5, 100]
    assert summary["score"] == 100 and summary["episode_count"] == 2
    assert summary["latest_episode_id"] == "two" and summary["pending_episode_count"] == 2


@pytest.mark.parametrize("identifier, score", [
    ("intake_banknotes_counted", 65), ("active_intake_departure", 35)])
def test_source_grounded_partial_support_changes_estimate_without_a_jury_finding(identifier, score):
    rows, episodes = clip_assessments(load_catalog(ROOT), [], [], 10, confidence_observations=[
        observation(identifier, .35, state="partial_support")])
    summary = next(row for row in rows if row["item_id"] == identifier)
    assert summary["score"] == score and summary["model_confidence"] == .35
    assert summary["applicability"] == "unknown" and summary["evidence_refs"]
    assert len(episodes) == 1 and not any(row["confirmed_finding"] for row in rows)


@pytest.mark.parametrize("unknown_first", [False, True])
def test_untriggered_unknown_window_does_not_mask_completion_but_marks_incomplete_coverage(unknown_first):
    supported = observation("intake_banknotes_counted", .82)
    unknown = observation("intake_banknotes_counted", 0, episode="unscoped-window-1", timestamp=8,
                          state="unresolved", scope_certainty="window_only")
    observations = [unknown, supported] if unknown_first else [supported, unknown]
    rows, episodes = clip_assessments(load_catalog(ROOT), [], [], 10,
                                    confidence_observations=observations, known_episode_ids={"one"})
    summary = next(row for row in rows if row["item_id"] == "intake_banknotes_counted")
    assert summary["score"] == 18 and summary["assessment"] == "completion_evidence"
    assert summary["episode_count"] == 1 and [row["episode_id"] for row in episodes] == ["one"]
    assert summary["unscoped_unknown_observation_count"] == 1
    assert summary["unassessed_episode_count"] == 0 and summary["episode_coverage_complete"] is False
    assert not summary["confirmed_finding"]


def test_only_unknown_untriggered_windows_retain_baseline_without_pending_episodes():
    rows, episodes = clip_assessments(load_catalog(ROOT), [], [], 10, confidence_observations=[
        observation("intake_banknotes_counted", 0, state="unresolved", scope_certainty="window_only")])
    summary = next(row for row in rows if row["item_id"] == "intake_banknotes_counted")
    assert not episodes and summary["episode_count"] == 0 and summary["pending_episode_count"] == 0
    assert summary["score"] == 100 and summary["assessment"] == "unresolved"
    assert summary["unscoped_unknown_observation_count"] == 1
    assert summary["episode_coverage_complete"] is False


def test_applicable_pending_procedure_is_retained_even_without_a_known_transaction_boundary():
    rows, episodes = clip_assessments(load_catalog(ROOT), [], [], 10, confidence_observations=[
        observation("intake_banknotes_counted", .82),
        observation("intake_banknotes_counted", 0, episode="two", timestamp=8, state="unresolved",
                    applicability="applicable", scope_certainty="window_only")])
    summary = next(row for row in rows if row["item_id"] == "intake_banknotes_counted")
    assert summary["score"] == 100 and summary["episode_count"] == 2 and len(episodes) == 2
    assert summary["unscoped_unknown_observation_count"] == 0


@pytest.mark.parametrize("confidence", [0, -.1, 1.1, True, float("nan"), float("inf")])
def test_partial_support_requires_a_positive_finite_confidence(confidence):
    with pytest.raises(ValueError, match="positive confidence"):
        clip_assessments(load_catalog(ROOT), [], [], 10, confidence_observations=[
            observation("intake_banknotes_counted", confidence, state="partial_support")])


def test_weaker_or_repeated_observations_do_not_add_confidence():
    observed = observation("intake_banknotes_counted", .82)
    rows, episodes = clip_assessments(load_catalog(ROOT), [], [], 10, confidence_observations=[
        observed, observed, observation("intake_banknotes_counted", .60, timestamp=2)])
    assert len(episodes) == 1 and episodes[0]["score"] == 18
    assert len(episodes[0]["evidence_refs"]) == 1
    assert next(row for row in rows if row["item_id"] == "intake_banknotes_counted")["score"] == 18


def test_missing_check_in_a_known_transaction_is_explicitly_incomplete_coverage():
    rows, _ = clip_assessments(load_catalog(ROOT), [], [], 10,
                              confidence_observations=[observation("intake_banknotes_counted", .82)],
                              known_episode_ids={"one", "two"})
    summary = next(row for row in rows if row["item_id"] == "intake_banknotes_counted")
    assert summary["score"] == 18
    assert summary["unassessed_episode_count"] == 1 and summary["known_episode_count"] == 2
    assert summary["episode_coverage_complete"] is False
