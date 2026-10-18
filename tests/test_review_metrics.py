"""Human review metrics must not turn unknowns or weak anchors into truth."""
from copy import deepcopy

import pytest

from app.review_metrics import build_review_quality


def model_case(case_id="case-1", status="confirmed", **kwargs):
    return {"case_id": case_id, "type_id": "belongings", "status": status,
            "timecode_sec": 10, "probability": .8, "probability_scope": "joint_checks", **kwargs}


def record(cases=None, decisions=None, **state):
    return {"job_id": "synthetic-job", "source_fingerprint": "synthetic-source",
            "model_cases": cases if cases is not None else [model_case()],
            "review_state": {"revision": 1, "reviewer": "Проверяющий", "coverage_complete": True,
                             "source_group": "original-family", "camera_id": "camera-a", "session_id": "session-a",
                             "decisions": decisions if decisions is not None else [{"case_id": "case-1", "decision": "confirmed", "start_sec": 9, "end_sec": 14}],
                             "missed_events": [], **state}}


def test_empty_and_unreviewed_do_not_claim_zero_accuracy_or_calibration():
    report = build_review_quality([])
    assert report["status"] == "no_reviewed_labels"
    assert report["totals"]["automatic_decision"]["precision"] is None
    assert report["totals"]["event_coverage"]["recall"] is None
    assert report["totals"]["probability_evaluation"]["brier_score"] is None
    unreviewed = record(decisions=[], coverage_complete=False)
    report = build_review_quality([unreviewed])
    assert report["totals"]["candidate_adjudication"]["unreviewed"] == 1
    assert report["totals"]["automatic_decision"]["false_positive"] == 0
    assert report["totals"]["automatic_decision"]["true_negative"] == 0
    assert report["totals"]["event_coverage"]["recall"] is None


def test_needs_review_is_candidate_found_but_not_automatic_violation():
    item = record(cases=[model_case("one", "review_required"), model_case("two", "rejected"),
                         model_case("three", "confirmed"), model_case("four", "confirmed")],
                  decisions=[{"case_id": "one", "decision": "confirmed"},
                             {"case_id": "two", "decision": "confirmed"},
                             {"case_id": "three", "decision": "dismissed"},
                             {"case_id": "four", "decision": "uncertain"}])
    result = build_review_quality([item])
    automatic = result["totals"]["automatic_decision"]
    assert automatic["false_negative"] == 2
    assert automatic["false_positive"] == 1
    assert automatic["true_positive"] == automatic["true_negative"] == 0
    assert automatic["precision"] == 0
    assert result["totals"]["candidate_adjudication"]["confirmed"] == 2
    assert result["totals"]["known_misses"]["total"] == 2
    # Attesting full coverage cannot erase an unresolved row.
    assert result["records"]["eligible_complete_coverage"] == 0
    assert result["totals"]["event_coverage"]["recall"] is None


def test_manual_misses_count_once_and_do_not_duplicate_adjudicated_events():
    misses = [
        {"event_id": "matches-existing", "type_id": "belongings", "start_sec": 9, "end_sec": 14},
        {"event_id": "new", "type_id": "belongings", "start_sec": 30, "end_sec": 35},
        {"event_id": "duplicate-ui-id", "type_id": "belongings", "start_sec": 30, "end_sec": 35},
        {"event_id": "different-type", "type_id": "screen", "start_sec": 9, "end_sec": 14},
    ]
    result = build_review_quality([record(missed_events=misses)])
    total = result["totals"]
    assert total["known_misses"]["manual_events"] == 2
    assert total["deduplicated_missed_events"] == 2
    assert total["event_coverage"]["true_positive"] == 1
    assert total["event_coverage"]["false_negative"] == 2
    assert total["event_coverage"]["recall"] == pytest.approx(1 / 3)
    # A missing candidate is distinct from candidate adjudication.
    assert total["automatic_decision"]["reviewed_candidate_recall"] == 1
    per_type = {item["type_id"]: item for item in result["per_type"]}
    assert per_type["screen"]["event_coverage"]["recall"] == 0


def test_two_different_missed_events_can_match_only_one_existing_case():
    result = build_review_quality([record(missed_events=[
        {"event_id": "a", "type_id": "belongings", "start_sec": 9, "end_sec": 11},
        {"event_id": "b", "type_id": "belongings", "start_sec": 12, "end_sec": 14},
    ])])
    assert result["totals"]["deduplicated_missed_events"] == 1
    assert result["totals"]["known_misses"]["manual_events"] == 1


@pytest.mark.parametrize("field", ["source_group", "camera_id", "session_id", "reviewer"])
def test_missing_review_metadata_prevents_full_event_recall(field):
    result = build_review_quality([record(**{field: ""})])
    assert result["totals"]["automatic_decision"]["precision"] == 1
    assert result["totals"]["event_coverage"]["recall"] is None


def test_incomplete_video_does_not_contaminate_eligible_event_denominator():
    complete = record()
    partial = record(cases=[model_case(status="rejected")], coverage_complete=False)
    partial["source_fingerprint"] = "different-source"
    result = build_review_quality([complete, partial])
    assert result["totals"]["automatic_decision"]["reviewed_candidate_recall"] == .5
    assert result["totals"]["event_coverage"]["recall"] == 1
    assert result["totals"]["event_coverage"]["has_excluded_records"] is True
    assert result["records"]["eligible_complete_coverage"] == 1


def test_anchor_is_not_interval_and_human_boundaries_do_not_rewrite_model():
    item = record()
    original = deepcopy(item)
    result = build_review_quality([item])
    assert result["totals"]["temporal_localization"]["mean_iou"] is None
    assert result["totals"]["temporal_localization"]["anchors_without_intervals"] == 1
    assert item == original
    item["model_cases"][0].update(start_sec=10, end_sec=15)
    result = build_review_quality([item])
    assert result["totals"]["temporal_localization"]["pairs"] == 1
    assert result["totals"]["temporal_localization"]["mean_iou"] == pytest.approx(4 / 6)
    assert result["totals"]["temporal_localization"]["iou_at_0_5"] == 1


def test_probabilities_must_be_valid_and_match_human_violation_semantics():
    configurations = [
        (.8, "joint_checks", "confirmed"), (.2, "event_occurrence", "dismissed"),
        (.9, "action_only", "dismissed"), (.99, "procedure_completion", "confirmed"),
        (True, "joint_checks", "confirmed"), ("0.5", "joint_checks", "confirmed"),
        (float("nan"), "joint_checks", "confirmed"), (-.1, "joint_checks", "confirmed"),
        (1.1, "joint_checks", "confirmed"), (.7, "joint_checks", "uncertain"),
        (None, "joint_checks", "confirmed"),
    ]
    cases, decisions = [], []
    for index, (probability, scope, label) in enumerate(configurations):
        identifier = str(index)
        cases.append(model_case(identifier, probability=probability, probability_scope=scope))
        decisions.append({"case_id": identifier, "decision": label})
    calibration = build_review_quality([record(cases, decisions)])["totals"]["probability_evaluation"]
    assert calibration["samples"] == 2
    assert calibration["brier_score"] == pytest.approx(.04)
    assert calibration["expected_calibration_error"] == pytest.approx(.2)
    assert calibration["excluded"] == {"unreviewed_or_uncertain": 1, "incompatible_scope": 2, "invalid_probability": 6}
    assert calibration["establishes_calibration"] is False


def test_probability_endpoints_have_valid_reliability_bins():
    item = record([model_case("zero", probability=0), model_case("one", probability=1)],
                  [{"case_id": "zero", "decision": "dismissed"}, {"case_id": "one", "decision": "confirmed"}])
    calibration = build_review_quality([item])["totals"]["probability_evaluation"]
    assert calibration["brier_score"] == 0
    assert calibration["bins"][0]["samples"] == 1
    assert calibration["bins"][9]["samples"] == 1


def test_exact_duplicate_jobs_use_latest_human_review_once():
    previous = record(decisions=[{"case_id": "case-1", "decision": "dismissed"}])
    previous["review_updated_at"] = "2026-10-18T10:00:00Z"
    latest = record()
    latest["review_updated_at"] = "2026-10-18T11:00:00Z"
    unreviewed_later = record(decisions=[], coverage_complete=False)
    unreviewed_later["review_updated_at"] = "2026-10-19T10:00:00Z"
    result = build_review_quality([latest, previous, unreviewed_later])
    assert result["records"]["exact_duplicate_analyses_excluded"] == 2
    assert result["totals"]["automatic_decision"]["true_positive"] == 1
    assert result["totals"]["automatic_decision"]["false_positive"] == 0


def test_group_audit_keeps_augmented_families_and_shared_cameras_together():
    original = record()
    augmented = record(camera_id="camera-b")
    augmented["source_fingerprint"] = "augmentation-different-bytes"
    camera_sibling = record(source_group="new-family", camera_id="camera-b", session_id="other-session")
    camera_sibling["source_fingerprint"] = "other-video"
    independent = record(source_group="separate", camera_id="camera-c", session_id="session-c")
    independent["source_fingerprint"] = "independent-video"
    report = build_review_quality([original, augmented, camera_sibling, independent])
    grouping = report["grouping"]
    assert grouping["source_families"] == 3
    assert grouping["independent_camera_groups"] == 2
    assert sorted(item["record_count"] for item in grouping["split_manifest"]) == [1, 3]
    assert grouping["status"] == "insufficient_groups"
    assert grouping["split_performed"] is False
    # Aggregate reports do not expose IDs, source names or review comments.
    assert all(item["partition"] == "unassigned" for item in grouping["split_manifest"])


def test_unknown_groups_do_not_invent_independence_and_conflicting_labels_are_unknown():
    item = record(decisions=[{"case_id": "case-1", "decision": "confirmed"},
                             {"case_id": "case-1", "decision": "dismissed"},
                             {"case_id": "no-model-case", "decision": "confirmed"}], source_group="")
    result = build_review_quality([item])
    assert result["grouping"]["status"] == "missing_metadata"
    assert result["grouping"]["unknown_group_records"] == 1
    assert result["records"]["orphan_decisions"] == 1
    assert result["totals"]["candidate_adjudication"]["uncertain"] == 1
    assert result["totals"]["automatic_decision"]["precision"] is None


def test_duplicate_case_ids_do_not_get_two_labels_from_one_human_decision():
    result = build_review_quality([record([model_case(), model_case()])])
    assert result["records"]["duplicate_case_ids"] == 1
    assert result["totals"]["candidate_adjudication"]["unreviewed"] == 2
    assert result["records"]["eligible_complete_coverage"] == 0


def test_identical_reviewed_intervals_are_one_event_despite_multiple_proposals():
    item = record([model_case("first"), model_case("second", "review_required")],
                  [{"case_id": "first", "decision": "confirmed", "start_sec": 9, "end_sec": 14},
                   {"case_id": "second", "decision": "confirmed", "start_sec": 9, "end_sec": 14}])
    result = build_review_quality([item])["totals"]
    assert result["automatic_decision"]["true_positive"] == 1
    assert result["automatic_decision"]["false_negative"] == 1
    assert result["event_coverage"]["true_positive"] == 1
    assert result["event_coverage"]["false_negative"] == 0
    assert result["event_coverage"]["recall"] == 1


def test_ambiguous_overlaps_and_missing_human_intervals_prevent_event_accuracy():
    cases = [model_case("first"), model_case("second", "review_required")]
    decisions = [{"case_id": "first", "decision": "confirmed", "start_sec": 9, "end_sec": 14},
                 {"case_id": "second", "decision": "confirmed", "start_sec": 12, "end_sec": 17}]
    result = build_review_quality([record(cases, decisions)])
    assert result["records"]["eligible_complete_coverage"] == 0
    assert result["totals"]["event_coverage"]["recall"] is None
    decisions[1].pop("end_sec")
    result = build_review_quality([record(cases, decisions)])
    assert result["records"]["eligible_complete_coverage"] == 0
    assert result["totals"]["automatic_decision"]["reviewed_candidate_recall"] == .5


def test_covered_clean_video_has_no_event_recall_denominator():
    result = build_review_quality([record(cases=[], decisions=[])])
    assert result["records"]["eligible_complete_coverage"] == 1
    assert result["totals"]["event_coverage"]["recall"] is None
    assert result["totals"]["automatic_decision"]["precision"] is None


def test_missing_candidate_can_be_sole_reference_on_covered_video():
    item = record(cases=[], decisions=[], missed_events=[
        {"event_id": "entirely-missed", "type_id": "screen", "start_sec": 20, "end_sec": 30}])
    result = build_review_quality([item])
    assert result["totals"]["event_coverage"]["recall"] == 0
    assert result["totals"]["known_misses"]["manual_events"] == 1
    assert result["totals"]["automatic_decision"]["reviewed_candidate_recall"] is None


def test_unknown_model_status_is_not_negative_and_blocks_event_metrics():
    result = build_review_quality([record(cases=[model_case(status="future-status")])])
    assert result["totals"]["automatic_decision"]["false_negative"] == 0
    assert result["totals"]["unknown_model_status"] == 1
    assert result["records"]["eligible_complete_coverage"] == 0


def test_duplicate_analysis_does_not_erase_old_provenance_links():
    old = record(source_group="family-old", camera_id="camera-old")
    old["review_updated_at"] = 1
    new = record(source_group="family-new", camera_id="camera-new")
    new["review_updated_at"] = 2
    sibling = record(source_group="family-old", camera_id="camera-old")
    sibling["source_fingerprint"] = "augmentation-of-old"
    report = build_review_quality([old, new, sibling])
    assert report["records"]["evaluated"] == 2
    assert report["grouping"]["independent_camera_groups"] == 1
    assert report["grouping"]["split_manifest"][0]["record_count"] == 3


def test_unreviewed_duplicates_still_choose_latest_model_by_creation_time():
    old = record(cases=[model_case(type_id="old-type")], decisions=[], coverage_complete=False)
    old.update(created_at=1, review_updated_at=None)
    new = record(cases=[model_case(type_id="new-type")], decisions=[], coverage_complete=False)
    new.update(created_at=2, review_updated_at=None)
    report = build_review_quality([new, old])
    assert [item["type_id"] for item in report["per_type"]] == ["new-type"]


def test_unbounded_integer_inputs_do_not_crash_or_turn_into_probabilities():
    item = record(cases=[model_case(probability=10 ** 10000, start_sec=10 ** 10000, end_sec=10 ** 10001)])
    result = build_review_quality([item])
    assert result["totals"]["probability_evaluation"]["samples"] == 0
    assert result["totals"]["probability_evaluation"]["excluded"]["invalid_probability"] == 1
    assert result["totals"]["temporal_localization"]["pairs"] == 0
