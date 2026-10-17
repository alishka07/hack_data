"""Case explanations use host findings, explicit checks, and coverage, not scores alone."""
from copy import deepcopy

import pytest

from worker.decision import build_case_decision


COVERAGE = {"windows_completed": 2, "windows_planned": 2, "visibility_gaps": []}


def verification(decision="supported", value="yes"):
    return {"decision": decision, "visibility": "clear", "checks": {"required_predicate": value},
            "check_frames": {"required_predicate": [1]}, "observed_action": "Synthetic visible event", "confidence": .9}


def attempt(**changes):
    return {"type_id": "event", "episode_id": "episode-1", "timecode_sec": 4.,
            "action": verification(), "context": verification(), **changes}


def queued(**changes):
    return {"type_id": "event", "episode_id": "episode-1", "timecode_sec": 4.,
            "reason_codes": ["legal_context_uncertain"], **changes}


def finding(**changes):
    return {"type_id": "event", "episode_id": "episode-1", "timecode_sec": 4.1,
            "candidate_timecode_sec": 4., "description": "Synthetic host finding", **changes}


def test_no_flags_is_limited_no_supported_violation():
    result = build_case_decision([], [], [], COVERAGE)
    assert result["status"] == "no_supported_violation"
    assert result["cases"] == []
    assert "does not certify compliance" in result["summary"]


def test_only_host_findings_authorize_confirmation():
    attempted = attempt()
    inputs = ([finding()], [], [attempted], deepcopy(COVERAGE))
    before = deepcopy(inputs)
    result = build_case_decision(*inputs)
    assert result["status"] == "confirmed_violation"
    assert result["counts"]["confirmed"] == 1
    assert len(result["cases"]) == 1 and result["cases"][0]["status"] == "confirmed"
    assert {check["stage"] for check in result["cases"][0]["checks"]} == {"action", "context"}
    assert inputs == before
    unconfirmed = build_case_decision([], [], [attempted], COVERAGE)
    assert unconfirmed["status"] == "review_required"
    assert unconfirmed["counts"]["confirmed"] == 0


def test_refinement_resolves_unknown_checks_in_display():
    tried = attempt(action=verification("uncertain", "unknown"), refined_action=verification())
    result = build_case_decision([finding()], [], [tried], COVERAGE)
    case = result["cases"][0]
    assert case["action_decision"] == "supported"
    assert all(check["status"] == "yes" for check in case["checks"])
    assert "action_checks_unresolved" not in case["reason_codes"]


@pytest.mark.parametrize("stage", ["action", "context"])
def test_explicit_contrary_evidence_cannot_be_hidden_by_refinement(stage):
    tried = attempt(**{stage: verification("rejected", "no"), "refined_" + stage: verification()})
    result = build_case_decision([], [queued(reason_codes=["action_rejected"])], [tried], COVERAGE)
    case = result["cases"][0]
    assert result["status"] == "no_supported_violation"
    assert case["status"] == "rejected" and case["requires_review"] is False
    assert case[stage + "_decision"] == "rejected"
    assert any(check["stage"] == stage and check["status"] == "no" for check in case["checks"])
    assert result["counts"]["rejected"] == 1 and result["counts"]["review_required"] == 0


def test_rejected_candidate_without_review_queue_does_not_become_unresolved():
    result = build_case_decision([], [], [attempt(context=verification("rejected", "no"))], COVERAGE)
    assert result["status"] == "no_supported_violation"
    assert result["cases"][0]["status"] == "rejected"
    assert result["counts"]["review_required"] == 0


def test_review_only_candidate_has_explanation_without_verifier_attempt():
    result = build_case_decision([], [queued(reason_codes=["review_only_rule", "omission_unverifiable_from_sparse_samples"])], [], COVERAGE)
    case = result["cases"][0]
    assert result["status"] == case["status"] == "review_required"
    assert case["checks"] == [] and case["action_decision"] == "not_checked"
    assert "cannot establish" in case["reasons"][1]


@pytest.mark.parametrize("coverage, reason", [
    ({**COVERAGE, "visibility_gaps": [{"start_sec": 0, "end_sec": 1, "visibility": "partial"}]}, "visual_coverage_gaps"),
    ({**COVERAGE, "windows_completed": 1}, "incomplete_screening"),
])
def test_coverage_limitations_prevent_an_empty_clean_case(coverage, reason):
    result = build_case_decision([], [], [], coverage)
    assert result["status"] == "review_required"
    assert reason in result["reason_codes"] and result["counts"]["confirmed"] == 0


def test_untouched_manual_defaults_and_evidenced_exemptions_do_not_invent_cases():
    rows = [{"item_id": "manual", "assessment": "unassessed", "score": 100, "scoring_kind": "required_procedure"},
            {"item_id": "exempt", "applicability": "not_applicable", "assessment": "unresolved", "score": None},
            {"item_id": "completed", "assessment": "completed", "score": 0, "scoring_kind": "required_procedure", "pending_episode_count": 0}]
    result = build_case_decision([], [], [], COVERAGE, rows)
    assert result["status"] == "no_supported_violation" and result["cases"] == []


@pytest.mark.parametrize("row", [
    {"assessment": "unresolved", "score": 100, "scoring_kind": "required_procedure"},
    {"assessment": "observed", "score": 100, "scoring_kind": "event_occurrence", "model_confidence": 1.},
    {"assessment": "completion_evidence", "score": 30, "scoring_kind": "required_procedure", "pending_episode_count": 1},
])
def test_unresolved_catalog_evidence_can_request_review_but_never_confirm(row):
    row = {"item_id": "catalog-event", "label": "Synthetic catalog check", "confirmed_finding": False,
           "evidence_refs": [{"timestamp_sec": 2.5}], **row}
    result = build_case_decision([], [], [], COVERAGE, [row])
    assert result["status"] == "review_required"
    assert result["counts"]["confirmed"] == 0 and result["counts"]["unresolved_assessments"] == 1
    assert result["cases"][0]["timecode_sec"] == 2.5


def test_catalog_summary_does_not_duplicate_an_existing_queued_case():
    row = {"item_id": "event", "assessment": "unresolved"}
    result = build_case_decision([], [queued()], [attempt(context=verification("uncertain", "unknown"))], COVERAGE, [row])
    assert len(result["cases"]) == 1
    assert result["counts"]["unresolved_assessments"] == 1


def test_precise_candidate_anchor_matches_only_the_actual_finding_attempt():
    trials = [attempt(), attempt(timecode_sec=4.3, context=verification("uncertain", "unknown"))]
    result = build_case_decision([finding(timecode_sec=4.25)], [queued(timecode_sec=4.3)], trials, COVERAGE)
    assert [case["status"] for case in result["cases"]] == ["confirmed", "review_required"]
    assert result["counts"]["confirmed"] == 1 and result["counts"]["candidates"] == 2


def test_older_finding_without_candidate_anchor_matches_its_candidate_interval():
    old = finding(start_sec=3., end_sec=5.)
    del old["candidate_timecode_sec"]
    result = build_case_decision([old], [], [attempt()], COVERAGE)
    assert len(result["cases"]) == 1 and result["cases"][0]["status"] == "confirmed"


def test_check_evidence_resolves_against_the_effective_pass_request_order():
    original = [{"source_index": 100, "timestamp_sec": 3.}, {"source_index": 110, "timestamp_sec": 3.3}]
    refined = [{"source_index": 200, "timestamp_sec": 6.}, {"source_index": 220, "timestamp_sec": 6.6}]
    tried = attempt(action=verification("uncertain", "unknown"), action_frames=original,
                    refined_action=verification(), refined_action_frames=refined)
    result = build_case_decision([finding()], [], [tried], COVERAGE)
    checks = result["cases"][0]["checks"]
    assert checks[0]["evidence_refs"] == [{"source_index": 220, "timestamp_sec": 6.6}]
    assert checks[1]["evidence_refs"] == []  # Legacy context without a host mapping.


def test_contradiction_preserves_original_pass_source_references():
    tried = attempt(action=verification("rejected", "no"),
                    action_frames=[{"source_index": 10, "timestamp_sec": 1.}, {"source_index": 20, "timestamp_sec": 2.}],
                    refined_action=verification(), refined_action_frames=[{"source_index": 999, "timestamp_sec": 99.}])
    result = build_case_decision([], [], [tried], COVERAGE)
    assert result["cases"][0]["checks"][0]["evidence_refs"] == [{"source_index": 20, "timestamp_sec": 2.}]


def test_out_of_range_request_reference_cannot_create_source_evidence():
    tried = attempt(action_frames=[{"source_index": 10, "timestamp_sec": 1.}])
    result = build_case_decision([], [], [tried], COVERAGE)
    assert result["cases"][0]["checks"][0]["frame_indices"] == [1]
    assert result["cases"][0]["checks"][0]["evidence_refs"] == []


@pytest.mark.parametrize("stage, code", [("action", "action_supported"), ("context", "legal_context_supported")])
@pytest.mark.parametrize("defect", ["zero_confidence", "empty_citations"])
def test_model_support_that_fails_the_host_gate_explains_why_review_remains(stage, code, defect):
    answer = {**verification(), "evidence_indices": [1]}
    if defect == "zero_confidence":
        answer["confidence"] = 0
    else:
        answer["check_frames"] = {"required_predicate": []}
        answer["evidence_indices"] = []
    source_frames = [{"source_index": 100, "timestamp_sec": 3.}, {"source_index": 110, "timestamp_sec": 3.3}]
    tried = attempt(**{stage: answer, stage + "_frames": source_frames})
    result = build_case_decision([], [queued(reason_codes=[code])], [tried], COVERAGE)
    case = result["cases"][0]
    assert result["status"] == case["status"] == "review_required"
    assert result["counts"]["confirmed"] == 0 and case[stage + "_decision"] == "supported"
    explanation = case["reasons"][case["reason_codes"].index(code)]
    assert "model reported" in explanation and "host evidence checks did not pass" in explanation
    assert all(word in explanation for word in ("citations", "confidence", "visibility"))
    stage_check = next(check for check in case["checks"] if check["stage"] == stage)
    assert stage_check["status"] == "yes"
    assert stage_check["evidence_refs"] == ([] if defect == "empty_citations" else [source_frames[1]])
