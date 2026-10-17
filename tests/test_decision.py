"""Case explanations use host findings, explicit checks, and coverage, not scores alone."""
from copy import deepcopy

import pytest

from worker.decision import build_case_decision, enrich_case_probabilities


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


def grounded_attempt(**changes):
    frames = [{"source_index": 10, "timestamp_sec": 3.}, {"source_index": 20, "timestamp_sec": 4.}]
    return attempt(action_frames=frames, context_frames=frames, **changes)


@pytest.mark.parametrize("confidence", [0, .74, 1])
def test_case_probability_uses_joint_model_confidence_including_zero(confidence):
    tried = grounded_attempt(action={**verification(), "confidence": confidence})
    before = deepcopy(tried)
    case = build_case_decision([], [], [tried], COVERAGE)["cases"][0]
    assert case["probability"] == min(confidence, .9)
    assert case["probability_scope"] == "joint_checks"
    assert case["probability_calibrated"] is False
    assert tried == before


@pytest.mark.parametrize("confidence", [None, True, False, float("nan"), float("inf"), -.01, 1.01, "0.74"])
def test_invalid_action_confidence_does_not_become_a_probability(confidence):
    case = build_case_decision([], [], [grounded_attempt(action={**verification(), "confidence": confidence})], COVERAGE)["cases"][0]
    assert case["probability"] is None and case["probability_scope"] is None


@pytest.mark.parametrize("defect", ["no_frames", "all_not_visible", "unusable", "host_unverified", "invalid_normalized"])
def test_unobserved_or_invalid_response_has_no_probability(defect):
    tried = grounded_attempt()
    if defect == "no_frames":
        tried["action_frames"] = []
    elif defect == "all_not_visible":
        tried["action"]["checks"] = {"required_predicate": "not_visible"}
    elif defect == "unusable":
        tried["action"]["visibility"] = "unusable"
    elif defect == "host_unverified":
        tried["action"]["_host_unverified"] = True
    else:
        tried["action"].update(confidence=0., semantic_issues=["invalid_confidence"])
    assert build_case_decision([], [], [tried], COVERAGE)["cases"][0]["probability"] is None


def test_rejected_probability_is_not_inverted_and_keeps_original_pass():
    tried = grounded_attempt(action={**verification("rejected", "no"), "confidence": .74},
                             refined_action={**verification(), "confidence": .99})
    case = build_case_decision([], [], [tried], COVERAGE)["cases"][0]
    assert case["status"] == "rejected" and case["probability"] == .74


def test_action_only_probability_and_refined_source_mapping():
    tried = grounded_attempt(context={}, action=verification("uncertain", "unknown"),
                             refined_action={**verification(), "confidence": .74},
                             refined_action_frames=[{"source_index": 70, "timestamp_sec": 5.},
                                                    {"source_index": 80, "timestamp_sec": 6.}])
    case = build_case_decision([], [], [tried], COVERAGE)["cases"][0]
    assert case["probability"] == .74 and case["probability_scope"] == "action_only"


def test_finding_probability_wins_and_legacy_interval_matching_is_preserved():
    found = finding(start_sec=3., end_sec=5., model_confidence=.74,
                    evidence_frames=[{"filename": "event.jpg", "timestamp_sec": 4.1}])
    del found["candidate_timecode_sec"]
    case = build_case_decision([found], [], [grounded_attempt()], COVERAGE)["cases"][0]
    assert case["status"] == "confirmed" and case["probability"] == .74


def test_probabilities_cannot_cross_candidate_time_or_episode_boundaries():
    trials = [grounded_attempt(), grounded_attempt(timecode_sec=4.3, context={}),
              grounded_attempt(episode_id="episode-2", context={})]
    trials[1]["action"]["confidence"] = .51
    trials[2]["action"]["confidence"] = .24
    found = finding(model_confidence=.74, evidence_frames=[{"filename": "event.jpg", "timestamp_sec": 4.1}])
    cases = build_case_decision([found], [], trials, COVERAGE)["cases"]
    assert [case["probability"] for case in cases] == [.74, .51, .24]


def test_catalog_procedure_reports_completion_not_residual_score():
    row = {"item_id": "procedure", "episode_id": "episode-1", "assessment": "completion_evidence",
           "automation": "review_only", "scoring_kind": "required_procedure", "model_confidence": .74,
           "score": 26., "pending_episode_count": 1,
           "evidence_refs": [{"source_index": 4, "timestamp_sec": 2.5}]}
    case = build_case_decision([], [], [], COVERAGE, [row])["cases"][0]
    assert case["probability"] == .74 and case["probability_scope"] == "procedure_completion"


@pytest.mark.parametrize("changes", [{"automation": "manual"}, {"applicability": "not_applicable"},
                                     {"assessment": "unassessed"}, {"evidence_refs": [{"timestamp_sec": 2.5}]}])
def test_catalog_manual_unassessed_excluded_or_unlinked_scores_have_no_probability(changes):
    row = {"item_id": "event", "assessment": "observed", "scoring_kind": "event_occurrence",
           "automation": "review_only", "model_confidence": .74, "score": 74.,
           "evidence_refs": [{"source_index": 4, "timestamp_sec": 2.5}], **changes}
    cases = [{"type_id": "event", "episode_id": "", "timecode_sec": 2.5, "status": "review_required"}]
    assert enrich_case_probabilities(cases, [], [], [row])[0]["probability"] is None


def test_enrichment_does_not_rebuild_saved_decisions_or_modify_inputs():
    cases = [{"type_id": "event", "episode_id": "episode-1", "timecode_sec": 4.,
              "status": "rejected", "reasons": ["Preserved legacy explanation"], "requires_review": False}]
    before = deepcopy(cases)
    result = enrich_case_probabilities(cases, [grounded_attempt()], [])
    assert cases == before
    assert {key: value for key, value in result[0].items() if not key.startswith("probability")} == before[0]


def test_legacy_enrichment_ignores_non_object_diagnostic_list_entries():
    cases = [None, "broken", {"type_id": "event", "episode_id": "episode-1", "timecode_sec": 4., "status": "review_required"}]
    result = enrich_case_probabilities(cases, [None, 42, grounded_attempt()],
                                       [None, "broken"], [None, "broken", {"evidence_refs": None}])
    assert len(result) == 1 and result[0]["probability"] == .9


def test_legacy_enrichment_handles_null_verification_answers():
    cases = [{"type_id": "event", "episode_id": "episode-1", "timecode_sec": 4., "status": "review_required"}]
    result = enrich_case_probabilities(cases, [grounded_attempt(action=None, context=None)], [])
    assert result[0]["probability"] is None and result[0]["probability_scope"] is None


def test_legacy_enrichment_handles_null_citations_without_creating_confidence():
    cases = [{"type_id": "event", "episode_id": "episode-1", "timecode_sec": 4., "status": "review_required"}]
    answer = {**verification(), "evidence_indices": None, "check_frames": None}
    result = enrich_case_probabilities(cases, [grounded_attempt(action=answer, context=answer)], [])
    assert result[0]["probability"] is None and result[0]["probability_calibrated"] is False
