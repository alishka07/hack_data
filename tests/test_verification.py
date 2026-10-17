"""Semantic contradictions cannot survive strict JSON-shaped verifier output."""
from copy import deepcopy
import math

import pytest

from worker.verification import normalize_verification


def response(**changes):
    return {"decision": "supported", "confidence": .9, "visibility": "clear",
            "checks": {"action_visible": "yes"}, "check_frames": {"action_visible": [0]},
            "evidence_indices": [0], "observed_action": "A bag is placed on the counter.",
            "limitations": [], **changes}


def normalize(answer, count=4):
    return normalize_verification(answer, ["action_visible"], count)


def test_consistent_support_and_input_are_preserved():
    answer = response()
    before = deepcopy(answer)
    normalized = normalize(answer)
    assert normalized == {**answer, "semantic_issues": []}
    normalized["checks"]["action_visible"] = "no"
    assert answer == before


@pytest.mark.parametrize("decision", ["supported", "uncertain", "rejected"])
def test_any_explicit_no_rejects_even_with_limited_visibility(decision):
    result = normalize(response(decision=decision, visibility="unusable", checks={"action_visible": "no"}))
    assert result["decision"] == "rejected"
    assert "explicit_contradiction" in result["semantic_issues"]


@pytest.mark.parametrize("status", ["unknown", "not_visible", None, True, "yes please", [], {}])
def test_unresolved_or_invalid_checks_cannot_support(status):
    result = normalize(response(checks={"action_visible": status}))
    assert result["decision"] == "uncertain"
    assert result["check_frames"]["action_visible"] == []


@pytest.mark.parametrize("visibility", ["partial", "unusable", "", None, [], {}])
def test_visibility_prevents_support(visibility):
    assert normalize(response(visibility=visibility))["decision"] == "uncertain"


@pytest.mark.parametrize("indices", [[], [True], [-1], [4], [0, 999], [0, 1, 2, 3], "0", None])
def test_invalid_or_missing_check_citations_are_unknown(indices):
    result = normalize(response(check_frames={"action_visible": indices}))
    assert result["decision"] == "uncertain"
    assert result["checks"]["action_visible"] == "unknown"
    assert result["check_frames"]["action_visible"] == []


@pytest.mark.parametrize("indices", [[], [True], [-1], [4], [0, 999], [0, 1, 2, 3], "0", None])
def test_invalid_or_missing_global_evidence_prevents_support(indices):
    result = normalize(response(evidence_indices=indices))
    assert result["decision"] == "uncertain"
    assert all(type(index) is int and 0 <= index < 4 for index in result["evidence_indices"])


@pytest.mark.parametrize("confidence", [True, False, float("nan"), float("inf"), -.1, 1.1, None, "0.9", 0, 10**1000])
def test_invalid_confidence_cannot_support_or_serialize_nan(confidence):
    result = normalize(response(confidence=confidence))
    assert result["decision"] == "uncertain"
    assert type(result["confidence"]) is float and math.isfinite(result["confidence"])


@pytest.mark.parametrize("text", ["", "no", " NO. ", "unknown", "Not visible", "not_visible", "N/A", None, True])
def test_placeholder_actions_are_removed(text):
    result = normalize(response(observed_action=text))
    assert result["decision"] == "uncertain"
    assert result["observed_action"] == ""


@pytest.mark.parametrize("decision", ["uncertain", "rejected"])
def test_never_promotes_existing_uncertainty_or_rejection(decision):
    assert normalize(response(decision=decision))["decision"] == decision


def test_combined_real_failure_cannot_report_support():
    result = normalize({"decision": "supported", "visibility": "unusable",
                        "checks": {"action_visible": "not_visible"}, "evidence_indices": [],
                        "observed_action": "no"})
    assert result["decision"] == "uncertain"
    assert result["observed_action"] == ""
    assert result["confidence"] == 0


@pytest.mark.parametrize("answer", [None, [], {}, {"checks": []}, {"check_frames": []}])
def test_missing_shape_is_safe_and_uncertain(answer):
    result = normalize(answer)
    assert result["decision"] == "uncertain"


@pytest.mark.parametrize("decision", [[], {}, True, 1, None])
def test_malformed_decision_cannot_crash_or_support(decision):
    assert normalize(response(decision=decision))["decision"] == "uncertain"


def test_empty_frame_request_cannot_support():
    assert normalize(response(), count=0)["decision"] == "uncertain"


def test_applicable_exception_rejects_context():
    result = normalize(response(exception_checks={"exception_0": "applies"}, exception_frames={"exception_0": [1]}))
    assert result["decision"] == "rejected"


def test_excluded_exception_requires_valid_evidence():
    valid = response(exception_checks={"exception_0": "excluded"}, exception_frames={"exception_0": [1]})
    assert normalize(valid)["decision"] == "supported"
    valid["exception_frames"]["exception_0"] = []
    assert normalize(valid)["decision"] == "uncertain"


def test_unknown_exception_citations_are_removed():
    result = normalize(response(exception_checks={"exception_0": "unknown"}, exception_frames={"exception_0": [1]}))
    assert result["decision"] == "uncertain"
    assert result["exception_frames"]["exception_0"] == []


def test_empty_exception_registry_does_not_invent_unknown_exceptions():
    assert normalize(response(exception_checks={}, exception_frames={}))["decision"] == "supported"


def test_duplicate_citations_are_deduplicated():
    result = normalize(response(evidence_indices=[0, 0], check_frames={"action_visible": [0, 0]}))
    assert result["decision"] == "supported"
    assert result["evidence_indices"] == result["check_frames"]["action_visible"] == [0]


def test_check_mapping_order_does_not_affect_rejection():
    result = normalize(response(checks={"other_check": "no", "action_visible": "yes"}))
    assert result["decision"] == "rejected"


@pytest.mark.parametrize("count", [-1, True, 1.5, None])
def test_invalid_frame_count_is_a_programming_error(count):
    with pytest.raises(ValueError):
        normalize(response(), count)
