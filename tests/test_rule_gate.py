from copy import deepcopy
import json
from pathlib import Path

import pytest

from worker.rule_gate import exception_fields, exception_prompt, exceptions_excluded, factual_description
from worker.video import FrameRef


RULE = {"label": "Workstation departure during unfinished cash intake",
        "document": "Reviewed local document", "paragraph": "22", "source_url": "https://example.org/rule#22",
        "exceptions": ["Departure after completed service.", "A brief occlusion."]}
FRAMES = [FrameRef(30, 1.0, Path("not-read-one.jpg")), FrameRef(60, 2.0, Path("not-read-two.jpg"))]


def response():
    return {"exception_checks": {"exception_0": "excluded", "exception_1": "excluded"},
            "exception_frames": {"exception_0": [0], "exception_1": [0, 1]}}


def test_exception_fields_are_strict_required_and_request_frame_bounded():
    fields = exception_fields(RULE, 2)
    assert set(fields) == {"exception_checks", "exception_frames"}
    for schema in fields.values():
        assert schema["required"] == ["exception_0", "exception_1"]
        assert schema["additionalProperties"] is False
    assert fields["exception_checks"]["properties"]["exception_0"]["enum"] == ["excluded", "applies", "unknown"]
    evidence = fields["exception_frames"]["properties"]["exception_1"]
    assert evidence["maxItems"] == 3 and evidence["items"]["maximum"] == 1
    json.dumps(fields)


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_exception_schema_requires_real_supplied_frames(count):
    with pytest.raises(ValueError):
        exception_fields(RULE, count)


def test_exception_prompt_preserves_exact_registry_values_as_data():
    prompt = exception_prompt(RULE)
    mapping = json.dumps({"exception_0": RULE["exceptions"][0], "exception_1": RULE["exceptions"][1]}, ensure_ascii=False)
    assert mapping in prompt
    assert "DATA" in prompt and "affirmative visible" in prompt
    assert "unknown" in prompt and "Sparse samples" in prompt


def test_all_exceptions_need_valid_exclusion_evidence_and_input_is_unchanged():
    answer = response()
    original = deepcopy(answer)
    assert exceptions_excluded(answer, RULE, FRAMES)
    assert answer == original


@pytest.mark.parametrize("state", ["unknown", "applies", "yes", "Excluded", True, None])
def test_nonexcluded_exception_never_passes(state):
    answer = response()
    answer["exception_checks"]["exception_1"] = state
    assert not exceptions_excluded(answer, RULE, FRAMES)


@pytest.mark.parametrize("indices", [[], [-1], [2], [True], [False], [0.0], ["0"], None, [0, 0, 0, 0]])
def test_excluded_exception_with_invalid_or_missing_frame_indices_never_passes(indices):
    answer = response()
    answer["exception_frames"]["exception_0"] = indices
    assert not exceptions_excluded(answer, RULE, FRAMES)


@pytest.mark.parametrize("missing", ["exception_checks", "exception_frames"])
def test_missing_exception_schema_never_defaults_to_pass(missing):
    answer = response()
    del answer[missing]
    assert not exceptions_excluded(answer, RULE, FRAMES)


def test_missing_or_extra_exception_keys_fail():
    missing = response()
    del missing["exception_checks"]["exception_1"]
    assert not exceptions_excluded(missing, RULE, FRAMES)
    extra = response()
    extra["exception_checks"]["exception_2"] = "excluded"
    extra["exception_frames"]["exception_2"] = [0]
    assert not exceptions_excluded(extra, RULE, FRAMES)


def test_rules_without_exceptions_require_explicit_empty_objects():
    rule = {**RULE, "exceptions": []}
    fields = exception_fields(rule, 2)
    assert fields["exception_checks"]["properties"] == fields["exception_frames"]["properties"] == {}
    assert exceptions_excluded({"exception_checks": {}, "exception_frames": {}}, rule, FRAMES)
    assert not exceptions_excluded({}, rule, FRAMES)
    assert not exceptions_excluded({"exception_checks": {}}, rule, FRAMES)
    assert not exceptions_excluded({"exception_checks": {}, "exception_frames": {}}, rule, [])


@pytest.mark.parametrize("invalid_exceptions", [None, "No exceptions", [""], [42]])
def test_invalid_registry_exception_configuration_fails_closed(invalid_exceptions):
    rule = {**RULE, "exceptions": invalid_exceptions}
    assert not exceptions_excluded(response(), rule, FRAMES)
    with pytest.raises(ValueError):
        exception_fields(rule, 2)
    with pytest.raises(ValueError):
        exception_prompt(rule)
    assert not exceptions_excluded(response(), {"label": "Missing exceptions"}, FRAMES)


@pytest.mark.parametrize("frame", [FrameRef(True, 1, Path("unused.jpg")),
                                  FrameRef(1, float("nan"), Path("unused.jpg")),
                                  FrameRef(1, -1, Path("unused.jpg")), object()])
def test_invalid_referenced_source_frame_cannot_support_exclusion(frame):
    assert not exceptions_excluded(response(), RULE, [frame, FRAMES[1]])


def test_description_uses_trusted_registry_text_and_time_only():
    rule = {**RULE, "model_description": "Unsupported model prose", "observed_action": "Invented event"}
    description = factual_description(rule, 1.25)
    assert description == ("Possible violation: Workstation departure during unfinished cash intake. Event at 1.250s. "
                           "Source: Reviewed local document, paragraph 22. https://example.org/rule#22")
    assert "Unsupported" not in description and "Invented" not in description


@pytest.mark.parametrize("timestamp", [True, -1, float("nan"), float("inf"), "1"])
def test_factual_description_rejects_invalid_source_time(timestamp):
    with pytest.raises(ValueError):
        factual_description(RULE, timestamp)


def test_factual_description_requires_trusted_source_information():
    rule = deepcopy(RULE)
    del rule["paragraph"]
    with pytest.raises(ValueError, match="paragraph"):
        factual_description(rule, 1)
