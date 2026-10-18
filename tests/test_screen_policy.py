"""Synthetic screen-policy wiring, not an accuracy claim for computer vision.

Injected answers exercise source-order checks, policy provenance, and evidence
JPEGs without reading private clips or loading optional perception models.
"""
import base64
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import pytest

from worker.pipeline import Pipeline, load_rules, supported_verification, verification_schema
from worker.video import FrameRef


ROOT = Path(__file__).resolve().parents[1]
RULE_ID = "workstation_departure_screen_left_on"
CHECKS = (
    "computer_at_workstation_identified",
    "workstation_departure_observed",
    "workstation_unattended_after_departure",
    "same_screen_visibly_on_after_departure",
    "departure_screen_sequence_visible",
)
POSITIVE_ACTION = (
    "The cashier leaves the identified computer workstation; later the same "
    "workstation is visibly unattended and its computer screen remains on."
)


@pytest.fixture(autouse=True)
def isolate_optional_models(monkeypatch):
    monkeypatch.setenv("FORTE_PERCEPTION_BACKEND", "motion_only")
    monkeypatch.setenv("FORTE_TRACKER", "adjacent_iou")
    monkeypatch.delenv("FORTE_PERSON_MODEL", raising=False)
    monkeypatch.delenv("FORTE_PERSON_MANIFEST", raising=False)
    monkeypatch.setenv("FORTE_POSE_BACKEND", "disabled")
    monkeypatch.setenv("FORTE_OBJECT_BACKEND", "disabled")


@pytest.fixture(scope="module")
def synthetic_screen_source(tmp_path_factory):
    source = tmp_path_factory.mktemp("screen-source") / "synthetic.mov"
    subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
        "color=c=gray:s=160x240:r=30:d=2", "-an", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", "-y", str(source),
    ], check=True)
    return source


class ScreenRuntime:
    def __init__(self, *, statuses=None, missing_frame_check=None, evidence=True,
                 references=None, evidence_indices=None, observed_action=POSITIVE_ACTION):
        self.statuses = statuses or {}
        self.missing_frame_check = missing_frame_check
        self.evidence = evidence
        self.references = references or {}
        self.evidence_indices = evidence_indices
        self.observed_action = observed_action

    def ask(self, prompt, frames, schema, budget, max_tokens=512):
        properties = schema["properties"]
        if "candidates" in properties:
            response = {
                "visibility": "clear", "visible_facts": [],
                "transaction": {"kind": "unknown", "phase": "unknown", "frame_indices": []},
                "candidates": [{"type_id": RULE_ID, "anchor_frame": 3,
                                "start_frame": 0, "end_frame": 7,
                                "observed_action": POSITIVE_ACTION}],
            }
            if "scene_regions" in properties:
                response["scene_regions"] = []
            if "scenario_observations" in properties:
                response["scenario_observations"] = []
            return response
        first = min(range(len(frames)), key=lambda i: frames[i].timestamp)
        last = max(range(len(frames)), key=lambda i: frames[i].timestamp)
        names = properties["checks"]["properties"]
        defaults = {
            CHECKS[0]: [first, last], CHECKS[1]: [first, last],
            CHECKS[2]: [last], CHECKS[3]: [last], CHECKS[4]: [first, last],
        }
        response = {
            "decision": "supported", "confidence": 1., "visibility": "clear",
            "observed_action": self.observed_action,
            "checks": {name: self.statuses.get(name, "yes") for name in names},
            "check_frames": {name: self.references.get(name, defaults[name]) for name in names},
            "evidence_indices": (([last] if self.evidence_indices is None else self.evidence_indices)
                                 if self.evidence else []),
            "limitations": [],
        }
        if self.missing_frame_check in names:
            response["check_frames"].pop(self.missing_frame_check)
        if "exception_checks" in properties:
            exceptions = properties["exception_checks"]["properties"]
            response["exception_checks"] = {name: "excluded" for name in exceptions}
            response["exception_frames"] = {name: [first, last] for name in exceptions}
        return response


def analyze(source, directory, runtime):
    return Pipeline(ROOT, directory, runtime).run(
        {"id": "screen-policy-test", "video_path": str(source), "deadline_at": time.time() + 60},
        lambda *_: None,
    )


def answer_for(frames, *, action_only=False, **kwargs):
    rule = load_rules(ROOT)[RULE_ID]
    if action_only:
        rule = {**rule, "required_checks": rule["action_checks"]}
    answer = ScreenRuntime(**kwargs).ask("", frames, verification_schema(rule, len(frames)), None)
    return answer, rule


def source_frames(tmp_path):
    return [FrameRef(10, .5, tmp_path / "before.jpg"),
            FrameRef(20, 1., tmp_path / "after.jpg"),
            FrameRef(30, 1.5, tmp_path / "unrelated.jpg")]


def test_screen_policy_produces_post_departure_source_jpegs(synthetic_screen_source, tmp_path):
    rule = load_rules(ROOT)[RULE_ID]
    assert rule["basis"] == "workplace_policy"
    assert rule["auto_emit"] is True
    assert set(rule["required_checks"]) == set(CHECKS)
    assert set(rule["action_checks"]) == set(CHECKS[:4])
    result, details, timings = analyze(synthetic_screen_source, tmp_path, ScreenRuntime())
    assert result["verdict"] == "violation"
    assert len(result["violations"]) == 1
    finding = result["violations"][0]
    assert finding["type"] == RULE_ID
    assert 0 < finding["timecode_sec"] <= 2
    attempts = json.loads((tmp_path / "jobs" / "screen-policy-test" / "verification.json").read_text())
    attempt = next(item for item in attempts if item["type_id"] == RULE_ID)
    departure_end = max(attempt["action_frames"][index]["timestamp_sec"]
                        for index in attempt["action"]["check_frames"][CHECKS[1]])
    assert finding["timecode_sec"] >= departure_end
    assert finding["timecode_sec"] in [frame["timestamp_sec"] for frame in finding["evidence_frames"]]
    assert 1 <= len(finding["evidence_frames"]) <= 3
    assert any(frame["timestamp_sec"] < departure_end for frame in finding["evidence_frames"])
    for frame in finding["evidence_frames"]:
        jpeg = base64.b64decode(frame["image_base64"], validate=True)
        assert jpeg.startswith(b"\xff\xd8")
        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        assert image is not None and image.shape[:2] == (240, 160)
        assert 0 <= frame["timestamp_sec"] <= 2
    detail = details["findings"][0]
    assert detail["basis"] == "workplace_policy"
    assert detail["rule_url"] == "/policies/workplace-screen"
    assert "/policies/workplace-screen" in finding["description"]
    assert not any(word in finding["description"].casefold() for word in ("national bank", "nbk", "№78"))
    assert details["case_decision"]["status"] == "confirmed_violation"
    assert timings["action_calls"] == 1 and timings["context_calls"] == 1


@pytest.mark.parametrize("check", CHECKS)
@pytest.mark.parametrize("status", ["no", "unknown", "not_visible"])
def test_every_screen_predicate_requires_positive_evidence(tmp_path, check, status):
    frames = source_frames(tmp_path)
    answer, rule = answer_for(frames, statuses={check: status})
    assert not supported_verification(answer, rule, frames)


@pytest.mark.parametrize("check", CHECKS)
def test_screen_check_without_source_citations_cannot_pass(tmp_path, check):
    frames = source_frames(tmp_path)
    answer, rule = answer_for(frames, missing_frame_check=check)
    assert not supported_verification(answer, rule, frames)


@pytest.mark.parametrize("action_only", [False, True])
def test_one_snapshot_cannot_establish_a_departure(tmp_path, action_only):
    frames = [FrameRef(10, .5, tmp_path / "empty-chair-bright-screen.jpg")]
    answer, rule = answer_for(frames, action_only=action_only)
    assert not supported_verification(answer, rule, frames)


def test_distinct_crops_of_one_source_are_not_a_departure_sequence(tmp_path):
    frames = [FrameRef(10, .5, tmp_path / "full.jpg"),
              FrameRef(10, .5, tmp_path / "crop.jpg", "computer detail", (.1, .1, .8, .8))]
    answer, rule = answer_for(frames)
    assert not supported_verification(answer, rule, frames)


def test_later_screen_and_unattended_citations_establish_sequence(tmp_path):
    frames = source_frames(tmp_path)
    answer, rule = answer_for(frames)
    assert supported_verification(answer, rule, frames)


def test_supplied_frame_order_cannot_override_source_timestamps(tmp_path):
    frames = list(reversed(source_frames(tmp_path)))
    answer, rule = answer_for(frames)
    assert supported_verification(answer, rule, frames)


def test_screen_seen_only_before_departure_cannot_pass(tmp_path):
    frames = source_frames(tmp_path)
    answer, rule = answer_for(frames, references={CHECKS[1]: [1, 2], CHECKS[3]: [0]})
    assert not supported_verification(answer, rule, frames)


def test_screen_after_initial_presence_but_before_exit_cannot_pass(tmp_path):
    frames = source_frames(tmp_path)
    answer, rule = answer_for(frames, references={CHECKS[0]: [1], CHECKS[2]: [1], CHECKS[3]: [1]},
                              evidence_indices=[1])
    assert not supported_verification(answer, rule, frames)


def test_identified_computer_must_be_linked_to_the_post_departure_screen(tmp_path):
    frames = source_frames(tmp_path)
    answer, rule = answer_for(frames, references={CHECKS[0]: [0]})
    assert not supported_verification(answer, rule, frames)


@pytest.mark.parametrize("indices", [[0], [2]])
def test_context_sequence_citation_must_include_both_sides(tmp_path, indices):
    frames = source_frames(tmp_path)
    answer, rule = answer_for(frames, references={CHECKS[4]: indices})
    assert not supported_verification(answer, rule, frames)


def test_unattended_and_on_screen_must_share_a_post_departure_source(tmp_path):
    frames = source_frames(tmp_path)
    answer, rule = answer_for(frames, references={CHECKS[2]: [1], CHECKS[3]: [2]})
    assert not supported_verification(answer, rule, frames)


def test_primary_evidence_must_include_the_post_departure_screen(tmp_path):
    frames = source_frames(tmp_path)
    answer, rule = answer_for(frames, evidence_indices=[0])
    assert not supported_verification(answer, rule, frames)


def test_missing_primary_evidence_cannot_pass(tmp_path):
    frames = source_frames(tmp_path)
    answer, rule = answer_for(frames, evidence=False)
    assert not supported_verification(answer, rule, frames)


@pytest.mark.parametrize("action", [
    "The computer screen is off after the cashier leaves the workstation.",
    "An empty chair and a bright screen are visible, but no departure is observed.",
    "The person stands and reaches while remaining at the workstation.",
    "The cashier is cropped out; departure and unattended status are not visible.",
    "The display only reflects a window; whether the screen is on is unknown.",
])
def test_contradictory_screen_description_cannot_override_yes_flags(tmp_path, action):
    frames = source_frames(tmp_path)
    answer, rule = answer_for(frames, observed_action=action)
    assert not supported_verification(answer, rule, frames)


def test_visible_locked_but_lit_screen_still_qualifies(tmp_path):
    frames = source_frames(tmp_path)
    action = ("The cashier leaves the computer workstation, which is visibly unattended "
              "afterwards; the same computer screen remains visibly on at a lock screen.")
    answer, rule = answer_for(frames, observed_action=action)
    assert supported_verification(answer, rule, frames)


@pytest.mark.parametrize("check", [CHECKS[1], CHECKS[3], CHECKS[4]])
def test_off_screen_or_unobserved_departure_never_emits_in_pipeline(
        synthetic_screen_source, tmp_path, check):
    result, details, _ = analyze(synthetic_screen_source, tmp_path,
                                 ScreenRuntime(statuses={check: "not_visible"}))
    assert result == {"verdict": "no_violation", "violations": []}
    assert not any(item["type_id"] == RULE_ID for item in details["findings"])
    assert not any(case["type_id"] == RULE_ID and case["status"] == "confirmed"
                   for case in details["case_decision"]["cases"])
