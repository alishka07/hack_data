"""Synthetic wiring tests, not an accuracy evaluation of eating recognition.

All visual decisions below are injected. The gray clip only verifies that host
checks, policy provenance, source timestamps, and genuine evidence JPEGs survive
the complete pipeline without needing model weights or private footage.
"""
import base64
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import pytest

from worker.pipeline import Pipeline, load_rules, supported_verification, verification_schema
from worker.video import FrameRef


ROOT = Path(__file__).resolve().parents[1]
RULE_ID = "cashier_eating_at_workstation"
CHECKS = (
    "food_consumption_visibly_observed",
    "person_operating_cashier_workstation",
    "workstation_work_context_visible",
    "relevant_context_visible",
)
POSITIVE_ACTION = "The cashier visibly takes a bite of food while operating the cashier workstation."


@pytest.fixture(autouse=True)
def isolate_optional_models(monkeypatch):
    monkeypatch.setenv("FORTE_POSE_BACKEND", "disabled")
    monkeypatch.setenv("FORTE_OBJECT_BACKEND", "disabled")


@pytest.fixture(scope="module")
def synthetic_eating_source(tmp_path_factory):
    path = tmp_path_factory.mktemp("eating-source") / "synthetic.mov"
    subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
        "color=c=gray:s=160x240:r=30:d=2", "-an", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", "-y", str(path),
    ], check=True)
    return path


class EatingRuntime:
    """Always proposes eating; each test controls its verifier evidence."""

    def __init__(self, *, observed_action=POSITIVE_ACTION, statuses=None,
                 missing_frame_check=None, evidence=True, food_indices=None):
        self.observed_action = observed_action
        self.statuses = statuses or {}
        self.missing_frame_check = missing_frame_check
        self.evidence = evidence
        self.food_indices = [0, 3] if food_indices is None else food_indices

    def ask(self, prompt, frames, schema, budget, max_tokens=512):
        properties = schema["properties"]
        if "candidates" in properties:
            # Keep proposal wording positive so tests of contrary verifier prose
            # actually reach the host verification gate, not just screening.
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
        names = properties["checks"]["properties"]
        response = {
            "decision": "supported", "confidence": 1., "visibility": "clear",
            "observed_action": self.observed_action,
            "checks": {name: self.statuses.get(name, "yes") for name in names},
            "check_frames": {name: [0] for name in names},
            "evidence_indices": [0, 3, 7] if self.evidence else [],
            "limitations": [],
        }
        if CHECKS[0] in names:
            response["check_frames"][CHECKS[0]] = self.food_indices
        if self.missing_frame_check in names:
            # A yes without source citations must fail closed, including after
            # the pipeline's optional refinement request.
            response["check_frames"].pop(self.missing_frame_check)
        if "exception_checks" in properties:
            exception_names = properties["exception_checks"]["properties"]
            response["exception_checks"] = {name: "excluded" for name in exception_names}
            response["exception_frames"] = {name: [0] for name in exception_names}
        return response


def analyze(source, directory, runtime):
    return Pipeline(ROOT, directory, runtime).run(
        {"id": "eating-test", "video_path": str(source), "deadline_at": time.time() + 60},
        lambda *_: None,
    )


def assert_no_eating_violation(result, details):
    assert result == {"verdict": "no_violation", "violations": []}
    assert not any(item["type_id"] == RULE_ID for item in details["findings"])
    assert not any(case["type_id"] == RULE_ID and case["status"] == "confirmed"
                   for case in details["case_decision"]["cases"])


def test_eating_is_a_workplace_policy_violation_with_source_evidence(synthetic_eating_source, tmp_path):
    rule = load_rules(ROOT)[RULE_ID]
    assert rule["basis"] == "workplace_policy"
    assert set(rule["required_checks"]) == set(CHECKS)
    assert set(rule["action_checks"]) == set(CHECKS[:2])
    result, details, timings = analyze(synthetic_eating_source, tmp_path, EatingRuntime())
    assert result["verdict"] == "violation"
    assert len(result["violations"]) == 1
    finding = result["violations"][0]
    assert finding["type"] == RULE_ID
    assert 0 <= finding["timecode_sec"] <= 2
    assert finding["timecode_sec"] in [frame["timestamp_sec"] for frame in finding["evidence_frames"]]
    assert 1 <= len(finding["evidence_frames"]) <= 3
    for evidence in finding["evidence_frames"]:
        jpeg = base64.b64decode(evidence["image_base64"], validate=True)
        assert jpeg.startswith(b"\xff\xd8")
        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        assert image is not None and image.shape[:2] == (240, 160)
        assert 0 <= evidence["timestamp_sec"] <= 2
    detail = details["findings"][0]
    assert detail["basis"] == "workplace_policy"
    assert detail["rule_url"] == "/policies/workplace-eating"
    assert "workplace" in detail["rule_name"].casefold()
    assert "/policies/workplace-eating" in finding["description"]
    assert not any(text in finding["description"].casefold() for text in ("national bank", "nbk", "№78"))
    assert details["case_decision"]["status"] == "confirmed_violation"
    assert timings["action_calls"] == 1 and timings["context_calls"] == 1


@pytest.mark.parametrize("observed_action", [
    "Food remains on the cashier table; no eating is visible.",
    "The cashier drinks from a cup at the work table.",
    "The cashier touches their face with an empty hand at the work table.",
    "The cashier takes medication at the work table.",
    "The cashier does not eat at the work table.",
])
def test_non_eating_action_cannot_pass_even_with_model_yes_checks(
        synthetic_eating_source, tmp_path, observed_action):
    result, details, _ = analyze(synthetic_eating_source, tmp_path,
                                 EatingRuntime(observed_action=observed_action))
    assert_no_eating_violation(result, details)


@pytest.mark.parametrize("check", CHECKS)
@pytest.mark.parametrize("status", ["no", "unknown", "not_visible"])
def test_every_eating_policy_predicate_requires_affirmative_evidence(
        synthetic_eating_source, tmp_path, check, status):
    result, details, _ = analyze(synthetic_eating_source, tmp_path,
                                 EatingRuntime(statuses={check: status}))
    assert_no_eating_violation(result, details)


@pytest.mark.parametrize("check", CHECKS)
def test_eating_yes_check_without_source_citations_is_insufficient(
        synthetic_eating_source, tmp_path, check):
    result, details, _ = analyze(synthetic_eating_source, tmp_path,
                                 EatingRuntime(missing_frame_check=check))
    assert_no_eating_violation(result, details)


def test_eating_without_any_evidence_frames_is_not_a_violation(synthetic_eating_source, tmp_path):
    result, details, _ = analyze(synthetic_eating_source, tmp_path, EatingRuntime(evidence=False))
    assert_no_eating_violation(result, details)


def test_one_food_source_frame_is_not_an_eating_sequence(synthetic_eating_source, tmp_path):
    result, details, _ = analyze(synthetic_eating_source, tmp_path, EatingRuntime(food_indices=[0]))
    assert_no_eating_violation(result, details)


def test_two_views_of_the_same_source_frame_are_not_an_eating_sequence(tmp_path):
    rule = load_rules(ROOT)[RULE_ID]
    # The full frame and its crop are two model inputs but only one source
    # observation. Repeating that source cannot establish food consumption.
    frames = [FrameRef(10, .5, tmp_path / "source.jpg"),
              FrameRef(10, .5, tmp_path / "crop.jpg", "counter detail", (.1, .1, .8, .8)),
              FrameRef(20, 1., tmp_path / "other.jpg")]
    runtime = EatingRuntime(food_indices=[0, 1])
    answer = runtime.ask("", frames, verification_schema(rule, len(frames)), None)
    answer["evidence_indices"] = [0, 1]
    assert not supported_verification(answer, rule, frames)
