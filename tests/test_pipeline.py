"""Synthetic media tests; no private bank footage or model descriptions are used."""
import json
from pathlib import Path
import subprocess
import time

import cv2
import pytest

from worker.pipeline import (Candidate, Pipeline, anchored_targets, deduplicate_candidates, source_gap_intersects,
                             deduplicate_findings, load_rules, supported_verification)
from worker.runtime import LocalRuntime
from worker.video import AnalysisError, Budget, FrameCache, FrameRef, VideoInfo, extract_frames, nearest_index, probe_video

ROOT = Path(__file__).resolve().parents[1]


def test_source_gap_at_context_boundary_is_not_discarded():
    pts = (0., 1., 2., 10., 11., 12., 13., 14.)
    assert source_gap_intersects(pts, 3., 14.)
    assert source_gap_intersects(pts, 1., 5.)
    assert not source_gap_intersects(pts, 10., 14.)
    assert not source_gap_intersects(pts, 0., 2.)


@pytest.fixture(autouse=True)
def isolate_optional_model_assets(monkeypatch):
    """Pipeline contracts use synthetic media, independently of installed weights."""
    monkeypatch.setenv("FORTE_POSE_BACKEND", "disabled")
    monkeypatch.setenv("FORTE_OBJECT_BACKEND", "disabled")


@pytest.fixture
def synthetic_video(tmp_path):
    path = tmp_path / "synthetic.mov"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", "color=c=gray:s=160x240:r=30:d=2",
                    "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-y", str(path)], check=True)
    return path


class FakeRuntime:
    def __init__(self, unknown=False, fail=False):
        self.unknown, self.fail = unknown, fail

    def ask(self, prompt, frames, schema, budget, max_tokens=512):
        if self.fail:
            raise AnalysisError("model_request_failed", "Synthetic model error")
        if "candidates" in schema["properties"]:
            ids = schema["properties"]["candidates"]["items"]["properties"]["type_id"]["enum"]
            response = {"visibility": "clear", "visible_facts": [{"kind": "cash_received", "state": "observed", "frame_indices": [0], "note": "Synthetic test only"}],
                    "transaction": {"kind": "intake", "phase": "active", "frame_indices": [0]}, "scenario_observations": [], "candidates": [
                {"type_id": ids[0], "anchor_frame": 2, "start_frame": 0, "end_frame": 4, "observed_action": "Synthetic test action"},
                {"type_id": ids[1], "anchor_frame": 5, "start_frame": 3, "end_frame": 7, "observed_action": "Synthetic second action"}]}
            if "scene_regions" in schema["properties"]:
                response["scene_regions"] = []
            return response
        response = {"decision": "supported", "confidence": 1., "visibility": "clear", "observed_action": "Synthetic test action",
                "checks": {name: "unknown" if self.unknown else "yes" for name in schema["properties"]["checks"]["properties"]},
                "check_frames": {name: [0] for name in schema["properties"]["checks"]["properties"]},
                "evidence_indices": [0, 3, 7], "limitations": []}
        if "exception_checks" in schema["properties"]:
            names = schema["properties"]["exception_checks"]["properties"]
            response.update(exception_checks={key: "unknown" if self.unknown else "excluded" for key in names},
                            exception_frames={key: [] if self.unknown else [0] for key in names})
        return response


def test_pipeline_multiple_events_real_jpeg_and_timestamps(synthetic_video, tmp_path):
    job = {"id": "test-job", "video_path": str(synthetic_video), "deadline_at": time.time() + 60}
    stages = []
    result, details, timings = Pipeline(ROOT, tmp_path, FakeRuntime()).run(job, lambda stage, *_: stages.append(stage))
    assert result["verdict"] == "violation"
    assert len(result["violations"]) == 2
    assert details["coverage"]["windows_completed"] == details["coverage"]["windows_planned"]
    assert timings["finding_count"] == 2
    assert len(details["assessments"]) == 33
    observed = [row for row in details["assessments"] if row["confirmed_finding"]]
    assert len(observed) == 2 and all(row["score"] == 100 for row in observed)
    assert len(details["assessment_episodes"]) == 2
    procedures = [row for row in details["assessments"] if row["scoring_kind"] == "required_procedure"]
    assert all(row["score"] == 100 and row["assessment"] == "unassessed" for row in procedures)
    import base64
    import numpy as np
    for finding in result["violations"]:
        assert set(finding) == {"type", "timecode_sec", "description", "evidence_frames"}
        assert 1 <= len(finding["evidence_frames"]) <= 3
        assert finding["timecode_sec"] in [frame["timestamp_sec"] for frame in finding["evidence_frames"]]
        for evidence in finding["evidence_frames"]:
            assert 0 <= evidence["timestamp_sec"] <= 2
            jpeg = base64.b64decode(evidence["image_base64"], validate=True)
            assert jpeg.startswith(b"\xff\xd8")
            assert cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR) is not None
    assert "finalizing" in stages


def test_scene_detail_crops_respect_eight_image_verifier_budget(synthetic_video, tmp_path):
    class BoundedRuntime(FakeRuntime):
        def ask(self, prompt, frames, schema, budget, max_tokens=512):
            assert len(frames) <= 8
            answer = super().ask(prompt, frames, schema, budget, max_tokens)
            if "scene_regions" in schema["properties"]:
                answer["scene_regions"] = [{"zone": "counter", "status": "supported", "bbox": [.2, .2, .8, .8], "frame_indices": [0]}]
            return answer
    result, details, _ = Pipeline(ROOT, tmp_path, BoundedRuntime()).run(
        {"id": "bounded", "video_path": str(synthetic_video), "deadline_at": time.time() + 60}, lambda *_: None)
    assert result["verdict"] == "violation"
    assert details["scene"]["zones"][0]["status"] == "supported"
    attempts = json.loads((tmp_path / "jobs/bounded/verification.json").read_text())
    assert any(frame["crop_bbox"] is not None for attempt in attempts for frame in attempt["action_frames"])
    assert all(len(attempt["action_frames"]) <= 8 and len(attempt["context_frames"]) <= 8 for attempt in attempts)


def test_unknown_check_cannot_publish_finding(synthetic_video, tmp_path):
    result, details, _ = Pipeline(ROOT, tmp_path, FakeRuntime(unknown=True)).run(
        {"id": "unknown", "video_path": str(synthetic_video), "deadline_at": time.time() + 60}, lambda *_: None)
    assert result == {"verdict": "no_violation", "violations": []}
    assert details["review_queue"] == []
    assert len(details["dismissed_candidates"]) == 2
    assert details["coverage"]["unresolved_verification_count"] == 2
    assert all(item["disposition"] == "unconfirmed" for item in details["candidate_events"])
    assert not any(row["confirmed_finding"] for row in details["assessments"])
    assert all(row["score"] == row["initial_score"] for row in details["assessments"])
    assert details["case_decision"]["status"] == "review_required"
    assert details["case_decision"]["counts"]["confirmed"] == 0
    assert any(check["status"] == "unknown" and not check["evidence_refs"]
               for case in details["case_decision"]["cases"] for check in case["checks"])


def test_verification_budget_preserves_result_and_marks_unfinished_checks(synthetic_video, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from worker import pipeline as module
    class LimitedRuntime(FakeRuntime):
        calls = 0
        def ask(self, prompt, frames, schema, budget, max_tokens=512):
            if "checks" in schema["properties"]:
                self.calls += 1
                assert budget.deadline == deadline - module.VERIFICATION_FINISH_RESERVE_SEC
            return super().ask(prompt, frames, schema, budget, max_tokens)
    runtime = LimitedRuntime()
    deadline = time.time() + 60
    monkeypatch.setattr(module, "verification_admission", lambda *args, **kwargs:
                        SimpleNamespace(allowed=runtime.calls < 1, reason="verification_budget_exhausted"))
    result, details, timings = Pipeline(ROOT, tmp_path, runtime).run(
        {"id": "limited", "video_path": str(synthetic_video), "deadline_at": deadline}, lambda *_: None)
    assert runtime.calls == 1
    assert result["violations"] == []
    assert details["processing_status"] == "review_incomplete"
    assert details["coverage"]["windows_completed"] == details["coverage"]["windows_planned"]
    assert details["coverage"]["verification_complete"] is False
    assert details["coverage"]["unverified_candidate_count"] == 2
    assert timings["context_calls"] == 0
    assert details["case_decision"]["status"] == "review_required"
    assert "verification_incomplete" in details["case_decision"]["reason_codes"]


def test_unusable_views_remain_unknown_without_repeating_optional_refinements(synthetic_video, tmp_path):
    class UnusableRuntime(FakeRuntime):
        def ask(self, prompt, frames, schema, budget, max_tokens=512):
            answer = super().ask(prompt, frames, schema, budget, max_tokens)
            if "checks" in schema["properties"]:
                answer.update(visibility="unusable", checks={name: "not_visible" for name in answer["checks"]})
            return answer
    result, details, timings = Pipeline(ROOT, tmp_path, UnusableRuntime()).run(
        {"id": "hidden", "video_path": str(synthetic_video), "deadline_at": time.time() + 60}, lambda *_: None)
    assert result["violations"] == []
    assert timings["action_calls"] == 2 and timings["refinement_passes"] == 0
    assert details["coverage"]["unresolved_verification_count"] == 2
    assert details["case_decision"]["status"] == "review_required"


@pytest.mark.parametrize("code", ["model_timeout", "model_output_invalid", "model_transport_error"])
def test_late_verifier_failure_keeps_verified_finding_and_explicit_gap(synthetic_video, tmp_path, code):
    class InterruptedRuntime(FakeRuntime):
        calls = 0
        def ask(self, prompt, frames, schema, budget, max_tokens=512):
            if "checks" in schema["properties"]:
                self.calls += 1
                if self.calls == 3:
                    raise AnalysisError(code, "Synthetic late verification failure")
            return super().ask(prompt, frames, schema, budget, max_tokens)
    result, details, timings = Pipeline(ROOT, tmp_path, InterruptedRuntime()).run(
        {"id": "interrupted", "video_path": str(synthetic_video), "deadline_at": time.time() + 60}, lambda *_: None)
    assert result["verdict"] == "violation" and len(result["violations"]) == 1
    assert details["processing_status"] == "review_incomplete"
    assert details["coverage"]["unverified_candidate_count"] == 1
    assert details["unverified_candidates"][0]["reason"] == code
    assert details["unverified_candidates"][0]["verified"] is False
    assert timings["unverified_candidate_count"] == 1
    assert len(json.loads((tmp_path / "jobs/interrupted/verification.json").read_text())) == 2


@pytest.mark.parametrize("identifier", ["keys_stamps_secure_handling", "personal_belongings_in_cash_unit",
                                        "backoffice_processing_control", "vault_cash_segregation_storage",
                                        "previous_cash_stored", "loose_cash_issued_without_observed_count"])
def test_manual_scenarios_are_verified_with_original_evidence(synthetic_video, tmp_path, identifier):
    class ScenarioRuntime(FakeRuntime):
        def ask(self, prompt, frames, schema, budget, max_tokens=512):
            answer = super().ask(prompt, frames, schema, budget, max_tokens)
            if "candidates" in schema["properties"]:
                answer.update(candidates=[], visible_facts=[], scenario_observations=[{
                    "item_id": identifier, "state": "observed", "confidence": .85,
                    "anchor_frame": 3, "start_frame": 0, "end_frame": 7,
                    "visible_evidence": "Synthetic visible scenario for source-reference validation only"}])
            return answer
    result, details, timings = Pipeline(ROOT, tmp_path, ScenarioRuntime()).run(
        {"id": "scenario", "video_path": str(synthetic_video), "deadline_at": time.time() + 60}, lambda *_: None)
    assert result == {"verdict": "no_violation", "violations": []}
    assert timings["action_calls"] == 1 and timings["context_calls"] == 0
    assert details["scenario_coverage"][identifier]["supported_visual_episodes"] == 1
    queued = next(item for item in details["review_queue"] if item["type_id"] == identifier)
    assert queued["visual_scenario_supported"]
    assert "visible_scenario_supported" in queued["reason_codes"]
    assert 1 <= len(queued["evidence_frames"]) <= 3
    declared = {item["filename"] for item in details["evidence"]}
    for frame in queued["evidence_frames"]:
        assert frame["filename"] in declared
        image = cv2.imread(str(tmp_path / "evidence/scenario" / frame["filename"]))
        assert image.shape[:2] == (240, 160)
    assert details["case_decision"]["status"] == "review_required"


def test_unknown_manual_scenario_is_never_reported_as_visually_supported(synthetic_video, tmp_path):
    class UnknownScenarioRuntime(FakeRuntime):
        def ask(self, prompt, frames, schema, budget, max_tokens=512):
            answer = super().ask(prompt, frames, schema, budget, max_tokens)
            if "candidates" in schema["properties"]:
                answer.update(candidates=[], visible_facts=[], scenario_observations=[{
                    "item_id": "keys_stamps_secure_handling", "state": "uncertain", "confidence": .99,
                    "anchor_frame": 3, "start_frame": 0, "end_frame": 7,
                    "visible_evidence": "Synthetic ambiguous tiny item"}])
            return answer
    _, details, _ = Pipeline(ROOT, tmp_path, UnknownScenarioRuntime(unknown=True)).run(
        {"id": "unknown-scenario", "video_path": str(synthetic_video), "deadline_at": time.time() + 60}, lambda *_: None)
    assert details["scenario_coverage"]["keys_stamps_secure_handling"]["supported_visual_episodes"] == 0
    assert all(not item["visual_scenario_supported"] for item in details["review_queue"])


def test_detected_items_have_source_images_and_do_not_create_violations(synthetic_video, tmp_path, monkeypatch):
    from worker import pipeline as pipeline_module
    original_perception = pipeline_module.analyze_perception
    def injected_perception(info, cache, budget, directory):
        report = original_perception(info, cache, budget, directory)
        report["detections"] = [
            {"id": "person-test", "source_index": 0, "timestamp_sec": info.pts[0], "kind": "person",
             "label": "Visible person", "confidence": .9, "bbox": [.1, .1, .45, .9]},
            {"id": "document-test", "source_index": 0, "timestamp_sec": info.pts[0], "kind": "document",
             "label": "Document", "confidence": .7, "bbox": [.5, .6, .7, .8]},
            {"id": "cash-test", "source_index": len(info.pts)-1, "timestamp_sec": info.pts[-1], "kind": "cash",
             "label": "Visible cash", "confidence": .6, "bbox": [.6, .3, .8, .5]},
        ]
        return report
    monkeypatch.setattr(pipeline_module, "analyze_perception", injected_perception)
    class ObjectRuntime(FakeRuntime):
        def ask(self, prompt, frames, schema, budget, max_tokens=512):
            answer = super().ask(prompt, frames, schema, budget, max_tokens)
            answer.update(candidates=[], visible_facts=[])
            from worker.runtime import _validate_json_schema
            _validate_json_schema(answer, schema)
            return answer

    result, details, timings = Pipeline(ROOT, tmp_path, ObjectRuntime()).run(
        {"id": "objects", "video_path": str(synthetic_video), "deadline_at": time.time() + 60}, lambda *_: None)
    assert result == {"verdict": "no_violation", "violations": []}
    assert details["case_decision"]["status"] == "no_supported_violation"
    assert timings["detection_count"] == len(details["detections"]) == 3
    assert details["detection_summary"]["source_frames"] == 2
    declared = {entry["filename"] for entry in details["evidence"]}
    info = probe_video(synthetic_video, Budget(time.time() + 30))
    for item in details["detections"]:
        evidence = item["evidence_frame"]
        assert item["timestamp_sec"] == evidence["timestamp_sec"] == info.pts[item["source_index"]]
        assert evidence["filename"] in declared
        assert cv2.imread(str(tmp_path / "evidence/objects" / evidence["filename"])) is not None
    assert details["detections"][0]["evidence_frame"] == details["detections"][1]["evidence_frame"]


def test_uncertain_context_gets_targeted_refinement(synthetic_video, tmp_path):
    class RefinedRuntime(FakeRuntime):
        def __init__(self):
            super().__init__()
            self.context_calls = 0
            self.call_order = []
        def ask(self, prompt, frames, schema, budget, max_tokens=512):
            answer = super().ask(prompt, frames, schema, budget, max_tokens)
            if "checks" in answer:
                if len(answer["checks"]) > 3:
                    self.context_calls += 1
                    self.call_order.append("context")
                    if self.context_calls % 2:
                        answer["checks"][next(iter(answer["checks"]))] = "unknown"
                        answer["decision"] = "uncertain"
                else:
                    self.call_order.append("action")
            return answer
    runtime = RefinedRuntime()
    result, details, timings = Pipeline(ROOT, tmp_path, runtime).run(
        {"id": "refinement", "video_path": str(synthetic_video), "deadline_at": time.time() + 60}, lambda *_: None)
    assert len(result["violations"]) == 2
    assert timings["refinement_passes"] == 2
    assert runtime.call_order[0] == "action"
    assert details["versions"]["pipeline"] == "5.2"


def test_action_uncertainty_is_refined_before_context_rejection(synthetic_video, tmp_path):
    class ActionRuntime(FakeRuntime):
        def __init__(self):
            super().__init__()
            self.action_count = 0
        def ask(self, prompt, frames, schema, budget, max_tokens=512):
            answer = super().ask(prompt, frames, schema, budget, max_tokens)
            if "checks" in answer and len(answer["checks"]) <= 3:
                self.action_count += 1
                if self.action_count % 2:
                    answer["decision"] = "uncertain"
                    answer["checks"][next(iter(answer["checks"]))] = "unknown"
            return answer
    result, _, timings = Pipeline(ROOT, tmp_path, ActionRuntime()).run(
        {"id": "action-refine", "video_path": str(synthetic_video), "deadline_at": time.time() + 60}, lambda *_: None)
    assert len(result["violations"]) == 2
    assert timings["action_refinement_passes"] == 2


@pytest.mark.parametrize("refine_action", [False, True])
@pytest.mark.parametrize("refine_context", [False, True])
def test_single_frame_action_survives_every_verification_view(tmp_path, refine_action, refine_context):
    path = tmp_path / "brief-action.mov"
    marker_index = 103
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                    "color=c=gray:s=160x240:r=30:d=8", "-vf",
                    f"drawbox=color=white:t=fill:enable='eq(n,{marker_index})'",
                    "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-y", str(path)], check=True)
    rule = load_rules(ROOT)["active_intake_departure"]

    class BriefActionRuntime:
        def __init__(self):
            self.calls = {"action": 0, "context": 0}
            self.views = []

        def ask(self, prompt, frames, schema, budget, max_tokens=512):
            # The synthetic action is visible in exactly one decoded source frame.
            markers = [i for i, frame in enumerate(frames)
                       if cv2.imread(str(frame.path)).mean() > 240]
            if "candidates" in schema["properties"]:
                assert len(markers) == 1 and frames[markers[0]].index == marker_index
                anchor = markers[0]
                return {"visibility": "clear", "visible_facts": [],
                        "transaction": {"kind": "unknown", "phase": "unknown", "frame_indices": []},
                        "catalog_assessments": [], "detections": [], "candidates": [{
                            "type_id": rule["id"], "anchor_frame": anchor, "start_frame": anchor,
                            "end_frame": anchor, "observed_action": "Synthetic single-frame action"}]}
            names = schema["properties"]["checks"]["properties"]
            stage = "action" if set(names) == set(rule["action_checks"]) else "context"
            self.calls[stage] += 1
            self.views.append((stage, [frames[i].index for i in markers]))
            refine = refine_action if stage == "action" else refine_context
            supported = bool(markers) and not (refine and self.calls[stage] == 1)
            return {"decision": "supported" if supported else "uncertain", "confidence": .8 if supported else 0,
                    "visibility": "clear", "observed_action": "Synthetic single-frame action",
                    "checks": {name: "yes" if supported else "unknown" for name in names},
                    "check_frames": {name: markers for name in names},
                    "evidence_indices": markers, "limitations": [],
                    **({"exception_checks": {key: "excluded" for key in schema["properties"]["exception_checks"]["properties"]},
                        "exception_frames": {key: markers for key in schema["properties"]["exception_checks"]["properties"]}}
                       if "exception_checks" in schema["properties"] else {})}

    runtime = BriefActionRuntime()
    result, details, timings = Pipeline(ROOT, tmp_path, runtime).run(
        {"id": "brief-action", "video_path": str(path), "deadline_at": time.time() + 60}, lambda *_: None)
    assert runtime.views and all(indices == [marker_index] for _, indices in runtime.views)
    assert runtime.calls == {"action": 1 + int(refine_action), "context": 1 + int(refine_context)}
    assert len(result["violations"]) == 1 and not details["review_queue"]
    assert timings["action_refinement_passes"] == int(refine_action)
    evidence = result["violations"][0]["evidence_frames"]
    assert any(abs(frame["timestamp_sec"] - marker_index / 30) < .001 for frame in evidence)


def test_explicit_rejection_is_not_overridden_by_refinement(synthetic_video, tmp_path):
    class RejectedRuntime(FakeRuntime):
        def ask(self, prompt, frames, schema, budget, max_tokens=512):
            answer = super().ask(prompt, frames, schema, budget, max_tokens)
            if "checks" in answer:
                answer["decision"] = "rejected"
                answer["checks"][next(iter(answer["checks"]))] = "unknown"
            return answer
    result, _, timings = Pipeline(ROOT, tmp_path, RejectedRuntime()).run(
        {"id": "rejected", "video_path": str(synthetic_video), "deadline_at": time.time() + 60}, lambda *_: None)
    assert not result["violations"]
    assert timings["action_calls"] == 2 and timings["refinement_passes"] == 0


def test_observed_departure_seeds_candidate_without_model_proposal(synthetic_video, tmp_path):
    class ObservedRuntime(FakeRuntime):
        def ask(self, prompt, frames, schema, budget, max_tokens=512):
            answer = super().ask(prompt, frames, schema, budget, max_tokens)
            if "candidates" in answer:
                answer["candidates"] = []
                answer["visible_facts"] = [{"kind": "workstation_departure", "state": "observed", "frame_indices": [2], "note": "Synthetic action"}]
            return answer
    result, _, timings = Pipeline(ROOT, tmp_path, ObservedRuntime()).run(
        {"id": "trigger", "video_path": str(synthetic_video), "deadline_at": time.time() + 60}, lambda *_: None)
    assert len(result["violations"]) == 1 and timings["candidate_count"] == 1


def test_unsupported_and_omission_episodes_never_become_jury_findings(synthetic_video, tmp_path):
    class ReviewRuntime(FakeRuntime):
        def ask(self, prompt, frames, schema, budget, max_tokens=512):
            answer = super().ask(prompt, frames, schema, budget, max_tokens)
            if "candidates" in answer:
                for item, kind in zip(answer["candidates"], ("unmapped_visual_event", "loose_cash_issued_without_observed_count")):
                    item["type_id"] = kind
            return answer
    result, details, timings = Pipeline(ROOT, tmp_path, ReviewRuntime()).run(
        {"id": "review", "video_path": str(synthetic_video), "deadline_at": time.time() + 60}, lambda *_: None)
    assert result == {"verdict": "no_violation", "violations": []}
    assert len(details["review_queue"]) == 2
    assert timings["action_calls"] == 2


def test_model_failure_is_not_clean_result(synthetic_video, tmp_path):
    with pytest.raises(AnalysisError, match="Synthetic model error"):
        Pipeline(ROOT, tmp_path, FakeRuntime(fail=True)).run(
            {"id": "failure", "video_path": str(synthetic_video), "deadline_at": time.time() + 60}, lambda *_: None)


def test_continuous_scores_use_model_evidence_even_without_confirmed_findings(synthetic_video, tmp_path):
    class ConfidenceRuntime(FakeRuntime):
        def ask(self, prompt, frames, schema, budget, max_tokens=512):
            answer = super().ask(prompt, frames, schema, budget, max_tokens)
            if "candidates" in answer:
                assert "catalog_assessments" not in schema["properties"]
                assert "detections" not in schema["properties"]
                answer["candidates"] = answer["candidates"][:1]
                answer["visible_facts"] = [{"kind": "transaction_start", "state": "observed",
                                            "frame_indices": [0], "note": "Synthetic source boundary"}]
            else:
                answer["confidence"] = .68
                if "exception_checks" in schema["properties"]:
                    answer["decision"] = "uncertain"
                    answer["checks"] = {key: "unknown" for key in answer["checks"]}
            from worker.runtime import _validate_json_schema
            _validate_json_schema(answer, schema)
            return answer

    result, details, timings = Pipeline(ROOT, tmp_path, ConfidenceRuntime()).run(
        {"id": "continuous", "video_path": str(synthetic_video), "deadline_at": time.time() + 60}, lambda *_: None)
    assert result == {"verdict": "no_violation", "violations": []}
    rows = {row["item_id"]: row for row in details["assessments"]}
    assert rows["active_intake_departure"]["score"] == 68
    assert rows["active_intake_departure"]["applicability"] == "unknown"
    # Generic handling cannot clear the complete counting protocol.
    assert rows["intake_banknotes_counted"]["score"] == 100
    assert rows["intake_banknotes_counted"]["assessment"] == "unassessed"
    assert rows["intake_client_identification"]["score"] == 100
    assert not any(row["confirmed_finding"] for row in rows.values())
    assert all(row["calibrated"] is False for row in rows.values())
    assert len(details["assessment_episodes"]) == 1
    assert timings["confidence_observation_count"] == 1
    assert timings["noninitial_score_count"] == 1


def test_presentation_timestamps_and_expired_budget(synthetic_video):
    info = probe_video(synthetic_video, Budget(time.time() + 30))
    assert len(info.pts) == 60
    assert nearest_index((0, .1, .3), .28) == 2
    assert nearest_index((0, .1, .3), 99) == 2
    with pytest.raises(AnalysisError) as error:
        Budget(time.time() - 1).remaining()
    assert error.value.code == "deadline_exceeded"


@pytest.mark.parametrize("count", [4, 8])
@pytest.mark.parametrize("anchor", [0., .1, 1., 1.9, 2.])
def test_verification_sampling_retains_trigger_endpoints_and_frame_budget(count, anchor):
    targets = anchored_targets(0., 2., anchor, count)
    assert len(targets) == count and targets == sorted(targets)
    assert targets[0] == 0 and targets[-1] == 2 and anchor in targets
    assert len(set(targets)) == count


def test_long_frame_selection_does_not_exhaust_ffmpeg_expression_parser(tmp_path):
    path = tmp_path / "many-frames.mov"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", "color=c=gray:s=80x120:r=30:d=6",
                    "-an", "-c:v", "libx264", "-y", str(path)], check=True)
    info = probe_video(path, Budget(time.time() + 30))
    targets = [info.pts[i] for i in range(144)]
    frames = extract_frames(info, targets, tmp_path / "selected", Budget(time.time() + 30), 120)
    assert len(frames) == 144 and all(frame.path.is_file() for frame in frames)


def test_dedupe_keeps_distinct_types_and_episodes():
    candidates = [Candidate("a", 2, 0, 4, "x"), Candidate("a", 3, 1, 5, "x"),
                  Candidate("b", 2, 0, 4, "x"), Candidate("a", 20, 18, 22, "x")]
    assert len(deduplicate_candidates(candidates)) == 3
    assert len(deduplicate_candidates([Candidate("a", 2, 0, 4, "x", "episode-1"),
                                       Candidate("a", 2.1, 0, 4, "x", "episode-2")])) == 2


def test_frame_cache_reuses_exact_source_frame(synthetic_video, tmp_path):
    budget = Budget(time.time() + 30)
    info = probe_video(synthetic_video, budget)
    cache = FrameCache(info, tmp_path / "cache", budget)
    first = cache.get([0, 1], 120)
    second = cache.get([1, 0], 120)
    assert cache.batch == 1
    assert first[0] == second[1] and first[1] == second[0]


def test_final_dedupe_preserves_close_separate_episodes():
    findings = [{"type_id": "a", "timecode_sec": 1, "start_sec": 0, "end_sec": 1},
                {"type_id": "a", "timecode_sec": 2, "start_sec": 1.5, "end_sec": 2},
                {"type_id": "a", "timecode_sec": 2.1, "start_sec": 1.6, "end_sec": 2.2}]
    assert len(deduplicate_findings(findings)) == 2


def test_runtime_refuses_remote_endpoint(monkeypatch, tmp_path):
    monkeypatch.setenv("FORTE_LLM_URL", "https://example.com/v1")
    with pytest.raises(ValueError, match="loopback"):
        LocalRuntime(tmp_path, tmp_path)


def test_every_rule_check_and_valid_frame_required():
    rule = next(iter(load_rules(ROOT).values()))
    frames = [FrameRef(0, 0, Path("unused.jpg"))]
    answer = {"decision": "supported", "visibility": "clear", "checks": {k: "yes" for k in rule["required_checks"]},
              "check_frames": {k: [0] for k in rule["required_checks"]},
              "observed_action": "Synthetic", "evidence_indices": [0]}
    assert supported_verification(answer, rule, frames)
    answer["confidence"] = 0
    assert not supported_verification(answer, rule, frames)
    answer["confidence"] = .68
    assert supported_verification(answer, rule, frames)
    first = rule["required_checks"][0]
    answer["check_frames"][first] = []
    assert not supported_verification(answer, rule, frames)
    answer["check_frames"][first] = [0]
    answer["checks"][rule["required_checks"][0]] = "unknown"
    assert not supported_verification(answer, rule, frames)
    answer["checks"][rule["required_checks"][0]] = "yes"
    answer["evidence_indices"] = [999]
    assert not supported_verification(answer, rule, frames)
