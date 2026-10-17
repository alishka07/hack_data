"""Optional approved CPU event proposals; failed research models stay disabled.

The default gate uses only local artifact/evaluation JSON and the standard
library. Torch is imported only by a deliberately enabled, approved encoder.
No dataset labels, identity hashes, training cache or network is used to score
the submitted video. File SHA checks here bind model assets to their provenance.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

from worker.video import AnalysisError, FrameRef


def _json(path: Path) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("Temporal model and evaluation must be JSON objects")
    return value, raw


def _finite(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _metadata(model: dict, report: dict, model_path: Path) -> dict:
    event = report.get("metrics", {}).get("event", {}) if isinstance(report.get("metrics", {}), dict) else {}
    event = event if isinstance(event, dict) else {}
    return {"model": {"path": str(model_path), "model_type": model.get("model_type")
                      if isinstance(model.get("model_type"), str) else None,
                      "approved_for_inference": model.get("approved_for_inference") is True,
                      "feature_fingerprint": model.get("feature_fingerprint")
                      if isinstance(model.get("feature_fingerprint"), str) else None},
            "evaluation": {"approved_for_inference": report.get("approved_for_inference") is True,
                           **{key: event[key] for key in ("balanced_accuracy", "recall", "specificity")
                              if _finite(event.get(key))},
                           "balanced_accuracy_gain": report.get("balanced_accuracy_gain")
                           if _finite(report.get("balanced_accuracy_gain")) else None},
            "score_calibrated": False,
            "timing_scope": "input_window_proposal_not_validated_event_boundaries"}


def _approval(model: dict, report: dict, raw: bytes) -> tuple[str, str]:
    if model.get("approved_for_inference") is False or report.get("approved_for_inference") is False:
        return "failed_gate", "The temporal research model failed its inference approval gate"
    if model.get("approved_for_inference") is not True or report.get("approved_for_inference") is not True:
        return "not_evaluated", "Temporal inference requires explicit artifact and evaluation approval"
    if (report.get("final_model_sha256") != hashlib.sha256(raw).hexdigest()
            or report.get("feature_fingerprint") != model.get("feature_fingerprint")):
        return "invalid_artifact", "The temporal evaluation is not bound to this model artifact"
    metrics, gate = report.get("metrics", {}), report.get("inference_gate", {})
    event = metrics.get("event", {}) if isinstance(metrics, dict) else None
    if not isinstance(event, dict) or not isinstance(gate, dict):
        return "invalid_evaluation", "The temporal evaluation has no valid measured approval criteria"
    pairs = [(event.get("balanced_accuracy"), gate.get("minimum_nested_balanced_accuracy")),
             (event.get("recall"), gate.get("minimum_nested_recall")),
             (event.get("specificity"), gate.get("minimum_nested_specificity")),
             (report.get("balanced_accuracy_gain"), gate.get("minimum_balanced_accuracy_gain_over_baseline"))]
    if any(not _finite(value) or not _finite(minimum) for value, minimum in pairs):
        return "invalid_evaluation", "The temporal approval metrics must be explicit finite numbers"
    if (any(not 0 <= value <= 1 for value, _ in pairs[:3])
            or not -1 <= pairs[3][0] <= 1 or any(not 0 <= minimum <= 1 for _, minimum in pairs)):
        return "invalid_evaluation", "The temporal approval metrics and thresholds are outside valid ranges"
    if any(value < minimum for value, minimum in pairs):
        return "failed_gate", "Measured temporal performance does not meet its recorded approval criteria"
    return "approved", "Approved local temporal proposal model"


def _load_encoder(assets: Path, threads: int):
    from worker.action_encoder import ActionEncoder
    return ActionEncoder(assets, threads=threads)


def temporal_proposals(root, info, cache, budget) -> dict:
    """Return optional source-bound proposal intervals with no legal category.

    FORTE_TEMPORAL_ENABLED=1 is an explicit request to run the adapter, never an
    override of approval. Configuration/model failures then fail the job instead
    of silently dropping the requested stage. Disabled inference never loads a
    numerical model runtime.
    """
    setting = os.environ.get("FORTE_TEMPORAL_ENABLED", "0").strip()
    if setting not in {"0", "1"}:
        raise AnalysisError("temporal_configuration_invalid", "FORTE_TEMPORAL_ENABLED must be 0 or 1")
    enabled = setting == "1"
    root = Path(root)
    model_path = Path(os.environ.get("FORTE_TEMPORAL_MODEL", str(root / "models/case-detector-v2/model.json")))
    report_path = Path(os.environ.get("FORTE_TEMPORAL_EVALUATION", str(root / "docs/event-model-evaluation.json")))
    model, report, raw = {}, {}, b""
    try:
        model, raw = _json(model_path)
        report, _ = _json(report_path)
    except FileNotFoundError as exc:
        if enabled:
            raise AnalysisError("temporal_model_unapproved", "An approved temporal model and matching evaluation are required") from exc
        status = "disabled_failed_gate" if model.get("approved_for_inference") is False else "disabled_not_evaluated"
        return {"status": status, "proposals": [], **_metadata(model, report, model_path)}
    except (OSError, ValueError, UnicodeError) as exc:
        if enabled:
            raise AnalysisError("temporal_model_invalid", "The configured local temporal model or evaluation is unreadable") from exc
        return {"status": "disabled_invalid_artifact", "proposals": [], **_metadata(model, report, model_path)}
    status, reason = _approval(model, report, raw)
    metadata = _metadata(model, report, model_path)
    if status != "approved":
        if enabled:
            code = "temporal_model_unapproved" if status in {"failed_gate", "not_evaluated"} else "temporal_model_invalid"
            raise AnalysisError(code, reason)
        return {"status": "disabled_" + status, "reason": reason, "proposals": [], **metadata}
    if not enabled:
        return {"status": "disabled_by_configuration", "proposals": [], **metadata}

    budget.remaining()
    # All imports that can reach optional numerical/encoder dependencies are
    # behind both the explicit enable flag and the independent approval gate.
    try:
        import cv2
        import numpy as np
        from worker.action_encoder import window_metadata
        from worker.event_detector import _validate_model, _verify_assets
        from worker.event_features import contextualize_windows
        from worker.event_classifier import predict_event_windows

        _validate_model(model)
        assets = Path(os.environ.get("FORTE_TEMPORAL_ENCODER", str(root / "models/action-encoder")))
        _verify_assets(assets, model["feature_config"]["encoder"])
        threads = int(os.environ.get("FORTE_TEMPORAL_THREADS", "4"))
        if not 1 <= threads <= 64:
            raise ValueError("Temporal encoder threads must be between 1 and 64")
        budget.remaining()
        encoder = _load_encoder(assets, threads)
        budget.remaining()
        windows = window_metadata(info.pts, info.duration)
        features, sources = [], []
        for window in windows:
            budget.remaining()
            indices = window["input_indices"]
            frames = cache.get([info.pts[index] for index in indices], height=112)
            if (len(frames) != len(indices) or any(not isinstance(frame, FrameRef)
                    or type(frame.index) is not int or frame.index != index
                    or frame.timestamp != info.pts[index] for frame, index in zip(frames, indices))):
                raise AnalysisError("temporal_source_invalid", "A temporal input frame is not bound to its authoritative source timestamp")
            images = []
            for frame in frames:
                budget.remaining()
                image = cv2.imread(str(frame.path), cv2.IMREAD_COLOR)
                if image is None:
                    raise AnalysisError("temporal_frame_failed", "A cached temporal source frame could not be read")
                images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            budget.remaining()
            encoded = np.asarray(encoder.encode_clips([images]))
            budget.remaining()
            if encoded.shape != (1, 512) or not np.isfinite(encoded).all():
                raise ValueError("Temporal encoder must return one finite 512-dimensional window")
            features.append(encoded[0])
            sources.append(sorted(set(indices)))
        budget.remaining()
        scores = np.asarray(predict_event_windows(model["classifier"], contextualize_windows(np.stack(features))))
        if scores.shape != (len(windows),) or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
            raise ValueError("The temporal classifier returned invalid window scores")
        proposals = []
        for window, indices, score in zip(windows, sources, scores):
            budget.remaining()
            if float(score) < model["threshold"]:
                continue
            anchor_index = min(indices, key=lambda index: abs(info.pts[index] - window["center_sec"]))
            proposals.append({"timecode_sec": info.pts[anchor_index], "start_sec": info.pts[indices[0]],
                              "end_sec": info.pts[indices[-1]], "reason": "temporal_score", "source_indices": indices,
                              "refs": [{"source_index": index, "timestamp_sec": info.pts[index]} for index in indices],
                              "score": float(score)})
        return {"status": "enabled_approved", "proposals": proposals, "threshold": model["threshold"],
                "windows_completed": len(windows), "device": "cpu", **metadata}
    except AnalysisError:
        raise
    except (ImportError, OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        raise AnalysisError("temporal_model_invalid", "The enabled temporal stage could not validate or run its local model") from exc
