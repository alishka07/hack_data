"""Version-bound empirical decision thresholds and source evidence selection."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import cv2

from worker.video import AnalysisError


def joint_score(action: dict, context: dict) -> float:
    """Correlated verifier calls never accumulate confidence."""
    values = [answer.get("confidence", 1.) for answer in (action, context)]
    if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise AnalysisError("model_schema_invalid", "Verifier scores must be finite and between zero and one")
    return min(values)


class ThresholdPolicy:
    """An empirical cutoff is calibrated separately from any probability claim."""
    def __init__(self, model_id: str, pipeline_version: str, path: Path | None = None):
        self.model_id, self.pipeline_version = model_id, pipeline_version
        self.rules, self.provenance = {}, {"status": "uncalibrated", "probability_calibrated": False,
                                         "reason": "No reviewed verifier-outcome calibration artifact installed"}
        if path is None and os.getenv("FORTE_CALIBRATION_PATH"):
            path = Path(os.environ["FORTE_CALIBRATION_PATH"])
        if path is not None:
            try:
                artifact = json.loads(path.read_text(encoding="utf-8"))
                if (model_id.startswith("unattested:") or artifact["schema_version"] != "1.0" or artifact["model_id"] != model_id
                        or artifact["pipeline_version"] != pipeline_version
                        or artifact["score_method"] != "minimum_action_context"
                        or artifact["group_overlap"] != 0 or not isinstance(artifact["rules"], dict)):
                    raise ValueError()
                for rule_id, rule in artifact["rules"].items():
                    if rule.get("status") == "validated":
                        threshold = rule["threshold"]
                        if (type(threshold) not in (int, float) or not math.isfinite(threshold)
                                or not 0 <= threshold <= 1 or rule["validation"]["selected"] < 1):
                            raise ValueError()
                        self.rules[rule_id] = rule
                self.provenance = {"status": "empirical_thresholds" if self.rules else "uncalibrated", "probability_calibrated": False,
                                   "validated_rules": sorted(self.rules), "group_overlap": 0,
                                   "model_id": model_id, "pipeline_version": pipeline_version}
            except (OSError, ValueError, TypeError, KeyError) as exc:
                raise AnalysisError("calibration_invalid", "Calibration artifact is invalid or belongs to another model/pipeline") from exc

    def evaluate(self, rule_id: str, action: dict, context: dict) -> dict:
        score = joint_score(action, context)
        record = self.rules.get(rule_id)
        threshold = record["threshold"] if record else 0.
        return {"score": score, "threshold": threshold, "accepted": score > 0 and score >= threshold,
                "threshold_calibrated": record is not None, "probability_calibrated": False,
                "method": "minimum_action_context", "rule_id": rule_id}


def _metrics(rows, threshold):
    selected = [row for row in rows if row["score"] >= threshold and row["score"] > 0]
    tp = sum(row["label"] for row in selected)
    positives = sum(row["label"] for row in rows)
    return {"examples": len(rows), "positives": positives, "negatives": len(rows) - positives,
            "source_groups": len({row["group_id"] for row in rows}),
            "positive_source_groups": len({row["group_id"] for row in rows if row["label"]}),
            "negative_source_groups": len({row["group_id"] for row in rows if not row["label"]}),
            "selected_positive_groups": len({row["group_id"] for row in selected if row["label"]}),
            "selected": len(selected), "true_positives": tp, "false_positives": len(selected) - tp,
            "precision": tp / len(selected) if selected else 0., "recall": tp / positives if positives else 0.}


def fit_thresholds(rows: list[dict], *, model_id: str, pipeline_version: str,
                   target_precision: float = .9, minimum_examples: int = 20,
                   minimum_class: int = 5) -> dict:
    """Tune cutoffs on calibration groups; audit on disjoint validation groups.

    Input labels must be reviewed candidate outcomes, not whole-video labels.
    An adapter/prompt/model change requires a new bound artifact and evaluation.
    These small-sample metrics are exploratory, not a production release claim.
    """
    if (not model_id or not pipeline_version or not 0 < target_precision <= 1
            or minimum_examples < 2 or minimum_class < 1):
        raise ValueError("Invalid calibration settings")
    groups, events = {}, set()
    for row in rows:
        if (not isinstance(row, dict) or row.get("partition") not in {"calibration", "validation"}
                or not all(isinstance(row.get(k), str) and row[k] for k in ("rule_id", "group_id", "event_id"))
                or type(row.get("label")) is not bool or row.get("reviewed") is not True
                or type(row.get("score")) not in (int, float) or not math.isfinite(row["score"])
                or not 0 <= row["score"] <= 1):
            raise ValueError("Calibration needs reviewed candidate labels, finite scores and family group IDs")
        if row["group_id"] in groups and groups[row["group_id"]] != row["partition"]:
            raise ValueError("Source/augmentation group crosses calibration and validation")
        groups[row["group_id"]] = row["partition"]
        event = (row["rule_id"], row["event_id"])
        if event in events:
            raise ValueError("Duplicate candidate outcome; repeated or augmented evidence cannot inflate calibration")
        events.add(event)
    result = {"schema_version": "1.0", "model_id": model_id, "pipeline_version": pipeline_version,
              "score_method": "minimum_action_context", "target_precision": target_precision,
              "group_overlap": 0, "probability_calibrated": False, "rules": {}}
    for rule_id in sorted({row["rule_id"] for row in rows}):
        parts = {part: [row for row in rows if row["rule_id"] == rule_id and row["partition"] == part]
                 for part in ("calibration", "validation")}
        counts = {part: _metrics(part_rows, 0.) for part, part_rows in parts.items()}
        if any(metric["source_groups"] < minimum_examples or min(metric["positive_source_groups"], metric["negative_source_groups"]) < minimum_class
               for metric in counts.values()):
            result["rules"][rule_id] = {"status": "insufficient_data", "counts": counts}
            continue
        choices = [(threshold, _metrics(parts["calibration"], threshold))
                   for threshold in sorted({row["score"] for row in parts["calibration"] if row["score"] > 0})]
        choices = [(t, m) for t, m in choices if m["precision"] >= target_precision and m["selected_positive_groups"] >= minimum_class]
        if not choices:
            result["rules"][rule_id] = {"status": "target_not_met", "counts": counts}
            continue
        threshold, calibration = max(choices, key=lambda pair: (pair[1]["recall"], pair[1]["precision"], -pair[0]))
        validation = _metrics(parts["validation"], threshold)
        passed = validation["precision"] >= target_precision and validation["selected_positive_groups"] >= minimum_class
        result["rules"][rule_id] = {"status": "validated" if passed else "validation_failed", "threshold": threshold,
                                    "calibration": calibration, "validation": validation}
    return result


def select_evidence(action_frames, action_indices, context_frames, context_indices, anchor, maximum=3):
    """Keep the action, then distinct cited context; use sharpness to break ties.

    Only already-cited images are eligible. The caller re-exports originals;
    quality scoring never changes the source timestamps or supports a predicate.
    """
    if not 1 <= maximum <= 3:
        raise ValueError("Evidence output must contain between one and three frames")
    action = {action_frames[i].index: action_frames[i] for i in action_indices}
    if not action:
        raise AnalysisError("evidence_failed", "No cited action image is available")
    context = {context_frames[i].index: context_frames[i] for i in context_indices}
    eligible = {**context, **action}
    quality = {}
    for index, frame in eligible.items():
        pixels = cv2.imread(str(frame.path), cv2.IMREAD_GRAYSCALE)
        if pixels is None:
            raise AnalysisError("evidence_failed", "A cited evidence image cannot be read")
        quality[index] = float(cv2.Laplacian(pixels, cv2.CV_64F).var())
    primary = min(action.values(), key=lambda frame: (abs(frame.timestamp - anchor), -quality[frame.index]))
    selected = [primary]
    # A cited context image before and after is more informative than three
    # adjacent sharp pictures of the action. Never manufacture missing context.
    for before in (True, False):
        candidates = [frame for frame in eligible.values() if frame.index not in {f.index for f in selected}
                      and (frame.timestamp < primary.timestamp if before else frame.timestamp > primary.timestamp)]
        if candidates and len(selected) < maximum:
            selected.append(max(candidates, key=lambda frame: (abs(frame.timestamp - primary.timestamp), quality[frame.index])))
    remaining = [frame for frame in eligible.values() if frame.index not in {f.index for f in selected}]
    selected.extend(sorted(remaining, key=lambda frame: -quality[frame.index])[:maximum - len(selected)])
    return sorted(selected, key=lambda frame: frame.timestamp), primary.timestamp
