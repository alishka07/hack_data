"""Catalog estimates derived from focused verification, without extra model tasks."""
from __future__ import annotations

import math
from pathlib import Path

from worker.confidence import OBSERVABLE_CHECKS
from worker.rule_gate import exceptions_excluded
from worker.video import FrameRef


def verification_observations(attempts, rules):
    observations = []
    for number, attempt in enumerate(attempts):
        identifier = attempt["type_id"]
        if identifier not in tuple(OBSERVABLE_CHECKS)[:3] or identifier not in rules:
            continue
        prefix = "refined_action" if attempt.get("refined_action") else "action"
        action = attempt.get(prefix) or {}
        context_prefix = "refined_context" if attempt.get("refined_context") else "context"
        context = attempt.get(context_prefix) or {}
        if (context.get("decision") == "rejected" or "no" in context.get("checks", {}).values()
                or "applies" in context.get("exception_checks", {}).values()):
            # A rejected rule application must not reappear as a catalog alert.
            continue
        # Only explicit positive support supplies an estimate. Unknown checks,
        # rejection and visibility gaps cannot turn a default into observed risk.
        confidence = action.get("confidence", 1. if action.get("decision") == "supported" else 0.)
        if (type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 < confidence <= 1
                or action.get("decision") == "rejected" or "no" in action.get("checks", {}).values()
                or "yes" not in action.get("checks", {}).values()):
            continue
        frames = attempt.get(prefix + "_frames", [])
        indices = set(action.get("evidence_indices", []))
        refs = {frame["source_index"]: frame for i, frame in enumerate(frames) if i in indices}
        if not refs:
            continue
        context_frames = attempt.get(context_prefix + "_frames", [])
        valid_sources = [FrameRef(row["source_index"], row["timestamp_sec"], Path("unused")) for row in context_frames]
        applicable = (context.get("decision") == "supported" and context.get("visibility") == "clear"
                      and all(context.get("checks", {}).get(check) == "yes"
                              and 1 <= len(context.get("check_frames", {}).get(check, [])) <= 3
                              and all(type(i) is int and 0 <= i < len(valid_sources)
                                      for i in context["check_frames"][check])
                              for check in rules[identifier]["required_checks"])
                      and exceptions_excluded(context, rules[identifier], valid_sources))
        if applicable:
            confidence = min(confidence, context.get("confidence", 1.))
        if confidence <= 0:
            continue
        evidence = sorted(refs.values(), key=lambda row: row["timestamp_sec"])
        observations.append({"item_id": identifier, "assessment": "event_evidence" if applicable else "partial_support",
                             "model_assessment": "focused_verification", "confidence": float(confidence),
                             "applicability": "applicable" if applicable else "unknown", "evidence_refs": evidence,
                             "start_sec": evidence[0]["timestamp_sec"], "end_sec": evidence[-1]["timestamp_sec"],
                             "window_id": 100000 + number, "observable": action.get("visibility") != "unusable",
                             "window_start_sec": min(row["timestamp_sec"] for row in frames),
                             "window_end_sec": max(row["timestamp_sec"] for row in frames)})
    return observations
