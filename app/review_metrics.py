"""Exploratory quality metrics from separately saved human reviews.

The input contains immutable model cases and independent human decisions. Missing
reviews, uncertain decisions and point anchors are never invented as negative
labels or event intervals. This module neither trains models nor changes jobs.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import math
from typing import Any


_MODEL_STATUSES = {"confirmed", "rejected", "review_required"}
_DECISIVE = {"confirmed", "dismissed"}
_PROBABILITY_SCOPES = {"joint_checks", "event_occurrence"}


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    except (OverflowError, ValueError):
        return None


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _interval(item: dict) -> tuple[float, float] | None:
    start, end = _number(item.get("start_sec")), _number(item.get("end_sec"))
    if start is None or end is None or start < 0 or end <= start:
        return None
    return start, end


def temporal_iou(first: tuple[float, float], second: tuple[float, float]) -> float:
    """Intersection over union for two real, positive-length event intervals."""
    intersection = max(0.0, min(first[1], second[1]) - max(first[0], second[0]))
    union = max(first[1], second[1]) - min(first[0], second[0])
    return intersection / union if union > 0 else 0.0


def _ratio(numerator: int | float, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _new_stats() -> dict:
    return {
        "candidate_adjudication": Counter(confirmed=0, dismissed=0, uncertain=0, unreviewed=0),
        "automatic_decision": Counter(true_positive=0, false_positive=0, false_negative=0, true_negative=0),
        "event_coverage": Counter(eligible_records=0, true_positive=0, false_negative=0),
        "known_misses": Counter(confirmed_nonpositive_candidates=0, manual_events=0),
        "localization_values": [],
        "anchors_without_intervals": 0,
        "probability_values": [],
        "probability_exclusions": Counter(unreviewed_or_uncertain=0, incompatible_scope=0, invalid_probability=0),
        "deduplicated_missed_events": 0,
        "invalid_missed_events": 0,
        "unknown_model_status": 0,
    }


def _finalize(stats: dict, has_incomplete_coverage: bool) -> dict:
    candidates = dict(stats["candidate_adjudication"])
    candidates["positive_fraction"] = _ratio(candidates["confirmed"], candidates["confirmed"] + candidates["dismissed"])
    automatic = dict(stats["automatic_decision"])
    automatic["precision"] = _ratio(automatic["true_positive"], automatic["true_positive"] + automatic["false_positive"])
    automatic["reviewed_candidate_recall"] = _ratio(automatic["true_positive"], automatic["true_positive"] + automatic["false_negative"])
    automatic["scope"] = "reviewed_candidates_only"
    event = dict(stats["event_coverage"])
    event["recall"] = _ratio(event["true_positive"], event["true_positive"] + event["false_negative"])
    event["denominator_status"] = "complete_reviewed_subset" if event["eligible_records"] else "incomplete"
    event["has_excluded_records"] = has_incomplete_coverage
    known = dict(stats["known_misses"])
    known["total"] = known["confirmed_nonpositive_candidates"] + known["manual_events"]
    values = stats["localization_values"]
    localization = {"pairs": len(values), "mean_iou": _ratio(sum(values), len(values)),
                    "iou_at_0_5": _ratio(sum(value >= 0.5 for value in values), len(values)),
                    "anchors_without_intervals": stats["anchors_without_intervals"]}
    probabilities = stats["probability_values"]
    bins = []
    for index in range(10):
        entries = [(p, label) for p, label in probabilities if min(int(p * 10), 9) == index]
        bins.append({"lower": index / 10, "upper": (index + 1) / 10, "samples": len(entries),
                     "mean_probability": _ratio(sum(p for p, _ in entries), len(entries)),
                     "observed_fraction": _ratio(sum(label for _, label in entries), len(entries))})
    calibration = {
        "status": "exploratory" if probabilities else "no_eligible_probabilities",
        "samples": len(probabilities),
        "brier_score": _ratio(sum((p - label) ** 2 for p, label in probabilities), len(probabilities)),
        "expected_calibration_error": _ratio(sum(item["samples"] * abs(item["mean_probability"] - item["observed_fraction"])
                                                for item in bins if item["samples"]), len(probabilities)),
        "bins": bins, "excluded": dict(stats["probability_exclusions"]),
        "scope": "human_violation_labels_with_joint_checks_or_event_occurrence_probability",
        "establishes_calibration": False,
    }
    return {"candidate_adjudication": candidates, "automatic_decision": automatic,
            "event_coverage": event, "known_misses": known, "temporal_localization": localization,
            "probability_evaluation": calibration, "deduplicated_missed_events": stats["deduplicated_missed_events"],
            "invalid_missed_events": stats["invalid_missed_events"], "unknown_model_status": stats["unknown_model_status"]}


def _review(record: dict) -> dict:
    value = record.get("review_state")
    return value if isinstance(value, dict) else {}


def _decisions(state: dict) -> dict[str, dict]:
    """Conflicting duplicate labels are uncertain, never silently last-wins."""
    output: dict[str, dict] = {}
    for value in state.get("decisions", []) if isinstance(state.get("decisions"), list) else []:
        if not isinstance(value, dict) or not _text(value.get("case_id")):
            continue
        key = value["case_id"]
        previous = output.get(key)
        if previous is not None and previous != value:
            output[key] = {"case_id": key, "decision": "uncertain"}
        else:
            output[key] = value
    return output


def _has_review(record: dict) -> bool:
    state = _review(record)
    return bool(state.get("decisions") or state.get("missed_events") or state.get("coverage_complete") is True)


def _deduplicate_records(records: list[dict]) -> tuple[list[dict], int]:
    """One evaluated analysis per exact video; prefer a saved, recent review."""
    chosen: dict[str, tuple[tuple, dict]] = {}
    for index, record in enumerate(records):
        fingerprint = _text(record.get("source_fingerprint"))
        key = "source:" + fingerprint if fingerprint else "record:" + str(index)
        # ISO UTC timestamps sort lexically; numeric timestamps are also accepted.
        stamp = record.get("review_updated_at") or record.get("created_at", "")
        numeric_stamp = _number(stamp)
        order = (int(_has_review(record)), numeric_stamp if numeric_stamp is not None else 0,
                 _text(stamp), index)
        if key not in chosen or order > chosen[key][0]:
            chosen[key] = order, record
    return [item[1] for item in chosen.values()], len(records) - len(chosen)


def _group_audit(records: list[dict]) -> dict:
    """Conservative components retain shared families, cameras and sessions."""
    parents = list(range(len(records)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    seen: dict[tuple[str, str], int] = {}
    known = []
    for index, record in enumerate(records):
        state = _review(record)
        complete = all(_text(state.get(field)) for field in ("source_group", "camera_id", "session_id"))
        known.append(complete)
        for field, value in [("source", _text(record.get("source_fingerprint"))),
                             ("family", _text(state.get("source_group"))),
                             ("camera", _text(state.get("camera_id"))),
                             ("session", _text(state.get("camera_id")) + ":" + _text(state.get("session_id"))
                              if _text(state.get("camera_id")) and _text(state.get("session_id")) else "")]:
            if not value:
                continue
            key = field, value
            if key in seen:
                parents[root(index)] = root(seen[key])
            else:
                seen[key] = index
    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        components[root(index)].append(index)
    manifest = []
    for members in components.values():
        tokens = []
        for index in members:
            record, state = records[index], _review(records[index])
            tokens.extend([_text(record.get("source_fingerprint")), _text(state.get("source_group")),
                           _text(state.get("camera_id")), _text(state.get("session_id"))])
        digest = hashlib.sha256("\n".join(sorted(tokens)).encode()).hexdigest()[:16]
        manifest.append({"group_id": digest, "record_count": len(members),
                         "metadata_complete": all(known[index] for index in members), "partition": "unassigned"})
    manifest.sort(key=lambda item: item["group_id"])
    unknown = sum(not item for item in known)
    eligible = sum(item["metadata_complete"] for item in manifest)
    families = {_text(_review(record).get("source_group")) for record in records} - {""}
    return {"status": "missing_metadata" if unknown else "ready_for_group_split" if eligible >= 3 else "insufficient_groups",
            "source_families": len(families), "independent_camera_groups": eligible,
            "unknown_group_records": unknown, "split_manifest": manifest,
            "split_performed": False,
            "leakage_warnings": (["Не у всех видео указаны исходная группа, камера и сессия; независимость не установлена."] if unknown else [])
                + (["Меньше трёх независимых групп: отдельные обучение, валидация и тест пока невозможны."] if eligible < 3 else []),
            "policy": "All original/augmented siblings, exact duplicates and shared cameras/sessions stay in one component. Assign whole components before training."}


def build_review_quality(records: list[dict]) -> dict:
    """Build aggregate diagnostics; see docs/reviewer-workflow.md for the contract.

    Every record has job_id, source_fingerprint, model_cases, review_state, and
    optionally review_updated_at/created_at. Saved reviews are never ground truth
    for unreviewed parts of a video. Raw model `confirmed` alone is a positive.
    """
    supplied = [item for item in records if isinstance(item, dict)]
    evaluated, duplicates = _deduplicate_records(supplied)
    by_type: dict[str, dict] = defaultdict(_new_stats)
    total = _new_stats()
    eligible_count = 0
    reviewed_count = 0
    duplicate_case_ids = 0
    orphan_decisions = 0

    for record in evaluated:
        state = _review(record)
        reviewed_count += int(_has_review(record))
        decisions = _decisions(state)
        raw_cases = record.get("model_cases")
        cases = [item for item in raw_cases if isinstance(item, dict)] if isinstance(raw_cases, list) else []
        id_counts = Counter(_text(item.get("case_id")) for item in cases)
        duplicate_case_ids += sum(count - 1 for key, count in id_counts.items() if key and count > 1)
        candidate_ids = {_text(item.get("case_id")) for item in cases} - {""}
        unmatched_decisions = len(set(decisions) - candidate_ids)
        orphan_decisions += unmatched_decisions
        reliable = [(item, decisions.get(_text(item.get("case_id")), {})) for item in cases
                    if _text(item.get("case_id")) and id_counts[_text(item.get("case_id"))] == 1]
        all_decided = (isinstance(raw_cases, list) and len(raw_cases) == len(cases) and not unmatched_decisions
                       and len(reliable) == len(cases)
                       and all(decision.get("decision") in _DECISIVE for _, decision in reliable))
        group_complete = all(_text(state.get(key)) for key in ("source_group", "camera_id", "session_id"))
        eligible = state.get("coverage_complete") is True and all_decided and group_complete and bool(_text(state.get("reviewer")))
        eligible = eligible and all(item.get("status") in _MODEL_STATUSES for item in cases)
        # Exact reviewed intervals identify repeated proposals for one event.
        # Ambiguous overlaps or point-only labels cannot establish event counts.
        events: dict[tuple[str, tuple[float, float]], bool] = {}
        event_boundaries_complete = True
        record_types = set()
        confirmed_refs: dict[str, list[tuple[dict, dict]]] = defaultdict(list)

        for item in cases:
            type_id = _text(item.get("type_id")) or "unknown"
            record_types.add(type_id)
            stats = by_type[type_id]
            identifier = _text(item.get("case_id"))
            decision = decisions.get(identifier, {}) if identifier and id_counts[identifier] == 1 else {}
            label = decision.get("decision")
            result = label if label in {"confirmed", "dismissed", "uncertain"} else "unreviewed"
            for bucket in (stats, total):
                bucket["candidate_adjudication"][result] += 1
            if label not in _DECISIVE:
                for bucket in (stats, total):
                    bucket["probability_exclusions"]["unreviewed_or_uncertain"] += 1
                continue
            positive = label == "confirmed"
            if positive:
                confirmed_refs[type_id].append((item, decision))
                reference = _interval(decision)
                if reference is None:
                    event_boundaries_complete = False
                else:
                    key = type_id, reference
                    events[key] = events.get(key, False) or item.get("status") == "confirmed"
            model_status = item.get("status")
            if model_status not in _MODEL_STATUSES:
                for bucket in (stats, total):
                    bucket["unknown_model_status"] += 1
            else:
                predicted = model_status == "confirmed"
                field = ("true_positive" if positive else "false_positive") if predicted else ("false_negative" if positive else "true_negative")
                for bucket in (stats, total):
                    bucket["automatic_decision"][field] += 1
                    if positive and not predicted:
                        bucket["known_misses"]["confirmed_nonpositive_candidates"] += 1
            if positive and model_status == "confirmed":
                predicted_interval, human_interval = _interval(item), _interval(decision)
                for bucket in (stats, total):
                    if predicted_interval is not None and human_interval is not None:
                        bucket["localization_values"].append(temporal_iou(predicted_interval, human_interval))
                    else:
                        bucket["anchors_without_intervals"] += 1
            probability = _number(item.get("probability"))
            for bucket in (stats, total):
                if item.get("probability_scope") not in _PROBABILITY_SCOPES:
                    bucket["probability_exclusions"]["incompatible_scope"] += 1
                elif probability is None or not 0 <= probability <= 1:
                    bucket["probability_exclusions"]["invalid_probability"] += 1
                else:
                    bucket["probability_values"].append((probability, int(positive)))

        raw_misses = state.get("missed_events")
        misses = raw_misses if isinstance(raw_misses, list) else []
        seen_misses = set()
        matched_cases: set[str] = set()
        for miss in misses:
            if not isinstance(miss, dict):
                total["invalid_missed_events"] += 1
                event_boundaries_complete = False
                continue
            type_id = _text(miss.get("type_id")) or "unknown"
            stats = by_type[type_id]
            record_types.add(type_id)
            interval = _interval(miss)
            if interval is None or not _text(miss.get("event_id")) or type_id == "unknown":
                event_boundaries_complete = False
                for bucket in (stats, total):
                    bucket["invalid_missed_events"] += 1
                continue
            # Two identical events are one missed reference, irrespective of UI ID.
            identity = (type_id, interval)
            if identity in seen_misses:
                for bucket in (stats, total):
                    bucket["deduplicated_missed_events"] += 1
                continue
            seen_misses.add(identity)
            matches = []
            for candidate, decision in confirmed_refs[type_id]:
                case_id = candidate["case_id"]
                if case_id in matched_cases:
                    continue
                reference = _interval(decision)
                point = _number(candidate.get("timecode_sec"))
                overlap = temporal_iou(interval, reference) if reference is not None else 0
                if overlap > 0 or (reference is None and point is not None and interval[0] <= point <= interval[1]):
                    matches.append((overlap, case_id))
            if matches:
                matched_cases.add(max(matches)[1])
                for bucket in (stats, total):
                    bucket["deduplicated_missed_events"] += 1
                continue
            for bucket in (stats, total):
                bucket["known_misses"]["manual_events"] += 1
            events[(type_id, interval)] = False

        distinct_events = list(events)
        ambiguous_overlap = any(first_type == second_type and temporal_iou(first_interval, second_interval) > 0
                                for index, (first_type, first_interval) in enumerate(distinct_events)
                                for second_type, second_interval in distinct_events[index + 1:])
        eligible = eligible and event_boundaries_complete and not ambiguous_overlap
        eligible_count += int(eligible)
        if eligible:
            total["event_coverage"]["eligible_records"] += 1
            for type_id in record_types:
                by_type[type_id]["event_coverage"]["eligible_records"] += 1
            for (type_id, _), predicted in events.items():
                field = "true_positive" if predicted else "false_negative"
                by_type[type_id]["event_coverage"][field] += 1
                total["event_coverage"][field] += 1

    incomplete = len(evaluated) != eligible_count
    labels = total["candidate_adjudication"]["confirmed"] + total["candidate_adjudication"]["dismissed"]
    return {"schema_version": 1, "status": "exploratory" if labels or total["known_misses"]["manual_events"] else "no_reviewed_labels",
            "records": {"total": len(supplied), "evaluated": len(evaluated), "with_review": reviewed_count,
                        "exact_duplicate_analyses_excluded": duplicates, "eligible_complete_coverage": eligible_count,
                        "excluded_incomplete_coverage": len(evaluated) - eligible_count,
                        "duplicate_case_ids": duplicate_case_ids, "orphan_decisions": orphan_decisions},
            "totals": _finalize(total, incomplete),
            "per_type": [{"type_id": key, **_finalize(value, incomplete)} for key, value in sorted(by_type.items())],
            # Keep provenance links from every analysis, including duplicate
            # videos; deduplicating metric samples must not erase family links.
            "grouping": _group_audit(supplied),
            "limitations": [
                "Это исследовательская оценка по решениям проверяющих, а не независимая тестовая точность.",
                "Непроверенные случаи и «Недостаточно данных» не считаются отсутствием нарушения.",
                "Точность и полнота по кандидатам относятся только к проверенным строкам; найденное человеком действие в «Требует проверки» не является автоматическим подтверждением модели.",
                "Полнота событий вычисляется только для полностью просмотренных видео с окончательными решениями, проверенными интервалами и указанными группой источника, камерой и сессией. Неоднозначные пересекающиеся интервалы исключены.",
                "Нулевой знаменатель показан как отсутствие данных; временная точка не подменяет интервал для IoU.",
                "Brier и диаграмма надёжности используют только оценки всего нарушения и решения человека. Оценки видимого действия и завершения процедуры исключены.",
                "Все версии одного исходника и общей камеры остаются в одной группе. Разделение и обучение автоматически не выполняются.",
            ]}
