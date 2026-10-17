"""Local review catalog and evidence-backed display states, outside jury scoring."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from worker.scoring import create_initial_assessments, update_event_confidence, update_procedure_confidence


SCORING_NOTE = (
    "Events start at 0% and rise with modeled occurrence confidence. Procedures start at 100% "
    "and fall by modeled confidence that the correct protocol was completed. These are "
    "uncalibrated model estimates, not measured probabilities or automatic verdicts. "
    "Updates are scoped to source-linked episodes; repeated frames do not add confidence. "
    "The summary retains the highest remaining estimate across assessed episodes. "
    "Untriggered unknown windows mark incomplete coverage rather than create new pending procedures. "
    "Unknown applicability and unobserved/manual checks remain explicit."
)


def load_catalog(root: Path) -> dict:
    value = json.loads((root / "rules" / "catalog.json").read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("version"), str):
        raise ValueError("The local review catalog needs a version")
    create_initial_assessments(value)
    return value


def clip_assessments(catalog: dict, findings: list[dict], review_queue: list[dict],
                     duration_sec: float, *, confidence_observations=(), known_episode_ids=()) -> tuple[list[dict], list[dict]]:
    """Update continuous estimates from model evidence before the legal gate.

    Model observations, including uncertain event and correct-protocol estimates,
    change scores independently of final findings. Episode keys are assigned by
    the host from source-linked boundaries; unbounded windows stay independent.
    A procedure completion cannot clear a separate pending transaction. Final
    findings only add their separately gated confirmation and confidence.
    """
    rows = create_initial_assessments(catalog)
    by_id = {row["item_id"]: row for row in rows}
    items = {item["id"]: item for item in catalog["items"]}
    known_ids = set(known_episode_ids)
    for row in rows:
        row.update(scope="clip_event_summary" if row["scoring_kind"] == "event_occurrence"
                   else "per_episode_summary", automation=items[row["item_id"]]["automation"],
                   confirmed_occurrences=0, episode_count=0, pending_episode_count=0,
                   latest_episode_id=None, summary_policy="maximum_episode_risk",
                   known_episode_count=len(known_ids), unassessed_episode_count=len(known_ids),
                   episode_coverage_complete=False, unscoped_unknown_observation_count=0)
    for proposed in review_queue:
        row = by_id.get(proposed.get("type_id"))
        if row is not None:
            row["assessment"] = "unresolved"

    episodes = {}

    def episode_for(identifier, episode_id, start, end, certainty="observed_start"):
        key = (identifier, episode_id)
        if key not in episodes:
            fresh = create_initial_assessments([items[identifier]])[0]
            fresh.update(scope="episode_assessment", episode_id=episode_id, start_sec=start, end_sec=end,
                         scope_certainty=certainty, automation=items[identifier]["automation"],
                         latest_assessed_at_sec=end, confirmed_occurrences=0)
            episodes[key] = fresh
        episode = episodes[key]
        episode["start_sec"] = min(episode["start_sec"], start)
        episode["end_sec"] = max(episode["end_sec"], end)
        episode["latest_assessed_at_sec"] = max(episode["latest_assessed_at_sec"], end)
        return key, episode

    for observation in confidence_observations:
        identifier = observation["item_id"]
        if identifier not in items or items[identifier]["automation"] not in {"automatic_when_supported", "review_only"}:
            raise ValueError("A model confidence observation is outside the observable catalog")
        item = items[identifier]
        state = observation["assessment"]
        expected = "event_evidence" if item["scoring_kind"] == "event_occurrence" else "completion_evidence"
        if state not in {expected, "partial_support", "unresolved"}:
            raise ValueError("Confidence evidence does not match the catalog scoring kind")
        if state == "unresolved" and observation["confidence"] != 0:
            raise ValueError("Unresolved observations cannot supply positive confidence")
        if state == "partial_support" and (type(observation["confidence"]) not in (int, float)
                                           or not 0 < observation["confidence"] <= 1):
            raise ValueError("Partial support requires positive confidence")
        update = update_event_confidence if item["scoring_kind"] == "event_occurrence" else update_procedure_confidence
        certainty = observation.get("scope_certainty", "window_only")
        if (item["scoring_kind"] == "required_procedure" and state == "unresolved"
                and observation["applicability"] == "unknown" and certainty == "window_only"):
            # An unreadable/untriggered window is a coverage gap, not evidence
            # of a new applicable transaction whose procedure is still pending.
            # Validate its source data without adding it to episode aggregation.
            update(create_initial_assessments([item])[0], confidence=observation["confidence"],
                   observable=False, applicability=observation["applicability"],
                   evidence_refs=observation["evidence_refs"], duration_sec=duration_sec)
            row = by_id[identifier]
            row["unscoped_unknown_observation_count"] += 1
            if row["assessment"] == "unassessed":
                row["assessment"] = "unresolved"
            continue
        key, episode = episode_for(identifier, observation["episode_id"], observation["start_sec"],
                                   observation["end_sec"], certainty)
        episodes[key] = update(episode, confidence=observation["confidence"],
                               observable=observation["observable"] and state != "unresolved",
                               applicability=observation["applicability"], evidence_refs=observation["evidence_refs"],
                               duration_sec=duration_sec)

    for finding in findings:
        identifier = finding.get("type_id")
        row = by_id.get(identifier)
        # The broad manual catalog does not enlarge the detector's legal scope.
        if (row is None or row["scoring_kind"] != "event_occurrence"
                or items[identifier]["automation"] != "automatic_when_supported"):
            raise ValueError("A finding is outside the automatic event catalog")
        references = [{"filename": frame["filename"], "timestamp_sec": frame["timestamp_sec"]}
                      for frame in finding["evidence_frames"]]
        episode_id = finding.get("episode_id", "finding")
        # Unbounded verification can match only the overlapping window example,
        # never all observations that happen to share the same type.
        if not episode_id.startswith("tentative-start-"):
            matching = [episode for (rid, _), episode in episodes.items() if rid == identifier
                        and episode["start_sec"] <= finding["timecode_sec"] <= episode["end_sec"]]
            if matching:
                episode_id = min(matching, key=lambda e: e["end_sec"] - e["start_sec"])["episode_id"]
        key, episode = episode_for(identifier, episode_id, min(e["timestamp_sec"] for e in references),
                                   max(e["timestamp_sec"] for e in references), "verified_event")
        episode = update_event_confidence(episode, confidence=finding.get("model_confidence", 1.0),
                                          observable=True, applicability="applicable", evidence_refs=references,
                                          duration_sec=duration_sec, confirmed_finding=True)
        episode.update(timecode_sec=finding["timecode_sec"], confirmed_occurrences=episode["confirmed_occurrences"] + 1)
        episodes[key] = episode

    for row in rows:
        relevant = [episode for (identifier, _), episode in episodes.items() if identifier == row["item_id"]]
        if not relevant:
            continue
        # Largest event probability / largest residual procedure probability.
        selected = max(relevant, key=lambda episode: (episode["score"], episode["latest_assessed_at_sec"]))
        latest = max(relevant, key=lambda episode: episode["latest_assessed_at_sec"])
        scope = row["scope"]
        row.update(deepcopy(selected))
        row.update(scope=scope, episode_count=len(relevant), latest_episode_id=latest["episode_id"],
                   latest_episode_start_sec=latest["start_sec"], latest_episode_end_sec=latest["end_sec"],
                   latest_assessed_at_sec=latest["latest_assessed_at_sec"],
                   pending_episode_count=sum(episode["score"] > 0 for episode in relevant)
                   if row["scoring_kind"] == "required_procedure" else 0,
                   confirmed_occurrences=sum(episode["confirmed_occurrences"] for episode in relevant),
                   confirmed_finding=any(episode["confirmed_finding"] for episode in relevant),
                   summary_policy="maximum_episode_risk")
        assessed_ids = {episode["episode_id"] for episode in relevant}
        row.update(known_episode_count=len(known_ids), unassessed_episode_count=len(known_ids - assessed_ids),
                   episode_coverage_complete=bool(known_ids) and not known_ids - assessed_ids
                   and not row["unscoped_unknown_observation_count"]
                   and all(episode["scope_certainty"] == "observed_start" for episode in relevant))
    return rows, sorted(episodes.values(), key=lambda episode: (episode["start_sec"], episode["item_id"], episode["episode_id"]))
