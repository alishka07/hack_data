"""Source-linked observations, without legal judgments or inferred absences."""
from __future__ import annotations

from copy import deepcopy
import math

from worker.video import AnalysisError, FrameRef


ACTION_KINDS = (
    "cash_received", "cash_count_manual", "cash_count_machine", "cash_stored",
    "cash_prepared", "cash_handed_over", "workstation_departure", "workstation_return",
    "client_arrival", "client_departure", "transaction_start", "transaction_end",
    "personal_object_interaction", "access_boundary_crossing", "other_observable_action",
    "cash_on_work_surface", "cash_moved_between_storage", "cash_storage_opened",
    "cash_storage_closed", "key_handling", "stamp_handling", "key_or_stamp_left_on_surface",
    "bag_or_container_handling", "phone_handling", "cash_processing_activity",
    "person_enters_processing_area", "person_leaves_processing_area", "food_consumption", "food_at_workstation",
    "computer_screen_visible", "computer_screen_on_unattended",
)


def _invalid() -> None:
    raise AnalysisError("model_schema_invalid", "A local observation contained invalid source references or fields")


def _number(value: object) -> bool:
    return not isinstance(value, bool) and (isinstance(value, int)
                                           or isinstance(value, float) and math.isfinite(value))


class ObservationLedger:
    """Keep explicit window facts; tentative episode labels carry no identity claim."""

    def __init__(self) -> None:
        self._windows: dict[int, dict] = {}

    @staticmethod
    def _references(indices: object, frames: list[FrameRef], *, allow_empty: bool = False) -> list[dict]:
        if not isinstance(indices, list) or len(indices) > 32 or (not indices and not allow_empty):
            _invalid()
        references: dict[int, dict] = {}
        for index in indices:
            if type(index) is not int or not 0 <= index < len(frames):
                _invalid()
            frame = frames[index]
            references[frame.index] = {"source_index": frame.index, "timestamp_sec": frame.timestamp}
        return sorted(references.values(), key=lambda ref: (ref["timestamp_sec"], ref["source_index"]))

    @staticmethod
    def _record(window_id: int, kind: str, state: str, note: str, references: list[dict], **fields) -> dict:
        return {
            "window_id": window_id, "kind": kind, "state": state, "note": note,
            "time_sec": references[0]["timestamp_sec"] if references else None,
            "source_indices": [ref["source_index"] for ref in references],
            "refs": references, **fields,
        }

    def add(self, window_id: int, frames: list[FrameRef], answer: dict) -> None:
        """Validate the whole observation before replacing this window atomically."""
        if type(window_id) is not int or window_id < 0 or not isinstance(frames, list):
            _invalid()
        source_times: dict[int, float] = {}
        for frame in frames:
            if (not isinstance(frame, FrameRef) or type(frame.index) is not int or frame.index < 0
                    or not _number(frame.timestamp) or frame.timestamp < 0):
                _invalid()
            if frame.index in source_times and source_times[frame.index] != frame.timestamp:
                _invalid()
            source_times[frame.index] = frame.timestamp
        if (not isinstance(answer, dict) or not isinstance(answer.get("visibility"), str)
                or answer["visibility"] not in {"clear", "partial", "unusable"}):
            _invalid()
        supplied = answer.get("visible_facts")
        if not isinstance(supplied, list) or len(supplied) > 32:
            _invalid()
        facts: list[dict] = []
        for item in supplied:
            if not isinstance(item, dict):
                _invalid()
            kind, state, note = item.get("kind"), item.get("state"), item.get("note")
            if (not isinstance(kind, str) or kind not in ACTION_KINDS
                    or not isinstance(state, str) or state not in {"observed", "uncertain"}
                    or not isinstance(note, str) or len(note) > 320):
                _invalid()
            references = self._references(item.get("frame_indices"), frames)
            facts.append(self._record(window_id, kind, state, note, references))
        transaction = answer.get("transaction")
        if not isinstance(transaction, dict):
            _invalid()
        kind, phase = transaction.get("kind"), transaction.get("phase")
        if (not isinstance(kind, str) or kind not in {"intake", "payout", "preparation", "unknown"}
                or not isinstance(phase, str) or phase not in {"active", "complete", "unknown"}):
            _invalid()
        references = self._references(transaction.get("frame_indices"), frames,
                                      allow_empty=(kind == "unknown" and phase == "unknown"))
        cue = self._record(window_id, "transaction_cue",
                           "uncertain" if "unknown" in (kind, phase) else "observed",
                           f"Window transaction cue: {kind}; phase: {phase}. Applies only to its referenced observations.",
                           references, transaction_kind=kind, phase=phase)
        self._windows[window_id] = {
            "window_id": window_id,
            "visibility": answer["visibility"],
            "sampled_extent_sec": [min(source_times.values()), max(source_times.values())] if source_times else [],
            "facts": facts, "transaction": cue,
        }

    def _records(self) -> list[dict]:
        records = []
        for window in self._windows.values():
            records.extend(window["facts"])
            # An unanchored unknown cue is preserved privately but is not a
            # source-linked fact that could establish transaction continuity.
            if window["transaction"]["refs"]:
                records.append(window["transaction"])
        return sorted(records, key=lambda record: (record["time_sec"], record["window_id"], record["kind"]))

    def relevant_facts(self, start: float, end: float, limit: int = 8) -> list[dict]:
        """Return bounded context, preserving all records in as_dict()."""
        if not _number(start) or not _number(end) or start > end or type(limit) is not int or limit < 0:
            raise ValueError("Context interval and limit must be valid")
        if limit == 0:
            return []
        records = [record for record in self._records()
                   if record["refs"][0]["timestamp_sec"] <= end
                   and record["refs"][-1]["timestamp_sec"] >= start]
        # Even spacing retains pre/action/post context rather than only the
        # latest window. This context limit never removes stored observations.
        limit = min(limit, 32)
        if len(records) > limit:
            if limit == 1:
                middle = (start + end) / 2
                records = [min(records, key=lambda record: abs(record["time_sec"] - middle))]
            else:
                records = [records[round(i * (len(records) - 1) / (limit - 1))] for i in range(limit)]
        return deepcopy(records)

    def episode_id(self, anchor: float) -> str:
        """Segment by observed boundary cues; never infer identity or missing steps."""
        if not _number(anchor) or anchor < 0:
            raise ValueError("Episode anchor must be a finite source time")
        boundaries = [record for record in self._records()
                      if record["state"] == "observed"
                      and record["kind"] in {"transaction_start", "transaction_end"}
                      and (record["time_sec"] <= anchor if record["kind"] == "transaction_start"
                           else record["time_sec"] < anchor)]
        if not boundaries:
            return "tentative-unbounded"
        latest = max(boundaries, key=lambda record: (record["time_sec"], record["kind"] == "transaction_start"))
        label = "start" if latest["kind"] == "transaction_start" else "after-end"
        return f"tentative-{label}-{latest['source_indices'][0]:010d}"

    def as_dict(self) -> dict:
        return deepcopy({
            "version": "2.0",
            "semantics": "Observations and tentative boundaries only; no person/cash identity, continuity proof, omitted action, or legal conclusion is inferred.",
            "windows": [self._windows[key] for key in sorted(self._windows)],
            "facts": [record for record in self._records() if record["kind"] != "transaction_cue"],
            "transaction_cues": [window["transaction"] for _, window in sorted(self._windows.items())],
        })
