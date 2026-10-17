"""Temporal proposals from visible geometry, never automatic rule findings.

Durations describe a sequence of sampled observations. Missing detections,
proposed scene geometry and wrist proximity cannot establish absence, a role,
ownership, contact or authorization. Every proposal must be verified visually.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os

from worker.video import AnalysisError, VideoInfo


@dataclass(frozen=True)
class EventConfig:
    dwell_sec: float = 10.
    empty_workstation_sec: float = 3.
    multiple_people_sec: float = 1.
    unattended_sec: float = 10.
    person_max_gap_sec: float = .8
    object_max_gap_sec: float = 6.
    hand_margin: float = .025
    unattended_margin: float = .12

    @classmethod
    def from_env(cls):
        try:
            result = cls(**{name: float(os.getenv("FORTE_EVENT_" + name.upper(), str(default)))
                            for name, default in asdict(cls()).items()})
            if any(not math.isfinite(v) or v <= 0 for v in asdict(result).values()):
                raise ValueError()
        except (ValueError, TypeError) as exc:
            raise AnalysisError("event_config_invalid", "Event durations and distances must be finite and positive") from exc
        return result


def _box(value):
    return (isinstance(value, (list, tuple)) and len(value) == 4
            and all(type(v) in (int, float) and math.isfinite(v) for v in value)
            and 0 <= value[0] < value[2] <= 1 and 0 <= value[1] < value[3] <= 1)


def _center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _distance(point, box):
    return math.hypot(max(box[0] - point[0], 0., point[0] - box[2]),
                      max(box[1] - point[1], 0., point[1] - box[3]))


def _iou(a, b):
    intersection = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union else 0.


def propose_events(report: dict, info: VideoInfo, config: EventConfig | None = None) -> dict:
    config = config or EventConfig.from_env()
    changes = sorted(row["timestamp_sec"] for row in report.get("motion", []) if row.get("scene_change_suspected"))

    def segment(timestamp):
        return sum(point <= timestamp for point in changes)

    def checked(record):
        index, timestamp = record.get("source_index"), record.get("timestamp_sec")
        if (type(index) is not int or not 0 <= index < len(info.pts)
                or type(timestamp) not in (int, float) or not math.isfinite(timestamp)
                or abs(info.pts[index] - timestamp) > 1e-6):
            raise AnalysisError("event_source_invalid", "An event observation does not match source presentation timestamps")
        return record

    detections = [checked(row) for row in report.get("detections", []) if _box(row.get("bbox"))]
    people = [row for row in detections if row.get("kind") == "person"]
    objects = [row for row in detections if row.get("kind") in {"cash", "bag", "phone", "drawer", "cash_drawer"}]
    frames = [checked(row) for row in report.get("frame_samples", [])]
    if not frames:
        # Older/injected detectors have no empty-frame visibility information.
        frames = list({row["source_index"]: {"source_index": row["source_index"],
                       "timestamp_sec": row["timestamp_sec"], "person_detection_available": False}
                       for row in people}.values())
    frames.sort(key=lambda row: row["timestamp_sec"])
    zones = []
    for zone in report.get("scene", {}).get("zones", []):
        name = zone.get("name", zone.get("zone"))
        sources = zone.get("source_indices", [])
        if not sources:
            sources = [checked(ref)["source_index"] for ref in zone.get("refs", [])]
        if _box(zone.get("bbox")) and sources and zone.get("status") not in {"unknown", "contradicted"}:
            if any(type(i) is not int or not 0 <= i < len(info.pts) for i in sources):
                raise AnalysisError("event_source_invalid", "A zone references an unavailable source frame")
            zones.append({"name": name, "bbox": zone["bbox"], "segments": {segment(info.pts[i]) for i in sources}})

    def active_zones(timestamp):
        return [zone for zone in zones if zone["segments"] == {segment(timestamp)}]

    proposals, occupancy, last_person, dwell, occupied_before, empty, multiple = [], [], {}, {}, set(), {}, {}

    def emit(reason, first, last, **extra):
        indices = sorted({first["source_index"], last["source_index"]})
        proposals.append({"reason": reason, "timecode_sec": last["timestamp_sec"],
                          "start_sec": max(0., first["timestamp_sec"] - 1.),
                          "end_sec": min(info.duration, last["timestamp_sec"] + 1.),
                          "source_indices": indices,
                          "refs": [{"source_index": i, "timestamp_sec": info.pts[i]} for i in indices],
                          "sampled_duration_sec": last["timestamp_sec"] - first["timestamp_sec"],
                          "status": "unconfirmed_trigger", "confidence_calibrated": False,
                          "limitations": "Sampled geometry only; role, absence, contact and ownership require visual verification.",
                          **extra})

    by_source = {}
    for person in people:
        by_source.setdefault(person["source_index"], []).append(person)
    previous_frame = None
    for frame in frames:
        timestamp = frame["timestamp_sec"]
        if previous_frame and (timestamp - previous_frame["timestamp_sec"] > config.person_max_gap_sec
                               or segment(timestamp) != segment(previous_frame["timestamp_sec"])):
            dwell.clear(); empty.clear(); multiple.clear(); last_person.clear(); occupied_before.clear()
        present = by_source.get(frame["source_index"], [])
        zone_people = {zone["name"]: [] for zone in active_zones(timestamp)}
        for person in present:
            point = _center(person["bbox"])
            current = {zone["name"] for zone in active_zones(timestamp) if _distance(point, zone["bbox"]) == 0}
            track = person.get("track_id")
            if not track:
                continue
            previous = last_person.get(track)
            adjacent = (previous is not None and timestamp - previous["timestamp_sec"] <= config.person_max_gap_sec
                        and segment(timestamp) == segment(previous["timestamp_sec"])
                        and person.get("continuity") == "adjacent_detection")
            if adjacent:
                for name in sorted(current - previous["zones"]):
                    emit("zone_entry", previous, person, zone=name, track_id=track)
                for name in sorted(previous["zones"] - current):
                    emit("zone_exit", previous, person, zone=name, track_id=track)
            for name in current:
                zone_people[name].append(track)
                key = (track, name)
                if not adjacent or name not in previous["zones"] or key not in dwell:
                    dwell[key] = {"first": person, "emitted": False}
                run = dwell[key]
                if not run["emitted"] and timestamp - run["first"]["timestamp_sec"] >= config.dwell_sec:
                    emit("zone_dwell", run["first"], person, zone=name, track_id=track)
                    run["emitted"] = True
            for key in [key for key in dwell if key[0] == track and key[1] not in current]:
                del dwell[key]
            last_person[track] = {**person, "zones": current}
        for name, tracks in zone_people.items():
            occupancy.append({"zone": name, "source_index": frame["source_index"], "timestamp_sec": timestamp,
                              "detected_people": len(set(tracks)), "role": "unknown", "visibility": "unknown",
                              "semantics": "Detection count in a proposed zone; zero does not establish absence."})
            key = (segment(timestamp), name)
            if tracks:
                occupied_before.add(key)
                empty.pop(key, None)
            elif (name == "workstation" and key in occupied_before and frame.get("person_detection_available")):
                run = empty.setdefault(key, {"first": frame, "emitted": False})
                if not run["emitted"] and timestamp - run["first"]["timestamp_sec"] >= config.empty_workstation_sec:
                    emit("cashier_absence_candidate", run["first"], frame, zone=name,
                         observation="No person detections in a previously occupied workstation; absence remains unknown")
                    run["emitted"] = True
            if name == "cash_zone" and len(set(tracks)) >= 2:
                run = multiple.setdefault(key, {"first": frame, "emitted": False})
                if not run["emitted"] and timestamp - run["first"]["timestamp_sec"] >= config.multiple_people_sec:
                    emit("multiple_people_in_cash_zone", run["first"], frame, zone=name,
                         observation="Multiple anonymous detections; permissions and roles unknown")
                    run["emitted"] = True
            else:
                multiple.pop(key, None)
        previous_frame = frame

    # Proximity is assessed only on exactly the same supplied source frame.
    interactions, last_interaction = [], {}
    for pose in report.get("poses", []):
        checked(pose)
        for point in pose.get("keypoints", []):
            if point.get("name") not in {"left_wrist", "right_wrist"} or point.get("status") != "supported":
                continue
            xy = (point.get("x"), point.get("y"))
            if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in xy):
                continue
            for obj in objects:
                if obj["source_index"] != pose["source_index"] or _distance(xy, obj["bbox"]) > config.hand_margin:
                    continue
                key = (pose.get("track_id"), obj["kind"], point["name"])
                prior = last_interaction.get(key)
                if prior is not None and pose["timestamp_sec"] - prior < 1.:
                    continue
                emit("hand_object_proximity", pose, pose, track_id=pose.get("track_id"),
                     object_kind=obj["kind"], object_detection_id=obj["id"], wrist=point["name"])
                interactions.append({"source_index": pose["source_index"], "timestamp_sec": pose["timestamp_sec"],
                                     "person_track_id": pose.get("track_id"), "object_detection_id": obj["id"],
                                     "state": "proximity_only", "contact": "not_visible"})
                last_interaction[key] = pose["timestamp_sec"]

    # Sparse object association is temporary geometry, independent of person IDs.
    object_runs, next_object = {}, 0
    objects_by_source = {}
    for obj in objects:
        objects_by_source.setdefault(obj["source_index"], []).append(obj)
    object_samples = report.get("object_samples", [])
    object_indices = {checked(row)["source_index"] for row in object_samples} if object_samples else set(objects_by_source)
    previous_object_timestamp = None
    for index in sorted(object_indices):
        timestamp = info.pts[index]
        if previous_object_timestamp is not None and (timestamp - previous_object_timestamp > config.object_max_gap_sec
                or segment(timestamp) != segment(previous_object_timestamp)):
            object_runs.clear()
        updated, used = {}, set()
        for obj in objects_by_source.get(index, []):
            if obj["kind"] not in {"cash", "bag", "phone"}:
                continue
            matches = [(key, run) for key, run in object_runs.items() if key not in used
                       and run["last"]["kind"] == obj["kind"] and _iou(run["last"]["bbox"], obj["bbox"]) >= .5]
            if matches:
                key, run = max(matches, key=lambda pair: _iou(pair[1]["last"]["bbox"], obj["bbox"]))
                used.add(key)
            else:
                next_object += 1
                key, run = f"object-{next_object}", {"first": obj, "last": obj, "emitted": False, "samples": 0}
            # Require same-frame person processing; never interpret missing analysis as no nearby person.
            inspected = next((frame for frame in frames if frame["source_index"] == index), None)
            near = any(_distance(_center(obj["bbox"]), person["bbox"]) <= config.unattended_margin
                       for person in by_source.get(index, []))
            if not inspected or not inspected.get("person_detection_available") or near:
                # An attended or unassessed sample cannot start an unattended
                # interval. Keep only the geometry needed for the next match.
                updated[key] = {"first": None, "last": obj, "emitted": False, "samples": 0}
                continue
            if run["first"] is None:
                run["first"] = obj
            run["samples"] += 1
            run["last"] = obj
            if (not run["emitted"] and run["samples"] >= 3
                    and timestamp - run["first"]["timestamp_sec"] >= config.unattended_sec):
                emit("object_unattended_candidate", run["first"], obj, object_track_id=key, object_kind=obj["kind"],
                     observation="Repeated stationary object detections without nearby person detections; ownership and continuous attendance unknown")
                run["emitted"] = True
            updated[key] = run
        object_runs = updated
        previous_object_timestamp = timestamp
    proposals.sort(key=lambda row: (row["timecode_sec"], row["reason"]))
    return {"version": "1.0", "proposals": proposals, "occupancy": occupancy, "interactions": interactions,
            "config": asdict(config), "calibrated": False,
            "limitations": ["All geometry-derived events are unconfirmed proposals",
                            "Dwell durations cover sampled detections, not uninterrupted visibility",
                            "No role, permission, cash ownership or actual absence is inferred"]}
