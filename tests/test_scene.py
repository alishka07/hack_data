"""Scene proposals are bounded source guidance, never authoritative room types."""
from pathlib import Path

import pytest

from worker.scene import collect_scene, interaction_box, scene_schema
from worker.runtime import _validate_json_schema
from worker.video import AnalysisError, FrameRef


def frames():
    return [FrameRef(10, 1., Path("unused")), FrameRef(20, 2., Path("unused"))]


def region(**updates):
    return {"zone": "counter", "status": "supported", "bbox": [.1, .2, .8, .6], "frame_indices": [1], **updates}


def test_source_binding_unknown_zones_and_crop_guidance():
    value = [region()]
    _validate_json_schema(value, scene_schema(2))
    result = collect_scene(value, frames())
    assert len(result["zones"]) == 7
    assert result["zones"][0]["refs"] == [{"source_index": 20, "timestamp_sec": 2.}]
    assert sum(row["status"] == "unknown" for row in result["zones"]) == 6
    assert interaction_box(result) == [.1, .2, .8, .6]
    assert not result["calibrated"]


@pytest.mark.parametrize("value", [region(frame_indices=[]), region(frame_indices=[True]), region(frame_indices=[2]),
                                   region(bbox=[0, 0, 0, 1]), region(bbox=[0, 0, float("nan"), 1]),
                                   region(bbox=[False, 0, 1, 1]), region(status="unknown"), region(zone="vault")])
def test_invalid_or_unsubstantiated_geometry_fails(value):
    with pytest.raises(AnalysisError):
        collect_scene([value], frames())


def test_unknown_geometry_and_missing_scene_do_not_fabricate_coordinates():
    result = collect_scene([region(status="unknown", bbox=[], frame_indices=[])], frames())
    assert interaction_box(result) is None
    assert all(row["status"] == "unknown" for row in result["zones"])
    assert all(row["refs"] == [] for row in collect_scene([], [])["zones"])


def test_duplicate_regions_are_invalid_and_cuts_prevent_reuse():
    with pytest.raises(AnalysisError):
        collect_scene([region(), region()], frames())
    result = collect_scene([region()], frames())
    result["scene_change_timestamps"] = [3.]
    assert interaction_box(result, 2.5) is not None
    assert interaction_box(result, 3.5) is None


def test_optional_unusable_geometry_can_only_be_downgraded_to_unknown():
    for invalid in (region(bbox=[0, 0, 0, 0]), region(frame_indices=[]), region(status="unknown")):
        result = collect_scene([invalid], frames(), strict=False)
        assert result["unusable_region_estimates"] == 1
        assert interaction_box(result) is None
        assert all(row["status"] == "unknown" and row["bbox"] == [] and row["refs"] == [] for row in result["zones"])


def test_tolerating_geometry_never_tolerates_foreign_sources_or_structure():
    for invalid in (region(frame_indices=[99]), region(zone="invented"), {"zone": "counter"}):
        with pytest.raises(AnalysisError):
            collect_scene([invalid], frames(), strict=False)
