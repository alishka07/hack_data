"""Scene proposals are bounded source guidance, never authoritative room types."""
from pathlib import Path

import pytest

from worker.scene import ZONES, collect_scene, detail_views, interaction_box, scene_schema
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
    assert len(result["zones"]) == len(ZONES) == 8
    assert result["zones"][0]["refs"] == [{"source_index": 20, "timestamp_sec": 2.}]
    assert sum(row["status"] == "unknown" for row in result["zones"]) == 7
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


def test_computer_detail_uses_its_source_box_and_keeps_full_scene_occupancy_context():
    from dataclasses import replace

    source = frames()
    computer_box = [.55, .2, .8, .45]
    scene = collect_scene([region(), region(zone="computer", bbox=computer_box)], source)

    class Cache:
        def crop(self, selected, box, view):
            assert box == computer_box
            assert view == "computer detail; full scene supplied separately"
            return [replace(frame, crop_bbox=tuple(box), view=view) for frame in selected]

    result = detail_views(Cache(), source, scene, 2., preferred_zone="computer", max_frames=4)
    assert all(frame in result for frame in source)
    crops = [frame for frame in result if frame.crop_bbox is not None]
    assert len(crops) == 2
    assert {(frame.index, frame.timestamp) for frame in crops} == {(10, 1.), (20, 2.)}
    assert interaction_box(scene) == [.1, .2, .8, .6]  # default counter behavior survives
    assert interaction_box(scene, zone="computer") == computer_box


def test_unknown_or_previous_camera_computer_never_substitutes_counter_as_device_crop():
    class NoCrop:
        def crop(self, *args):
            pytest.fail("An unknown or previous camera device box cannot establish screen pixels")

    source = frames()
    scene = collect_scene([region()], source)
    assert detail_views(NoCrop(), source, scene, 2., preferred_zone="computer") == source
    scene = collect_scene([region(zone="computer")], source)
    scene["scene_change_timestamps"] = [3.]
    assert detail_views(NoCrop(), source, scene, 3.5, preferred_zone="computer") == source
    with pytest.raises(ValueError):
        detail_views(NoCrop(), source, scene, 2., preferred_zone="invented")
