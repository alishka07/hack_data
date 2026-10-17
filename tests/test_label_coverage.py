from scripts.audit_label_coverage import summarize_records
import pytest


def test_sparse_hints_keep_negative_rows_negative_and_missing_unknown():
    report = summarize_records([
        {"binary_label": 1, "observed_categories": ["some_rule", "some_rule"]},
        {"binary_label": 0, "observed_categories": ["some_rule"]},
        {"binary_label": 1, "observed_categories": []},
    ], {"items": [{"id": "some_rule", "automation": "manual"}]}, {"rules": []})
    assert report["counts"] == {"label_rows": 3, "positive_rows": 2, "negative_rows": 1,
                                "category_ids": 1, "positive_rows_without_category": 1,
                                "negative_rows_without_category": 0,
                                "hinted_types_with_visual_detection_paths": 0,
                                "registered_visual_scenarios": 0, "legal_automatic_rules": 0}
    assert report["categories"][0]["positive_rows"] == 1
    assert report["categories"][0]["negative_rows"] == 1
    assert report["missing_catalog_ids"] == []
    assert report["missing_enabled_registry_ids"] == ["some_rule"]


def test_catalog_entry_alone_does_not_establish_automatic_coverage():
    report = summarize_records([
        {"binary_label": 1, "observed_categories": ["known_rule", "missing_rule"]},
    ], {"items": [{"id": "known_rule", "automation": "manual"}]}, {"rules": [
        {"id": "known_rule", "enabled": True, "auto_emit": False,
         "automation": "review_only", "required_checks": ["role_known"], "action_checks": ["action_visible"]},
    ]})
    known = next(row for row in report["categories"] if row["type_id"] == "known_rule")
    assert known["registry_enabled"] is True
    assert known["auto_emit"] is False
    assert known["required_checks"] == ["role_known"]
    assert report["missing_catalog_ids"] == ["missing_rule"]
    assert report["missing_enabled_registry_ids"] == ["missing_rule"]


def test_visual_paths_are_counted_separately_from_legal_rules():
    report = summarize_records([
        {"binary_label": 1, "observed_categories": ["review_rule", "alias_rule"]},
    ], {"items": []}, {"rules": [
        {"id": "other_legal_rule", "enabled": True, "auto_emit": True},
    ]}, {"scenarios": [
        {"id": "review_rule", "visual_checks": ["object_visible"], "disposition": "review_only"},
        {"id": "alias_rule", "visual_checks": ["storage_visible"], "disposition": "review_only", "alias_of": "review_rule"},
    ]})
    assert report["summary"] == "2/2 hinted types with visual detection paths; 1 legal automatic rules"
    assert report["counts"]["hinted_types_with_visual_detection_paths"] == 2
    assert report["counts"]["legal_automatic_rules"] == 1
    assert report["missing_visual_detection_path_ids"] == []
    assert all(row["auto_emit"] is False for row in report["categories"])
    alias = next(row for row in report["categories"] if row["type_id"] == "alias_rule")
    assert alias["visual_alias_of"] == "review_rule"


def test_empty_or_unapproved_visual_scenario_is_not_a_detection_path():
    report = summarize_records([
        {"binary_label": 1, "observed_categories": ["empty_rule", "unapproved_rule"]},
    ], {"items": []}, {"rules": []}, {"scenarios": [
        {"id": "empty_rule", "visual_checks": [], "disposition": "review_only"},
        {"id": "unapproved_rule", "visual_checks": ["visible"], "disposition": "automatic"},
    ]})
    assert report["counts"]["hinted_types_with_visual_detection_paths"] == 0
    assert report["missing_visual_detection_path_ids"] == ["empty_rule", "unapproved_rule"]


@pytest.mark.parametrize("record", [
    {"binary_label": None, "observed_categories": []},
    {"binary_label": True, "observed_categories": []},
    {"binary_label": 1, "observed_categories": ["free form description"]},
])
def test_invalid_labels_are_rejected(record):
    with pytest.raises(ValueError):
        summarize_records([record], {"items": []}, {"rules": []})
