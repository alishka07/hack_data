"""The visual taxonomy covers cases without extending automatic legal findings."""
from pathlib import Path

from worker.ledger import ObservationLedger
from worker.scenarios import load_scenarios, scenario_type_ids
from worker.transactions import build_transactions
from worker.video import FrameRef


ROOT = Path(__file__).resolve().parents[1]


def test_curated_visual_coverage_and_unmapped_source_are_explicit():
    scenarios = load_scenarios(ROOT)
    expected = {"previous_transaction_cash_exposed", "previous_cash_stored",
                "cash_intake_without_client_presence", "backoffice_processing_control",
                "keys_stamps_secure_handling", "vault_cash_segregation_storage",
                "personal_belongings_in_cash_unit", "personal_food_at_workstation",
                "workstation_departure_screen_left_on"}
    assert expected <= set(scenario_type_ids(scenarios))
    assert scenarios["previous_cash_stored"]["alias_of"] == "previous_transaction_cash_exposed"
    assert scenarios["vault_cash_segregation_storage"]["legal_source"] is None
    assert all(row["disposition"] == "review_only" for row in scenarios.values())
    assert scenarios["backoffice_processing_control"]["legal_source"]["paragraph"] == "42"


def test_food_presence_scenario_needs_food_location_and_work_context_without_eating():
    scenario = load_scenarios(ROOT)["personal_food_at_workstation"]
    assert scenario["visual_cues"] == ["food_at_workstation"]
    assert set(scenario["visual_checks"]) == {
        "food_item_visually_identified", "food_on_cashier_work_surface",
        "workstation_work_context_visible", "relevant_context_visible"}
    assert scenario["legal_source"] is None and scenario["related_legal_rule_ids"] == []
    assert scenario["disposition"] == "review_only"
    assert "without eating" in scenario["check_descriptions"]["food_on_cashier_work_surface"]
    assert "break room" in scenario["check_descriptions"]["workstation_work_context_visible"]
    assert "medicine" in scenario["check_descriptions"]["food_item_visually_identified"]


def test_all_visual_cues_are_valid_observations_without_new_transaction_conclusions():
    scenarios = load_scenarios(ROOT)
    kinds = sorted({kind for row in scenarios.values() for kind in row["visual_cues"]})
    ledger = ObservationLedger()
    for number, kind in enumerate(kinds):
        frame = FrameRef(number, float(number), Path("synthetic.jpg"))
        ledger.add(number, [frame], {"visibility": "clear", "visible_facts": [
            {"kind": kind, "state": "observed", "note": "Synthetic positive visual cue", "frame_indices": [0]}],
            "transaction": {"kind": "unknown", "phase": "unknown", "frame_indices": []}})
    data = ledger.as_dict()
    assert {row["kind"] for row in data["facts"]} == set(kinds)
    transactions = build_transactions(ledger)
    assert all(row.get("disposition") != "finding" for row in transactions.get("proposals", []))
    assert not any("ownership" in row for row in data["facts"])


def test_screen_departure_scenario_requires_departure_then_same_screen_on():
    scenario = load_scenarios(ROOT)["workstation_departure_screen_left_on"]
    assert scenario["visual_cues"] == ["computer_screen_visible", "workstation_departure",
                                       "computer_screen_on_unattended"]
    assert scenario["legal_source"] is None and scenario["related_legal_rule_ids"] == []
    assert scenario["disposition"] == "review_only"
    descriptions = scenario["check_descriptions"]
    assert "actual exit" in descriptions["workstation_departure_observed"]
    assert "empty chair alone" in descriptions["workstation_unattended_after_departure"]
    screen = descriptions["same_screen_visibly_on_after_departure"]
    assert "same identified" in screen and "locked but lit" in screen
    assert "standby" in screen and "closed-lid" in screen
    assert "before/departure/after" in descriptions["departure_screen_sequence_visible"]
