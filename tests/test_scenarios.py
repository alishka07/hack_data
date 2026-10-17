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
                "personal_belongings_in_cash_unit", "cashier_eating_at_workstation"}
    assert expected <= set(scenario_type_ids(scenarios))
    assert scenarios["previous_cash_stored"]["alias_of"] == "previous_transaction_cash_exposed"
    assert scenarios["vault_cash_segregation_storage"]["legal_source"] is None
    assert all(row["disposition"] == "review_only" for row in scenarios.values())
    assert scenarios["backoffice_processing_control"]["legal_source"]["paragraph"] == "42"


def test_eating_scenario_preserves_positive_consumption_and_work_context_requirements():
    scenario = load_scenarios(ROOT)["cashier_eating_at_workstation"]
    assert scenario["visual_cues"] == ["food_consumption"]
    assert set(scenario["visual_checks"]) == {
        "food_consumption_visibly_observed", "person_operating_cashier_workstation",
        "workstation_work_context_visible", "relevant_context_visible"}
    assert scenario["legal_source"] is None and scenario["related_legal_rule_ids"] == []
    assert scenario["disposition"] == "review_only"
    assert "bitten or chewed" in scenario["check_descriptions"]["food_consumption_visibly_observed"]
    assert "break room" in scenario["check_descriptions"]["workstation_work_context_visible"]


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
