"""Conservative, source-linked Russian descriptions for exported reviews."""
from __future__ import annotations

import math


TITLES = {
    "personal_food_at_workstation": "Личные вещи на рабочем месте: продукты питания",
    "workstation_departure_screen_left_on": "Уход с рабочего места при включённом экране",
    "active_intake_departure": "Уход во время незавершённого приёма наличных",
    "cash_intake_without_client_presence": "Приём наличных без присутствия клиента",
    "previous_transaction_cash_exposed": "Наличные предыдущего клиента вне хранения",
    "intake_banknotes_counted": "Пересчёт принимаемых банкнот",
    "intake_coins_counted": "Пересчёт принимаемых монет",
    "intake_client_presence_required": "Присутствие клиента при приёме наличных",
    "previous_cash_stored": "Хранение наличных предыдущей операции",
    "loose_cash_issued_without_observed_count": "Пересчёт выдаваемых наличных",
    "damaged_incomplete_strap_count": "Пересчёт повреждённой или неполной пачки",
    "incomplete_coin_sack_count": "Пересчёт неполного мешка монет",
    "payout_bank_packaging": "Упаковка выдаваемых наличных",
    "client_recount_supervision": "Наблюдение за пересчётом клиентом",
    "backoffice_processing_control": "Контроль обработки наличных",
    "intake_authenticity_validity_checks": "Проверка подлинности и платёжности",
    "machine_security_features": "Проверка защитных признаков оборудованием",
    "intake_client_identification": "Идентификация клиента при приёме",
    "payout_client_identification": "Идентификация клиента при выдаче",
    "intake_confirmation_document": "Документ о приёме наличных",
    "payout_confirmation_document": "Документ о выдаче наличных",
    "personal_belongings_in_cash_unit": "Личные вещи в кассовой зоне",
    "personal_cash_valuables_in_cash_unit": "Личные деньги и ценности в кассовой зоне",
    "personal_device_exception_eligibility": "Допустимость личного устройства",
    "vault_entry_authorization": "Допуск в хранилище",
    "vault_other_person_accompaniment": "Сопровождение посетителей хранилища",
    "keys_stamps_secure_handling": "Обращение с ключами и штампами",
    "cash_unit_isolation": "Изоляция кассового узла",
    "cashier_room_lockable_entrance": "Запираемый вход в кассовое помещение",
    "cash_transfer_device_serviceable": "Исправность устройства передачи наличных",
    "cash_transfer_controls_inside": "Расположение управления передачей наличных",
    "counting_room_lockable_entrance": "Запираемый вход в помещение пересчёта",
    "collection_loading_access_restriction": "Ограничение доступа при инкассации",
    "collection_loading_view_restriction": "Ограничение обзора при инкассации",
    "vault_cash_segregation_storage": "Раздельное хранение наличных",
    "cashier_eating_at_workstation": "Приём пищи на рабочем месте",
}
SCOPES = {"joint_checks": "Все условия правила", "action_only": "Только видимое действие",
          "event_occurrence": "Наличие события", "procedure_completion": "Выполнение процедуры"}
HUMAN_DECISIONS = {"confirmed": "Подтверждено проверяющим", "dismissed": "Отклонено проверяющим",
                   "uncertain": "Недостаточно данных", "unreviewed": "Ещё не проверено человеком"}
CHECK_LABELS = {
    "actual_workstation_departure": "Уход с рабочего места", "workstation_departure": "Уход с рабочего места",
    "workstation_departure_observed": "Последовательность ухода с рабочего места",
    "cash_acceptance_or_count_observed": "Видимый приём или пересчёт наличных",
    "cash_form_visible": "Видимость формы и упаковки наличных", "cash_handling_observed": "Обращение с наличными",
    "cash_handover_observed": "Передача наличных", "cash_intake_established": "Установленный приём наличных",
    "cash_on_work_surface_observed": "Наличные вне контейнера", "cash_outside_designated_storage": "Наличные вне места хранения",
    "cash_payout_established": "Установленная выдача наличных", "cash_persistence_visually_supported": "Наличные в последовательных кадрах",
    "cash_processing_observed": "Обработка наличных", "cash_storage_relationship_observed": "Расположение наличных у хранилища",
    "client_absence_visually_supported": "Отсутствие клиента в обозреваемой зоне", "client_area_visible": "Видимость клиентской зоны",
    "client_cash_lineage_established": "Связь наличных с операцией клиента", "client_demonstrably_absent": "Подтверждение отсутствия клиента",
    "client_service_active": "Продолжение обслуживания клиента", "client_service_transition_observed": "Переход к другой операции",
    "complete_preparation_to_handover_visible": "Видимость подготовки и передачи целиком",
    "computer_at_workstation_identified": "Опознание компьютера на рабочем месте",
    "departure_screen_sequence_visible": "Видимость ухода и последующего состояния экрана",
    "food_consumption_visibly_observed": "Видимый приём пищи", "food_item_visually_identified": "Опознание продуктов питания",
    "food_on_cashier_work_surface": "Продукты на рабочем столе", "personal_food_on_work_table": "Продукты на рабочем столе",
    "handling_or_placement_observed": "Перемещение или размещение предмета", "key_or_stamp_visually_supported": "Опознание ключа или штампа",
    "loose_cash_established": "Неупакованные наличные", "next_distinct_client_transaction_active": "Начало следующей операции другого клиента",
    "no_earlier_or_offscreen_count_possible": "Исключение более раннего или скрытого пересчёта",
    "no_manual_or_machine_count_observed": "Отсутствие видимого ручного или машинного пересчёта",
    "non_cash_object_visually_supported": "Опознание предмета, не относящегося к наличным",
    "object_cash_area_relationship_visible": "Расположение предмета в кассовой зоне", "object_continuity_established": "Непрерывность наблюдения одного предмета",
    "packaging_exception_excluded": "Исключение допустимой выдачи в упаковке", "person_operating_cashier_workstation": "Работа человека на кассовом месте",
    "previous_client_cash_identified": "Идентификация наличных предыдущего клиента", "previous_transaction_completed": "Завершение предыдущей операции",
    "processing_area_visible": "Видимость зоны обработки наличных", "relevant_context_visible": "Видимость необходимого контекста",
    "same_screen_visibly_on_after_departure": "Включённый экран после ухода", "service_not_completed": "Незавершённое обслуживание",
    "storage_container_visible": "Видимость контейнера или хранилища", "storage_relationship_visible": "Расположение наличных относительно поверхности или хранилища",
    "workstation_unattended_after_departure": "Рабочее место без оператора после ухода", "workstation_work_context_visible": "Рабочий контекст кассового места",
}


def number(value) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def rows(value) -> list[dict]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def title(identifier) -> str:
    return TITLES.get(identifier, "Другая проверка") if isinstance(identifier, str) else "Другая проверка"


def format_time(value) -> str:
    if not number(value) or value < 0:
        return "—"
    seconds = int(value)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}" if hours else f"{minutes:02}:{seconds:02}"


def model_status(case) -> str:
    if case.get("status") == "confirmed":
        return "confirmed"
    if case.get("status") != "rejected" and case.get("requires_review") is not False and (
            case.get("status") == "review_required" or case.get("requires_review") is True):
        return "review_required"
    return "rejected"


def model_label(case) -> str:
    status = model_status(case)
    if status == "confirmed":
        return "Нарушение подтверждено моделью"
    if status == "review_required":
        return "Требует проверки"
    contrary = (case.get("action_decision") == "rejected" or case.get("context_decision") == "rejected"
                or any(check.get("status") == "no" for check in rows(case.get("checks"))))
    codes = case.get("reason_codes") or []
    return "Не подтвердилось" if contrary or any(code in codes for code in (
        "action_rejected", "legal_context_rejected", "action_contradiction", "context_contradiction")) else "Недостаточно данных"


def observation_facts(case) -> list[tuple[str, str]]:
    checks: dict[str, list[dict]] = {}
    for check in rows(case.get("checks")):
        if check.get("stage") in {"action", "context"} and isinstance(check.get("name"), str):
            checks.setdefault(check["name"], []).append(check)

    def sources(name):
        items = checks.get(name, [])
        if not items or any(item.get("status") != "yes" for item in items):
            return set()
        refs = [{(ref["source_index"], ref["timestamp_sec"]) for ref in rows(item.get("evidence_refs"))
                 if type(ref.get("source_index")) is int and 0 <= ref["source_index"] <= 2**53 - 1
                 and number(ref.get("timestamp_sec")) and ref["timestamp_sec"] >= 0} for item in items]
        return set.union(*refs) if all(refs) else set()

    blocked = any(code in (case.get("reason_codes") or []) for code in (
        "source_timestamp_gap", "scene_change_context", "insufficient_before_after"))

    def sequence(refs):
        return not blocked and len({index for index, _ in refs}) >= 2 and len({time for _, time in refs}) >= 2

    departures = [refs for name in ("actual_workstation_departure", "workstation_departure_observed", "workstation_departure")
                  if sequence(refs := sources(name))]

    def after(name):
        refs = sources(name)
        return bool(refs and any(min(time for _, time in refs) >= max(time for _, time in before) for before in departures))

    phrases = [
        ("food", ["food_item_visually_identified", "food_on_cashier_work_surface"], "shared", "На рабочем столе видны продукты питания."),
        ("food", ["personal_food_on_work_table"], "", "На рабочем столе видны продукты питания."),
        ("food", ["food_item_visually_identified"], "", "В кадре видны продукты питания."),
        ("object", ["non_cash_object_visually_supported", "object_cash_area_relationship_visible"], "shared", "В кассовой зоне виден предмет, не относящийся к наличным."),
        ("object", ["non_cash_object_visually_supported"], "", "В кадре виден предмет, не относящийся к наличным."),
        *(("departure", [name], "sequence", "Человек покидает рабочее место.") for name in
          ("actual_workstation_departure", "workstation_departure_observed", "workstation_departure")),
        ("screen", ["same_screen_visibly_on_after_departure"], "", "После ухода человека экран виден включённым." if after("same_screen_visibly_on_after_departure") else "Экран компьютера виден включённым."),
        ("computer", ["computer_at_workstation_identified"], "", "На рабочем месте виден компьютер."),
        ("unattended", ["workstation_unattended_after_departure"], "", "После ухода человека рабочее место видно без оператора." if after("workstation_unattended_after_departure") else "Рабочее место видно без оператора."),
        ("cash", ["cash_on_work_surface_observed"], "", "Вне контейнера видны предметы, похожие на банкноты."),
        ("cash", ["cash_outside_designated_storage"], "", "Наличные видны вне места хранения."),
        ("handover", ["cash_handover_observed"], "", "Видна передача предметов, похожих на банкноты."),
        *(("cash", [name], "", "Видна работа с предметами, похожими на банкноты.") for name in
          ("cash_acceptance_or_count_observed", "cash_handling_observed")),
        ("cash", ["cash_processing_observed"], "", "Видна обработка предметов, похожих на банкноты."),
        ("cash", ["loose_cash_established"], "", "В кадре видны неупакованные наличные."),
        *(("client", ["client_area_visible", name], "shared", "В обозреваемой клиентской зоне клиент не виден.") for name in
          ("client_absence_visually_supported", "client_demonstrably_absent")),
        ("key", ["key_or_stamp_visually_supported"], "", "В кадре виден предмет, похожий на ключ или штамп."),
        ("storage", ["cash_storage_relationship_observed"], "", "У места хранения видны предметы, похожие на банкноты."),
        ("storage", ["storage_relationship_visible"], "", "Видно расположение предметов, похожих на банкноты."),
        ("storage", ["storage_container_visible"], "", "В кадре виден контейнер или место хранения."),
        ("eating", ["food_consumption_visibly_observed"], "", "Человек принимает пищу."),
    ]
    result = []
    for family, names, requirement, text in phrases:
        refs = [sources(name) for name in names]
        if (all(refs) and (requirement != "shared" or set.intersection(*refs))
                and (requirement != "sequence" or all(sequence(value) for value in refs))):
            result.append((family, text))
    return result


def justification(cases: list[dict]) -> str:
    if not cases:
        return "По доступным кадрам действие не подтверждено."
    facts = [observation_facts(case) for case in cases]
    common = [(family, text) for family, text in facts[0] if all(any(other == text for _, other in row) for row in facts)]
    families, sentences = set(), []
    for family, text in common:
        if family not in families:
            families.add(family)
            sentences.append(text)
        if len(sentences) == 2:
            break
    if sentences:
        return " ".join(sentences)
    if len(cases) > 1 and any(facts):
        return "Наблюдения в этом диапазоне различаются; подробности приведены для каждого момента."
    return ("Для описания действия недостаточно сохранённых данных; проверьте кадры."
            if all(case.get("status") == "confirmed" for case in cases)
            else "По доступным кадрам действие не подтверждено.")
