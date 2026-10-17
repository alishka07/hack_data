"""Describe implemented observable subchecks separately from full bank rules."""

from rule_settings import DOOR_OPEN_THRESHOLD_LABEL

UNIMPLEMENTED = {'peripheral_test', 'permitted_desk_equipment', 'lights_after_departure', 'eating_while_cash_present'}
SCOPE = {
    'clean_desk_banned_items': 'Устойчивый телефон, флешка или блокнот в рабочей зоне',
    'food_drinks_at_workplace': 'Устойчивая еда или чашка у стола / кандидата кассира независимо от денег и ухода клиента',
    'unattended_assets': 'Предметы остаются после видимого ухода / отсутствия людей',
    'one_client_at_a_time': 'Следующий посетитель подходит, пока предыдущий ещё виден',
    'restricted_access_card_transfer': 'Дополнительный человек появляется со стороны кассира',
    'safe_closed': 'Открытая дверца сейфа после видимого ухода кандидата кассира',
    'door_open': f'Непрерывное видимое открытие двери не менее {DOOR_OPEN_THRESHOLD_LABEL}',
}


def describe_coverage(coverage, actions, states, service):
    result = []
    for original in coverage:
        row = dict(original)
        key = row['rule_id']
        checks = [dict(check) for check in actions.get('checks', []) if check['rule_id'] == key]
        if key in SCOPE and not checks:
            available = row['status'] in {'partial', 'checked'}
            count = row.get('event_count', 0)
            checks.append(dict(check_id=key + '_visible_evidence', title=SCOPE[key], available=available,
                               status='observed' if count else 'analyzed' if available else 'insufficient_evidence',
                               observation_count=count,
                               detail='Условие по видимым кадрам проверено; кандидатов для просмотра нет.' if available and not count else row.get('detail', '')))
        row['checks'] = checks
        if key in UNIMPLEMENTED:
            row.update(implementation_status='not_implemented', assessment_status='not_implemented',
                       status_label='Ещё не реализовано', reason_code='no_validated_method')
        elif row['status'] == 'not_checked':
            row.setdefault('implementation_status', 'implemented_partial')
            row.setdefault('reason_code', 'insufficient_evidence')
            row.update(assessment_status='insufficient_evidence', status_label='Недостаточно данных')
        else:
            row.setdefault('implementation_status', 'implemented_partial')
            row.setdefault('reason_code', 'partial_observable_scope')
            row.update(assessment_status='partial_scope', status_label='Визуальные признаки проверены')
        if key in {'door_open', 'safe_closed'}:
            counts = states.get('state_counts', {}).get('door' if key == 'door_open' else 'safe', {})
            if counts:
                row['detail'] += (f" Наблюдений: закрыто — {counts.get('closed', 0)}, открыто — {counts.get('open', 0)},"
                                  f" не определено — {counts.get('unknown', 0)}.")
        result.append(row)
    return result
