"""Create a standalone Russian HTML review of the completed cash-model pilot.

Read-only inputs; embeds existing ROI pixels and SVG overlays. No inference,
network access, training, threshold fitting or automatic promotion is performed.
"""
from __future__ import annotations

import argparse
import base64
from collections import defaultdict
import hashlib
import html
import json
import math
from pathlib import Path
import statistics

from calibrate_threshold import counts_at_threshold, nms, xywh_to_xyxy

ROOT = Path(__file__).resolve().parent
DEFAULT_PILOT = ROOT / 'training/cash_pilot_20261017'
MODE_NAMES = {
    'rtdetr_roi': 'Новая модель банкнот · область стола',
    'gdino_fullframe': 'Grounding DINO · полный кадр',
    'gdino_roi': 'Grounding DINO · область стола',
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def escape(value):
    return html.escape(str(value), quote=True)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def number(value, digits=3):
    if value is None:
        return '—'
    if not math.isfinite(float(value)):
        raise ValueError('Non-finite metric')
    return f'{float(value):.{digits}f}'


def percent(value):
    return '—' if value is None else f'{float(value) * 100:.1f}%'


def format_time(value):
    value = float(value or 0)
    return f'{value:.3f}'.rstrip('0').rstrip('.') + ' с'


def load_predictions(path, allowed_image_ids, expected_floor):
    cache = read_json(path)
    if not cache.get('complete'):
        raise ValueError(f'Prediction cache is incomplete: {path}')
    records = cache['records']
    payload = json.dumps(records, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    if hashlib.sha256(payload).hexdigest() != cache.get('records_sha256'):
        raise ValueError(f'Prediction cache payload changed: {path}')
    ids = [row['image_id'] for row in records]
    if len(ids) != len(set(ids)) or set(ids) != allowed_image_ids:
        raise ValueError('Prediction cache images differ from test split')
    if cache.get('signature', {}).get('raw_score_floor') != expected_floor:
        raise ValueError('Prediction cache score floor differs from evaluation')
    rows = [pred for record in records for pred in record['predictions']]
    for record in records:
        if any(pred['image_id'] != record['image_id'] for pred in record['predictions']):
            raise ValueError('Prediction assigned to another source image')
    times = [float(record['inference_seconds']) for record in records if 'inference_seconds' in record]
    if any(not math.isfinite(value) or value < 0 for value in times):
        raise ValueError('Invalid recorded inference timing')
    timing = {'samples': len(times), 'total_seconds': sum(times) if times else None,
              'mean_seconds': statistics.mean(times) if times else None,
              'median_seconds': statistics.median(times) if times else None}
    return rows, timing


def prepare_predictions(rows, threshold, nms_iou=.6):
    grouped = defaultdict(list)
    for row in rows:
        score = float(row['score'])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Invalid prediction score')
        grouped[row['image_id']].append({'box': xywh_to_xyxy(row['bbox']), 'score': score})
    return {image_id: [row for row in nms(predictions, nms_iou) if row['score'] >= threshold]
            for image_id, predictions in grouped.items()}


def image_data_url(path):
    suffix = Path(path).suffix.lower()
    mime = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.webp': 'image/webp'}.get(suffix)
    if mime is None:
        raise ValueError(f'Unsupported embedded image format: {path}')
    return f'data:{mime};base64,' + base64.b64encode(Path(path).read_bytes()).decode('ascii')


def svg_overlay(data_url, width, height, truth, predictions, title):
    pieces = [f'<svg role="img" aria-label="{escape(title)}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">',
              f'<image href="{data_url}" width="{width}" height="{height}"/>']
    for index, box in enumerate(truth, 1):
        x1, y1, x2, y2 = box
        pieces.append(f'<rect class="truth-box" x="{x1:g}" y="{y1:g}" width="{x2-x1:g}" height="{y2-y1:g}"/>')
    for index, row in enumerate(predictions, 1):
        x1, y1, x2, y2 = row['box']
        label = f'{index}: {row["score"]:.3f}'
        label_x = max(1, min(width - 65, x1))
        label_y = min(height - 3, max(11, y1 - 3))
        pieces += [f'<rect class="prediction-box" x="{x1:g}" y="{y1:g}" width="{x2-x1:g}" height="{y2-y1:g}"/>',
                   f'<text class="prediction-label" x="{label_x:g}" y="{label_y:g}">{escape(label)}</text>']
    pieces.append('</svg>')
    return ''.join(pieces)


def result_message(models):
    candidate = models.get('rtdetr_roi')
    if not candidate:
        return 'В отчёте нет результата новой модели. Автоматические заключения о нарушениях не включены.'
    comparisons = []
    for mode, result in models.items():
        if mode == 'rtdetr_roi':
            continue
        worse = []
        candidate_ap, baseline_ap = candidate['test']['raw_ap']['map_50'], result['test']['raw_ap']['map_50']
        if candidate_ap is not None and baseline_ap is not None and candidate_ap < baseline_ap - 1e-9:
            worse.append('AP50')
        if candidate['test']['fixed_threshold']['f1'] < result['test']['fixed_threshold']['f1'] - 1e-9:
            worse.append('F1 при выбранном пороге')
        if worse:
            comparisons.append(f'{MODE_NAMES.get(mode, mode)} ({", ".join(worse)})')
    if comparisons:
        return ('На отложенных кадрах новая модель уступает: ' + '; '.join(comparisons) + '. '
                'Она не включена для автоматических заключений о нарушениях. Результат сохранён как эксперимент для разбора ошибок.')
    return ('Этот малый тест сам по себе не подтверждает готовность новой модели. '
            'Автоматические заключения о нарушениях не включены; нужна проверка рамок человеком и новые записи.')


def build_report(evaluation, dataset, output):
    evaluation, dataset, output = Path(evaluation).resolve(), Path(dataset).resolve(), Path(output)
    if not output.is_absolute():
        raise ValueError('--output must be an absolute HTML path')
    output = output.resolve()
    if output.suffix.lower() not in ('.html', '.htm'):
        raise ValueError('--output must name an HTML file')
    report_path = evaluation if evaluation.is_file() else evaluation / 'cash_pilot_evaluation.json'
    evaluation_root = report_path.parent
    report = read_json(report_path)
    if report.get('status') != 'completed' or report.get('thresholds_locked_before_test_inference') is not True:
        raise ValueError('Only a completed evaluation with validation threshold lock can be exported')
    locked_path = evaluation_root / 'locked_thresholds.json'
    locked = read_json(locked_path)
    if sha256(locked_path) != report['threshold_lock_sha256'] or locked.get('fit_split') != 'val' or locked.get('test_labels_used') is not False:
        raise ValueError('Validation threshold lock is invalid or changed')
    splits = {split: read_json(dataset / f'{split}.json') for split in ('train', 'val', 'test')}
    for split in splits:
        if sha256(dataset / f'{split}.json') != report['signature']['split_sha256'][split]:
            raise ValueError(f'Dataset {split} changed after evaluation')
    test = splits['test']
    if len(test['categories']) != 1 or test['categories'][0]['name'] != 'cash':
        raise ValueError('Expected cash-only annotations')
    cid = test['categories'][0]['id']
    ids = {item['id'] for item in test['images']}
    gt = defaultdict(list)
    for ann in test['annotations']:
        gt[ann['image_id']].append(xywh_to_xyxy(ann['bbox']))
    modes = [mode for mode in MODE_NAMES if mode in report['models']]
    modes += [mode for mode in report['models'] if mode not in modes]
    predictions, timings, per_case = {}, {}, defaultdict(dict)
    nms_iou = report['signature'].get('nms_iou', .6)
    match_iou = report['signature'].get('match_iou', .5)
    floor = report['signature']['prediction_floor']
    for mode in modes:
        threshold = report['models'][mode]['threshold']
        if threshold != locked['models'][mode]['threshold']:
            raise ValueError('Displayed threshold differs from locked validation threshold')
        rows, timings[mode] = load_predictions(evaluation_root / f'{mode}_test_predictions.json', ids, floor)
        if any(row['category_id'] != cid for row in rows):
            raise ValueError('Unexpected prediction category')
        predictions[mode] = prepare_predictions(rows, threshold, nms_iou)
        metrics = counts_at_threshold(predictions[mode], gt, threshold, match_iou)
        expected = report['models'][mode]['test']['fixed_threshold']
        if any(metrics[key] != expected[key] for key in ('tp', 'fp', 'fn')):
            raise ValueError(f'{mode} displayed predictions disagree with reported TP/FP/FN')
        for item in test['images']:
            image_id = item['id']
            per_case[image_id][mode] = counts_at_threshold(
                {image_id: predictions[mode].get(image_id, [])}, {image_id: gt[image_id]}, threshold, match_iou)
    counts = {split: len(data['images']) for split, data in splits.items()}
    groups = {split: len({item['source_sha256'].lower() for item in data['images']}) for split, data in splits.items()}
    negative_count = sum(not gt[item['id']] for item in test['images'])
    result = result_message(report['models'])
    metric_rows = []
    for mode in modes:
        row = report['models'][mode]
        ap, metric = row['test']['raw_ap'], row['test']['fixed_threshold']
        metric_rows.append('<tr>' + f'<th scope="row">{escape(MODE_NAMES.get(mode, mode))}</th>' + ''.join(
            f'<td>{escape(value)}</td>' for value in [number(row['threshold']), number(ap['map_50']), number(ap['map_50_95']),
              metric['tp'], metric['fp'], metric['fn'], percent(metric['precision']), percent(metric['recall']),
              f'{metric["negative_images_with_false_positive"]}/{metric["annotated_negative_images"]} кадров; {metric["false_positives_on_negative_images"]} рамок']) + '</tr>')
    cases = []
    for index, item in enumerate(test['images'], 1):
        image_id = item['id']
        source_video = item['source_video']
        title = f'{index}. {source_video} · {format_time(item.get("timestamp_sec"))} · кадр {item.get("source_frame", "—")}'
        image_path = (dataset / item['file_name']).resolve()
        if not image_path.is_relative_to(dataset):
            raise ValueError('ROI file outside dataset')
        data_url = image_data_url(image_path)
        cards = []
        for mode in modes:
            predicted = predictions[mode].get(image_id, [])
            metric = per_case[image_id][mode]
            threshold = report['models'][mode]['threshold']
            cards.append(f'<article class="model-card" data-mode="{escape(mode)}"><h4>{escape(MODE_NAMES.get(mode, mode))}</h4>'
                         f'<p class="small">Порог {threshold:.3f} · найдено рамок: {len(predicted)}</p>'
                         + svg_overlay(data_url, item['width'], item['height'], gt[image_id], predicted, f'{title}, {MODE_NAMES.get(mode, mode)}')
                         + f'<p class="case-counts"><span>Совпало: <b>{metric["tp"]}</b></span><span>Лишних: <b>{metric["fp"]}</b></span><span>Пропущено: <b>{metric["fn"]}</b></span></p></article>')
        expectation = f'В предварительной разметке: {len(gt[image_id])} рамок банкнот.' if gt[image_id] else 'Отрицательный пример: в предварительной разметке этой области денег нет.'
        cases.append(f'<section class="example" id="case-{image_id}"><div class="case-header"><div><h3>{escape(title)}</h3><p>{escape(expectation)}</p></div>'
                     f'<a href="http://127.0.0.1:8765/annotations#{escape(source_video)}" target="_blank" rel="noopener">Открыть исходное видео ↗</a></div>'
                     '<div class="legend"><span><i class="gt-key"></i>Предварительная разметка</span><span><i class="pred-key"></i>Предсказание модели и оценка</span></div>'
                     '<div class="model-grid">' + ''.join(cards) + '</div></section>')
    timing_rows = []
    for mode in modes:
        value = timings[mode]
        timing_rows.append(f'<tr><th scope="row">{escape(MODE_NAMES.get(mode, mode))}</th><td>{value["samples"]}</td>'
                           f'<td>{number(value["mean_seconds"])} с</td><td>{number(value["median_seconds"])} с</td>'
                           f'<td>{number(value["total_seconds"])} с</td></tr>')
    split_rows = ''.join(f'<tr><th>{escape(name)}</th><td>{counts[split]}</td><td>{len(splits[split]["annotations"])}</td><td>{groups[split]}</td></tr>'
                         for split, name in [('train', 'Обучение'), ('val', 'Выбор порогов'), ('test', 'Отложенная проверка')])
    details = {'created_utc': report.get('created_utc'), 'evaluation': str(report_path), 'dataset': str(dataset),
               'threshold_lock_sha256': report['threshold_lock_sha256'], 'models': report['signature']['models']}
    command = ('python detect_cash_roi.py --video X26.mov --case-view '
               '--weights training/cash_pilot_20261017/model/best.pth '
               '--calibration training/cash_pilot_20261017/evaluation/calibration.json '
               '--output results/cash_pilot_manual --no-video')
    body = f'''<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Проверка модели банкнот — результаты пилота</title><style>
:root{{--navy:#202d43;--accent:#9b214b;--muted:#56657b;--line:#d9e0e8;--paper:#fff;--ground:#158172}}
*{{box-sizing:border-box}}body{{margin:0;background:#f4f6f9;color:var(--navy);font:16px/1.55 "Segoe UI",Arial,sans-serif}}main{{max-width:1390px;margin:auto;padding:32px 28px 60px}}header{{border-top:5px solid var(--accent);padding:24px 28px;background:var(--paper);margin-bottom:22px}}h1{{font-size:30px;line-height:1.2;margin:7px 0 13px}}h2{{font-size:23px;margin:0 0 12px}}h3{{font-size:19px;margin:0 0 5px}}h4{{font-size:16px;line-height:1.35;margin:0 0 5px}}p{{margin:8px 0}}a{{color:var(--accent);text-underline-offset:3px}}.eyebrow{{font-size:13px;color:var(--muted);letter-spacing:.04em}}.lead{{max-width:1020px}}.decision{{border-left:4px solid var(--accent);background:#fff6f9;padding:15px 18px;margin-top:18px}}.block,.example{{background:var(--paper);border:1px solid var(--line);padding:23px;margin-bottom:20px}}.small,.muted{{font-size:13px;color:var(--muted)}}.stats{{display:flex;gap:30px;flex-wrap:wrap;margin:20px 0}}.stat b{{font-size:26px;display:block;line-height:1.3}}.table-scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{text-align:left;padding:12px 10px;border-bottom:1px solid var(--line);vertical-align:top}}thead th{{background:#eef2f7;font-size:13px}}tbody th{{font-weight:600;min-width:190px}}td{{font-variant-numeric:tabular-nums;white-space:nowrap}}.controls{{display:flex;flex-wrap:wrap;gap:18px;align-items:end;margin:20px 0}}label{{font-size:13px;font-weight:600;display:flex;flex-direction:column;gap:6px}}select{{max-width:100%;min-width:220px;padding:10px;border:1px solid #aeb9c8;border-radius:4px;background:white;color:var(--navy);font:14px "Segoe UI",Arial}}button{{font:14px "Segoe UI",Arial;padding:10px 16px;background:var(--accent);color:white;border:0;border-radius:4px;cursor:pointer}}.case-header{{display:flex;justify-content:space-between;gap:20px;align-items:start}}.case-header a{{white-space:nowrap;font-size:14px}}.case-header p{{font-size:14px;color:var(--muted)}}.model-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:17px;margin-top:15px}}.model-card{{min-width:0;border:1px solid var(--line);padding:14px;background:#fcfdff}}.model-card svg{{display:block;width:100%;height:auto;background:#e3e8ee;margin:13px 0}}.truth-box{{fill:none;stroke:var(--ground);stroke-width:2;stroke-dasharray:5 3;vector-effect:non-scaling-stroke}}.prediction-box{{fill:none;stroke:var(--accent);stroke-width:2;vector-effect:non-scaling-stroke}}.prediction-label{{font:9px "Segoe UI",Arial;fill:#fff;stroke:#681331;stroke-width:3;paint-order:stroke;stroke-linejoin:round}}.legend{{display:flex;gap:24px;flex-wrap:wrap;font-size:13px;margin:12px 0}}.legend span{{display:flex;align-items:center;gap:7px}}.legend i{{display:inline-block;width:20px;height:12px}}.gt-key{{border:2px dashed var(--ground)}}.pred-key{{border:2px solid var(--accent)}}.case-counts{{display:flex;gap:12px;flex-wrap:wrap;font-size:13px}}.steps{{margin:12px 0;padding-left:23px}}.steps li{{padding:5px 0}}details{{margin:14px 0}}summary{{cursor:pointer;font-weight:600;color:var(--accent)}}pre{{white-space:pre-wrap;word-break:break-word;background:#eef2f7;padding:16px;font-size:12px;line-height:1.5}}.footnote{{font-size:13px;color:var(--muted);margin-top:16px}}[hidden]{{display:none!important}}.model-grid.single{{grid-template-columns:minmax(0,650px)}}@media(max-width:1000px){{.model-grid{{grid-template-columns:1fr}}.model-card{{max-width:700px}}}}@media(max-width:650px){{main{{padding:16px 10px 40px}}header,.block,.example{{padding:18px}}h1{{font-size:25px}}.case-header{{display:block}}.case-header a{{display:inline-block;margin-top:9px}}select{{min-width:0;width:100%}}label{{width:100%}}.stats{{gap:20px}}}}@media print{{body{{background:white}}main{{padding:0}}.controls{{display:none}}.block,.example{{break-inside:avoid}}.example[hidden]{{display:block!important}}}}
</style></head><body><main>
<header><div class="eyebrow">Локальный эксперимент · распознавание банкнот</div><h1>Пилот модели банкнот: результаты</h1>
<p class="lead">Сравнение на отложенных кадрах: новая модель и два варианта Grounding DINO. Во всех случаях оценивается одна и та же область стола. Рамки денег не устанавливают нарушение правил.</p>
<div class="stats"><div class="stat"><b>{counts['train']}</b>кадров для обучения</div><div class="stat"><b>{counts['val']}</b>для выбора порогов</div><div class="stat"><b>{counts['test']}</b>для проверки</div><div class="stat"><b>{groups['test']}</b>исходных записей в проверке</div></div>
<div class="decision">{escape(result)}</div></header>
<section class="block"><h2>Результаты на отложенных кадрах</h2><p>Порог каждой модели выбран по validation и зафиксирован до проверки test. Числа TP, FP и FN относятся к рамкам предметов.</p>
<div class="table-scroll"><table><thead><tr><th>Модель</th><th>Порог</th><th>AP50</th><th>AP50:95</th><th>TP<br>совпало</th><th>FP<br>лишних</th><th>FN<br>пропущено</th><th>Precision</th><th>Recall</th><th>Ложные срабатывания<br>на отрицательных кадрах</th></tr></thead><tbody>{''.join(metric_rows)}</tbody></table></div>
<p class="footnote">AP приведён в шкале 0–1 и рассчитан по исходным рамкам с оценкой ≥ {floor:g}, до удаления дублей. Для рабочих TP/FP/FN применяется удаление дублей при IoU ≥ {nms_iou:g}; совпадение с разметкой — IoU ≥ {match_iou:g}. Precision — доля совпавших рамок среди найденных; recall — доля найденных размеченных предметов. Это не «процент правильно найденных нарушений».</p></section>
<section class="block"><h2>Что проверить в примерах</h2><ol class="steps"><li>Просмотрите кадры ниже. Зелёный пунктир показывает предварительную разметку банкнот, бордовая рамка — предсказание модели.</li><li>Проверьте сам предмет: деньги, мешок, шоколад и документ могут выглядеть похоже. Ошибочную предварительную рамку нужно исправить до выводов о качестве.</li><li>Сравните три модели при их зафиксированных порогах. Число рядом с бордовой рамкой — оценка модели, а не вероятность нарушения.</li><li>Откройте исходное видео и проверьте действие до и после кадра. Подтверждения и исправления сохраняются на странице разметки видео.</li></ol>
<p class="small">Ссылка на исходное видео работает, когда локальное приложение запущено. Сам отчёт и все показанные кадры открываются без интернета.</p>
</section>
{''.join(cases)}
<section class="block"><h2>Границы этой проверки</h2><p>{escape(report.get('caveat', ''))}</p><p>В test — {len(test['annotations'])} предварительных рамок банкнот и {negative_count} кадров без размеченных денег. Соседние кадры одной записи связаны между собой: восемь кадров не означают восемь независимых ситуаций. Копии записей объединены по SHA-256.</p>
<div class="table-scroll"><table><thead><tr><th>Часть набора</th><th>Кадры</th><th>Рамки банкнот</th><th>Исходные записи по SHA-256</th></tr></thead><tbody>{split_rows}</tbody></table></div>
<p class="footnote">При подготовке сравнения не применялись подсказки с правильными предметами или исправления по совпадению с ранее размеченным кадром. Оценка сохранена отдельно; автоматическое включение новой модели не выполнялось.</p></section>
<section class="block"><h2>Измеренное время обработки</h2><p>Ниже только время, записанное при получении предсказаний test. Оно включает чтение изображения и вызов модели; загрузка весов, декодирование видео и подготовка отчёта сюда не входят.</p>
<div class="table-scroll"><table><thead><tr><th>Режим</th><th>Измерений</th><th>Среднее на кадр</th><th>Медиана</th><th>Сумма</th></tr></thead><tbody>{''.join(timing_rows)}</tbody></table></div>
<p class="footnote">Это не скорость обработки полного видео и не гарантия на другом компьютере. Полный кадр и область стола имеют разный объём входных данных.</p></section>
<section class="block"><h2>Продолжить проверку</h2><p><a href="http://127.0.0.1:8765/annotations">Открыть все видео и разметку</a>. Сначала уточните спорные предметы и рамки, затем добавьте новые записи для отдельной проверки. Изменённый test нельзя повторно считать нетронутой проверкой после настройки модели.</p>
<details><summary>Технические сведения и отдельный ручной запуск</summary><p>Эксперимент запускается отдельно от основной проверки нарушений. В примере задана область стола для ракурса 464 × 848; для другого ракурса нужна явная область. Рабочая папка: <code>{escape(ROOT)}</code>.</p><pre>{escape(command)}</pre><p>Веса, набор и контрольные суммы:</p><pre>{escape(json.dumps(details, ensure_ascii=False, indent=2))}</pre></details></section>
<p class="footnote">Отчёт создан из сохранённых результатов. Файлы изображений, разметки, весов и предсказаний при экспорте не изменялись.</p>
</main></body></html>'''
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(body, encoding='utf-8')
    return {'output': str(output), 'test_examples': len(test['images']), 'models': modes,
            'embedded_images': len(test['images']) * len(modes), 'bytes': output.stat().st_size,
            'independent_ground_truth': False, 'model_automatically_enabled': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, default=DEFAULT_PILOT / 'evaluation',
                        help='Completed evaluation directory or cash_pilot_evaluation.json')
    parser.add_argument('--dataset', type=Path, default=DEFAULT_PILOT / 'dataset_v2')
    parser.add_argument('--output', type=Path, required=True, help='Absolute standalone HTML output path')
    args = parser.parse_args()
    print(json.dumps(build_report(args.evaluation, args.dataset, args.output), ensure_ascii=False))


if __name__ == '__main__':
    main()
