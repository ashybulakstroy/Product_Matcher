"""Report the frozen independent result, including failures and abstentions."""
from pathlib import Path
import json,statistics
from run_control import OUT,STAGE,ROOT,write


def main():
    results=json.loads((OUT/'results.json').read_text());summary=json.loads((OUT/'summary.json').read_text())
    rows=[];diagnosis=[];method_summary=[]
    for r in results:
        folder=OUT/'cases'/r['job'];d=json.loads((folder/'detector_v2.json').read_text())
        before=r['v1']['horizontal'] if r['v1'] else None;after=r['v2']['horizontal'] if r['v2'] else None
        reason=d.get('reason','полный набор горизонталей найден')
        if not after:
            observations=dict(job=r['job'],reason=reason,reference=r['reference_horizontal'],
                accepted_partial_edges=len(d.get('edges',[])),rejected_proposals=len(d.get('rejected',[])))
            if d.get('rejected'):
                strongest=sorted(d['rejected'],key=lambda e:-e.get('ink_coverage',0))[:5]
                observations['strongest_rejected_evidence']=[dict(position=e['position'],ink_coverage=e.get('ink_coverage'),column_coverage=e.get('column_coverage'),intersection_evidence=e.get('intersection_evidence'),reason=e.get('reason')) for e in strongest]
            diagnosis.append(observations)
        rows.append(f'<tr><td><a href="cases/{r["job"]}/index.html">{r["job"][:10]}</a></td><td>{r["reference_horizontal"]}</td><td>{len(before["matches"]) if before else "отказ"}</td><td>{len(after["matches"]) if after else "отказ"}</td><td>{after["missing"] if after else "все: результат не выдан"}</td><td>{after["false"] if after else "—"}</td><td>{after["mean_matched_px"] if after else "—"}</td><td>{reason}</td></tr>')
    for scenario in ['v1_auto','v2_auto','human_controls']:
        for method in ['grid_mesh','tps_scipy']:
            entries=[e for r in results for e in r['methods'] if e['input']==scenario and e['method']==method]
            good=[e for e in entries if e['status']=='CALCULATED'];vals=[e['geometry']['mean_px'] for e in good if 'geometry'in e]
            method_summary.append(dict(input=scenario,method=method,calculated=len(good),
                folds=sum(e['safety']['folded_fraction']>0 for e in good),
                matched_horizontal=sum(len(e['against_confirmed']['horizontal']['matches']) for e in good),
                equal_count_geometry_cases=len(vals),mean_geometry_px=statistics.mean(vals) if vals else None))
    write(OUT/'failure_diagnostics.json',diagnosis);write(OUT/'method_summary.json',method_summary)
    a=summary['v1_horizontal'];b=summary['v2_horizontal']
    method_rows=''.join(f'<tr><td>{s["input"]}</td><td>{s["method"]}</td><td>{s["calculated"]}/8</td><td>{s["folds"]}</td><td>{s["matched_horizontal"]}</td><td>{s["mean_geometry_px"] if s["mean_geometry_px"] is not None else "не оценено"}</td></tr>' for s in method_summary)
    findings=f'''# V3 — независимый контроль зафиксированной версии

## Вывод

Улучшение, полученное на диагностических десяти фотографиях, на независимом наборе НЕ подтвердилось. Новая версия не готова к внедрению. Настройки и исходники не менялись после фиксации контрольного набора и получения его разметки.

Приняты все 8 JSON. Проверены подтверждения человека, источник/размер фото, отсутствие совпадений SHA с диагностическим набором, отметка независимости физического чека, согласованность двух семейств линий с одной матрицей узлов и отсутствие пересечений в разметке. Идентичность чеков подтверждена пользователем; только различие SHA за независимость не принималось. Файлы разметки не изменялись. Контрольные суммы — input_manifest.json.

## Результаты горизонталей

| Версия | Подтверждено | Найдено | Полнота | Не возвращено | Лишних предложений | Средняя ошибка найденных пар |
|---|---:|---:|---:|---:|---:|---:|
| V1 | 53 | {a['matched']} | {a['recall']:.1%} | {a['missing']} | {a['false']} | {a['mean_matched_line_error_px']:.3f} px |
| V2 | 53 | {b['matched']} | {b['recall']:.1%} | {b['missing']} | {b['false']} | {b['mean_matched_line_error_px']:.3f} px |

V2 вернул полные горизонтальные наборы на 4 из 8 фото (4, 6, 5 и 9 границ), на остальных 4 отказал. Отказ учитывается как отсутствие всего результата, а не как успешное отсутствие ложных линий. Поэтому ноль лишних НЕ компенсирует низкую полноту. Порог сопоставления — 8 пикселей исходника. Средняя ошибка вычисляется только по найденным парам и не характеризует 29 невозвращённых границ. На четырёх снимках с выданной сеткой также сопоставлены 20 из 20 вертикалей; по всему набору это 20 из 40 с учётом отказов.

## Где возникли отказы

- 112ea4224c: fewer_than_three_supported_edges. Есть физические предложения, но недостаточно проходит все проверки приёма.
- 1dde6b99ba: five_vertical_proposals_missing. До горизонтального восстановления не подтверждены пять вертикальных предложений.
- 49ad1a1d95: fewer_than_three_supported_edges. Чек с 15 подтверждёнными границами; новая версия отказала целиком, тогда как V1 вернул 9 сопоставленных границ.
- ada2876336: fewer_than_three_supported_edges. Новая версия отказала, V1 возвращал 3 сопоставленные границы из 6.

Точные покрытия колонок, подтверждения подходов к пересечениям и сильнейшие отсечённые предложения — failure_diagnostics.json. Ограничение здесь — обнаружение и строгие условия приёма физического следа, а не невозможность Mesh/TPS воспроизвести правильные контроли. Не следует объявлять каждую отказавшую линию отсутствующей на фотографии. Причины качества печати и складок требуют отдельного визуального разбора, они не были автоматически придуманы по коду отказа.

## Grid Mesh и TPS

Проверены на одних и тех же автоматических линиях V1 и V2 и отдельно на подтверждённых линиях всех 8 снимков. В сценарии V2-auto построены по 4 карты каждого метода; для четырёх отказов детектора карта не подменялась ручными линиями. Ручные контроли — отдельный сценарий human_controls, они не передавались детектору. Карты по количеству пропущенных строк не оценивались как нулевая ошибка. Все данные — results.json и method_summary.json, изображения — cases/<job>/index.html.

Запуск: 4 процесса по фотографиям; длительность основного прогона {summary['seconds']:.2f} сек. Frozen-код детектора, методы выпрямления, рабочий OCR, оригиналы и обе группы подтверждённых разметок не изменены; integrity.json. Новая версия в рабочий OCR не переносилась.

## Следующий этап

Результат контрольной проверки зафиксировать как отрицательный. Для следующей версии нужно изучить отказы выбора вертикалей и приёма горизонтальных рёбер, сохранив проверки против текста. После использования этих восьми фото для исправлений они станут регрессионными: окончательную независимую проверку следующей версии следует проводить на другом зафиксированном наборе. Текущие результаты не переписывать после подстройки.
'''
    (OUT/'FINDINGS.md').write_text(findings,encoding='utf-8')
    page=f'''<meta charset="utf-8"><title>V3: независимая проверка</title><style>body{{font:16px system-ui;margin:28px}}p{{max-width:1150px}}td,th{{border:1px solid #aaa;padding:8px}}table{{border-collapse:collapse}}.verdict{{background:#fff0d9;padding:16px}}</style><h1>V3 — независимая проверка, 8 подтверждённых фото</h1><p class="verdict"><b>Улучшение не подтвердилось. К внедрению не готово.</b> V2: {b['matched']}/53 горизонталей ({b['recall']:.1%}), V1: {a['matched']}/53 ({a['recall']:.1%}). V2 полностью отказывает на четырёх снимках. Код после получения разметки не менялся.</p><p><a href="FINDINGS.md">Заключение</a> · <a href="metrics.csv">CSV</a> · <a href="results.json">Все результаты</a> · <a href="failure_diagnostics.json">Отсечённые физические предложения</a> · <a href="integrity.json">Неизменность файлов</a></p><p>Лишних предложенных горизонталей: 0. Средняя ошибка найденных пар V2: {b['mean_matched_line_error_px']:.2f} px. Обе цифры относятся к выданным линиям: они не компенсируют пропуски. Индексы с 0; порог сопоставления 8 px.</p><table><tr><th>Фото</th><th>Подтверждено</th><th>V1 найдено</th><th>V2 найдено</th><th>V2 пропуски</th><th>V2 лишние</th><th>Ошибка пар, px</th><th>Статус</th></tr>{''.join(rows)}</table><h2>Отдельная проверка преобразований</h2><p>human_controls — проверка соблюдения подтверждённых контролей, не обход отказа детектора. Средняя геометрическая ошибка приводится только при совпадении количества линий.</p><table><tr><th>Входы</th><th>Метод</th><th>Построено</th><th>Фото с переворотом карты</th><th>Сопоставлено горизонталей</th><th>Средняя геометрическая ошибка, px</th></tr>{method_rows}</table><p>Рабочий OCR и методы не изменены. Результаты диагностического и независимого наборов сохраняются отдельно.</p>'''
    (OUT/'index.html').write_text(page,encoding='utf-8')
    for previous in [STAGE/'index.html',ROOT/'experiment_outputs/standalone_geometry_v3_confirmed/index.html']:
        text=previous.read_text(encoding='utf-8');link='<p><strong><a href="../standalone_geometry_v3_independent/index.html">Независимая проверка завершена: улучшение не подтвердилось</a></strong></p>'
        if '../standalone_geometry_v3_independent/index.html' not in text:previous.write_text(text.replace('<h1>',link+'<h1>',1),encoding='utf-8')
    print(json.dumps(method_summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
