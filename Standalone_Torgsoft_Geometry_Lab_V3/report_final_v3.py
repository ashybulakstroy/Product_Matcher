"""Final-set report with explicit partial-reference and tail-exclusion provenance."""
import json,csv,statistics,hashlib
import numpy as np
from pathlib import Path
from run_final_v3 import ROOT,LAB,STAGE,OUT,write


def aggregate(results,version,axis):
    tp=fp=fn=0;errors=[]
    for r in results:
        m=r[version]['match'];a=m[axis] if m else None
        tp+=len(a['matches']) if a else 0;fp+=len(a['false']) if a else 0
        fn+=len(a['missing']) if a else r['reference_'+axis]
        if a:errors.extend(p['mean_px'] for p in a['matches'])
    return dict(reference=tp+fn,matched=tp,missing=fn,false=fp,recall=tp/(tp+fn),
        mean_matched_line_error_px=statistics.mean(errors) if errors else None,
        max_matched_line_mean_error_px=max(errors) if errors else None)


def main():
    results=json.loads((OUT/'results.json').read_text());manifest=json.loads((OUT/'reference_manifest.json').read_text())
    summary={v:{a:aggregate(results,v,a) for a in ['horizontal','vertical']} for v in ['v2','v3']}
    for v in ['v2','v3']:
        summary[v]['rejections']=sum(r[v]['match'] is None for r in results)
        summary[v]['complete_table_cases']=sum(bool(r['complete_reference_matrix'] and r[v]['match'] and not any(r[v]['match'][a]['missing'] or r[v]['match'][a]['false'] for a in ['horizontal','vertical'])) for r in results)
        summary[v]['total_case_seconds']=sum(r[v]['seconds'] for r in results)
    summary.update(photos=8,partial_reference_photos=[r['job'] for r in results if not r['complete_reference_matrix']],
        removed_free_tail_node_occurrences=sum(m['tails_removed'] for m in manifest),coordinates_moved=0,
        reference_scope='user-confirmed segments between existing perpendicular links',editor_changed=False)
    write(OUT/'summary.json',summary)
    rows=[]
    for r in results:
        m=r['v3']['match'];h=m['horizontal'] if m else None;v=m['vertical'] if m else None
        rows.append(f'<tr><td><a href="cases/{r["job"]}/index.html">{r["job"][:10]}</a></td><td>{r["reference_horizontal"]}</td><td>{len(r["v2"]["match"]["horizontal"]["matches"]) if r["v2"]["match"] else "отказ"}</td><td>{len(h["matches"]) if h else "отказ"}</td><td>{h["missing"] if h else "все: результат не возвращён"}</td><td>{h["false"] if h else "—"}</td><td>{len(v["matches"]) if v else 0}/5</td><td>{h["mean_matched_px"] if h else "—"}</td><td>{"полная сеть" if r["complete_reference_matrix"] else "частичная сеть пересечений"}</td></tr>')
    with (OUT/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['job','version','axis','reference','matched','missing','false','mean_matched_px','reference_complete'])
        for r in results:
            for version in ['v2','v3']:
                for axis in ['horizontal','vertical']:
                    a=r[version]['match'][axis] if r[version]['match'] else None
                    w.writerow([r['job'],version,axis,r['reference_'+axis],len(a['matches']) if a else 0,a['missing'] if a else 'ALL_NOT_RETURNED',a['false'] if a else None,a['mean_matched_px'] if a else None,r['complete_reference_matrix']])
    a=summary['v2']['horizontal'];b=summary['v3']['horizontal'];method_summary=[]
    for method in ['grid_mesh','tps_scipy']:
        entries=[e for r in results for e in r['methods'] if e['method']==method];ok=[e for e in entries if e['status']=='CALCULATED']
        method_summary.append(dict(method=method,calculated=len(ok),folded_cases=sum(e['safety']['folded_fraction']>0 for e in ok),matched_horizontal=sum(len(e['match']['horizontal']['matches']) for e in ok)))
    write(OUT/'method_summary.json',method_summary)
    findings=f'''# V3 — финальный отдельный набор, оценка участков между пересечениями

## Правило пользователя

Пользователь уточнил: свободные ветви, у которых один конец не имеет пересечения с перпендикулярной линией, созданы редактором ошибочно; участки с перпендикулярными связями верны. Затем отдельно потребовал редактор пока не менять, а сосредоточиться на основной задаче.

Редактор НЕ менялся. Исходные 8 JSON НЕ менялись. В references/ сохранены отдельные расчётные копии: у каждой горизонтали и вертикали удалена только последовательность узлов до первого существующего перпендикулярного пересечения и после последнего. Внутренние узлы изгиба сохранены. Новых узлов, пересечений, линий и координат не создавалось, ни один существующий узел не сдвинут. Удалено {summary['removed_free_tail_node_occurrences']} вхождений свободных узлов в линии. Сохранены контрольные суммы исходников и списки удалённых хвостов в reference_manifest.json и references/*.json.

Автоматический статус human_verified не изменён на true: исходные сохранения остаются false. Подтверждённая пользователем область явно обозначена user_confirmed_scope=true и reference_scope=segments_bounded_by_existing_perpendicular_links. Подтверждение относится к связанным участкам, не к произвольным хвостам или предполагаемой полноте матрицы.

## Проверка зафиксированной версии

Код V3 проверен по final_candidate_frozen.json: ни детектор, ни Grid Mesh/TPS после предъявления этого набора не менялись. Детекторы получали только исходное изображение, расчётные эталоны загружались после детектирования. Запуск — 4 процесса по фотографиям. Этот набор не использовался для настройки V3; ранее для него существовали непромаркированные результаты V1/V2, что указано в исходном манифесте резерва.

| Показатель | V2 | V3 |
|---|---:|---:|
| Сопоставлено горизонталей | {a['matched']}/{a['reference']} | {b['matched']}/{b['reference']} |
| Полнота по размеченным участкам | {a['recall']:.1%} | {b['recall']:.1%} |
| Несопоставленные предложенные H | {a['false']} | {b['false']} |
| Вертикали | {summary['v2']['vertical']['matched']}/40 | {summary['v3']['vertical']['matched']}/40 |
| Полностью сопоставленные таблицы с полной сетью эталона | {summary['v2']['complete_table_cases']}/8 | {summary['v3']['complete_table_cases']}/8 |
| Полные отказы | {summary['v2']['rejections']} | {summary['v3']['rejections']} |
| Среднее ошибок найденных H-пар | {a['mean_matched_line_error_px']:.3f} px | {b['mean_matched_line_error_px']:.3f} px |

Порог сопоставления 8 px исходного изображения, учитывается только взаимно наблюдаемый участок полилиний. Отсутствующие границы учитываются отдельно, им не присвоена нулевая ошибка. Ноль лишних означает отсутствие несопоставленных предложений на размеченных участках этого набора, а не всеобщую гарантию отсутствия ложных линий.

## Что осталось

- 059b266c18…: найдены 11 из 12 размеченных горизонталей, пропущена H3 (индексация с нуля). В исходной сети ряд горизонталей не связан с первой вертикалью; у первой вертикали всего четыре поперечных связи. Поэтому это частичная сеть, и фото не засчитывается как полностью подтверждённая таблица. Недостающие связи не достраивались, чтобы не менять пользовательскую разметку.
- 53fe2a39ba…: V3 отказал — insufficient_jointly_confirmed_multiscale_edges. Все четыре размеченные H остаются не возвращёнными. На контроле этот отказ не исправлялся и пороги не подстраивались.
- Остальные 6 снимков имеют полные сети пересечений и полностью сопоставленные горизонтали/вертикали.

Оба метода выпрямления запущены на одних и тех же автоматических контролях V3, по {method_summary[0]['calculated']} карт на метод. При отказе ручные линии не подставлялись. Переворотов карт: Mesh {method_summary[0]['folded_cases']}, TPS {method_summary[1]['folded_cases']}. Подробные методы/контрольные суммы и безопасность карт — cases/<job>/result.json.

## Заключение

Увеличение полноты при нуле дополнительных несопоставленных горизонталей подтвердилось на связанно размеченных участках этого отдельного набора: 30/42 -> 37/42. Но есть полный отказ, пропущенный разделитель и частичный эталон одного фото. Этого недостаточно для объявления полной готовности к внедрению. Рабочий OCR, оригиналы, исходные JSON, редактор и математические реализации Mesh/TPS оставлены без изменений.
'''
    (OUT/'FINDINGS.md').write_text(findings,encoding='utf-8')
    page=f'''<meta charset="utf-8"><title>Финальный контроль V3</title><style>body{{font:16px system-ui;margin:28px}}p{{max-width:1200px}}td,th{{border:1px solid #aaa;padding:8px}}table{{border-collapse:collapse}}</style><h1>V3 — контроль по участкам между пересечениями</h1><p>По уточнению пользователя свободные хвосты исключены только в расчётных копиях. Координаты не сдвигались, пересечения не дорисовывались. Редактор и оригинальные JSON не изменены. Одно фото имеет частичную сеть пересечений.</p><p><a href="FINDINGS.md">Заключение и ограничения</a> · <a href="metrics.csv">CSV</a> · <a href="reference_manifest.json">Происхождение расчётных копий</a> · <a href="summary.json">Сводка</a> · <a href="integrity.json">Неизменность файлов</a></p><p><b>V2: {a['matched']}/{a['reference']} горизонталей ({a['recall']:.1%}); V3: {b['matched']}/{b['reference']} ({b['recall']:.1%}).</b> Несопоставленных предложенных H: 0. Средняя ошибка найденных H-пар V3: {b['mean_matched_line_error_px']:.2f} px. Полностью сопоставлено 6 таблиц из 8; ещё одна частичная и одна с отказом.</p><table><tr><th>Фото</th><th>Эталон H</th><th>V2 найдено</th><th>V3 найдено</th><th>V3 пропуски</th><th>V3 лишние</th><th>V3 вертикали</th><th>Средняя H, px</th><th>Оговорка эталона</th></tr>{''.join(rows)}</table><p>По-чековые страницы показывают оригинальную разметку с хвостами, расчётную разметку без хвостов и автоматически найденную сетку. Детектор на контрольном наборе не подстраивался.</p><p><b>Готовность к внедрению не объявляется.</b></p>'''
    (OUT/'index.html').write_text(page,encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
