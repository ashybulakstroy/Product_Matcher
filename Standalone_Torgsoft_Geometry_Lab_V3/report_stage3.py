"""Aggregate diagnostics separately from the frozen independent controls."""
import csv,hashlib,json,statistics
from pathlib import Path
import cv2,numpy as np
from stage3 import ROOT,LAB,OUT,write,audit_miss
from geometry_lab.structure_diagnostics import match_lines


def totals(results,key):
    tp=fn=fp=0;dist=[]
    for r in results:
        gt=json.loads((LAB/'confirmed'/f"{r['sha256']}.json").read_text(encoding='utf-8'))
        m=r[key]['horizontal'] if r[key] else None
        tp+=len(m['matches']) if m else 0;fn+=len(m['missing']) if m else len(gt['horizontal_polylines']);fp+=len(m['false']) if m else 0
        if m:dist.extend(p['mean_px'] for p in m['matches'])
    return dict(matched=tp,missing=fn,false=fp,reference=tp+fn,
        recall=tp/(tp+fn),precision=tp/(tp+fp) if tp+fp else None,
        mean_matched_line_error_px=statistics.mean(dist) if dist else None,
        p95_matched_line_mean_error_px=float(np.percentile(dist,95)) if dist else None)


def physical_causes(job,line):
    # Agent source-image inspection of every saved crop; no detector feedback.
    # Physical causes are visual diagnoses, distinct from the measured gate loss.
    if job.startswith('1d0681'):
        return ['weak_print' if line in [0,1,2] else 'fragmented_thin_print',
                'mild_curve', 'text_close_to_rule' if line in [1,5,6] else 'no_clear_fold_at_rule']
    if job.startswith('1d6a1'):
        return ['weak_low_contrast_print','thin_interrupted_stroke',
                'near_header_text' if line<=1 else 'text_close_to_rule' if line in [2,3,6,7,8,11] else 'footer_transition']
    if job.startswith('489c'):
        return ['weak_low_contrast_upper_print' if line<=12 else 'visible_rule_rejected_by_fragment_gates',
                'short_tightly_spaced_rows' if line<=12 else 'local_print_breaks',
                'text_close_to_rule' if line in [2,3,4,5,9,10,11,12,13,14,15,17,19,20,21,27] else 'no_clear_fold_at_rule']
    if job.startswith('4eae'):return ['curved_header','wrinkled_sheet_fold','thin_print']
    if job.startswith('6377'):
        return ['curve_at_sheet_crease' if line<=1 else 'physically_clear_rule',
                'v1_rejected_whole_grid_due_to_vertical_proposals']
    if job.startswith('ac2f'):return ['thin_curved_footer_rule','fragment_endpoint_gate']
    return ['needs_source_visual_confirmation']


def main():
    results=json.loads((OUT/'results.json').read_text());before=totals(results,'before');after=totals(results,'after')
    audits=[];regressions=[];rows=[]
    for r in results:
        job=r['job'];folder=OUT/'cases'/job;gt=json.loads((LAB/'confirmed'/f"{r['sha256']}.json").read_text(encoding='utf-8'))
        old_missing=set(r['before']['horizontal']['missing']) if r['before'] else set(range(len(gt['horizontal_polylines'])))
        new_missing=set(r['after']['horizontal']['missing']) if r['after'] else set(range(len(gt['horizontal_polylines'])))
        regression=sorted(new_missing-old_missing)
        regressions.extend(dict(job=job,line=i) for i in regression)
        for a in r['miss_audit']:
            a['visual_physical_causes']=physical_causes(job,a['reference_line']);a['recovered']=a['reference_line'] not in new_missing
            a['diagnosis_origin']='agent_visual_source_crops_and_measured_gates_not_human_cause_label'
            audits.append(dict(job=job,**a))
        for i in regression:
            image=cv2.imread(str(ROOT/next(c['image'] for c in json.loads((ROOT/'experiment_outputs/standalone_geometry_v3/selected.json').read_text()) if c['job']==job)))
            oldreport=json.loads((ROOT/'experiment_outputs/standalone_geometry_v3_confirmed/cases'/job/'physical_graph.json').read_text())
            a=audit_miss(image,gt,i,oldreport,folder);a.update(job=job,visual_physical_causes=physical_causes(job,i),new_regression=True,recovered=False);audits.append(a)
        old=r['before']['horizontal'] if r['before'] else None;new=r['after']['horizontal'] if r['after'] else None
        rows.append(f'<tr><td><a href="cases/{job}/index.html">{job[:10]}</a></td><td>{len(gt["horizontal_polylines"])}</td><td>{len(old["matches"]) if old else 0}</td><td>{len(new["matches"]) if new else 0}</td><td>{new["missing"] if new else "отказ"}</td><td>{new["false"] if new else "—"}</td><td>{new["mean_matched_px"] if new else "—"}</td></tr>')
    write(OUT/'misses_audit.json',audits)
    for job in {a['job'] for a in audits}:
        path=OUT/'cases'/job/'index.html';text=path.read_text(encoding='utf-8')
        if 'Все причины' not in text:text=text.replace('<h2>', '<p><a href="../../misses_audit.json">Все причины пропусков, восстановленные линии и регрессии</a></p><h2>',1);path.write_text(text,encoding='utf-8')
    # Sensitivity prevents a single generous matching tolerance hiding errors.
    sensitivity={}
    for tolerance in [4.,8.,12.]:
        copies=[]
        for r in results:
            gt=json.loads((LAB/'confirmed'/f"{r['sha256']}.json").read_text(encoding='utf-8'))
            p=OUT/'cases'/r['job']/'controls.json'
            copies.append(dict(sha256=r['sha256'],after=match_lines(json.loads(p.read_text()),gt,gt['source_shape'],tolerance) if p.exists() else None))
        sensitivity[str(tolerance)]=totals(copies,'after')
    summary=dict(before=before,after=after,matching_tolerance_px=8,threshold_sensitivity=sensitivity,
        prior_misses_audited=sum(len(r['miss_audit']) for r in results),regressions=regressions,
        workers=4,independent_accuracy='PENDING_HUMAN_CONTROL_ANNOTATIONS')
    write(OUT/'summary.json',summary)
    with (OUT/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['job','stage','reference','matched','missing','false','mean_matched_px'])
        for r in results:
            for key in ['before','after']:
                m=r[key]['horizontal'] if r[key] else None
                w.writerow([r['job'],key,m['reference'] if m else None,len(m['matches']) if m else 0,m['missing'] if m else 'REJECTED',m['false'] if m else None,m['mean_matched_px'] if m else None])
    with (OUT/'misses.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['job','reference_line','recovered','new_regression','physical_causes','algorithmic_causes','ridge_mean','observed_fraction','chord_deviation_px'])
        for a in audits:w.writerow([a['job'],a['reference_line'],a['recovered'],a.get('new_regression',False),' | '.join(a['visual_physical_causes']),' | '.join(a['algorithmic_reasons']),a['source_ridge_mean'],a['observed_sample_fraction'],a['chord_deviation_px']])
    method_rows=[]
    for method in ['grid_mesh','tps_scipy']:
        entries=[e for r in results for e in r['methods'] if e['method']==method]
        folded=sum(e.get('safety',{}).get('folded_fraction',0)>0 for e in entries)
        valid=sum(e['status']=='CALCULATED_NOT_VALIDATED' for e in entries)
        matched=sum(len(e.get('against_confirmed',{}).get('horizontal',{}).get('matches',[])) for e in entries)
        method_rows.append(f'<tr><td>{method}</td><td>{valid}/10</td><td>{folded}</td><td>{matched}</td></tr>')
    audit_rows=[]
    for a in audits:
        link=f'cases/{a["job"]}/{a["original_crop"]}'
        audit_rows.append(f'<tr><td>{a["job"][:8]} H{a["reference_line"]}</td><td>{"восстановлена" if a["recovered"] else "регрессия" if a.get("new_regression") else "пропуск"}</td><td>{", ".join(a["visual_physical_causes"])}</td><td>{", ".join(a["algorithmic_reasons"])}</td><td><a href="{link}">Фрагмент оригинала</a></td></tr>')
    page=f'''<meta charset="utf-8"><title>Physical Grid Graph: этап 3</title><style>body{{font:16px system-ui;margin:28px}}p{{max-width:1150px}}table{{border-collapse:collapse}}td,th{{padding:8px;border:1px solid #aaa}}</style><h1>Physical Grid Graph — этап 3</h1><p><a href="control_review.html"><b>Контрольный набор: редактор связанных узлов</b></a> · <a href="FINDINGS.md">Заключение</a> · <a href="metrics.csv">Метрики CSV</a> · <a href="misses.csv">Все причины CSV</a> · <a href="summary.json">Сводка JSON</a> · <a href="integrity.json">Неизменность OCR и методов</a></p><p>10 подтверждённых фотографий — диагностический набор. Новый контроль — 8 других фото, не использованных при настройке этой версии; требуется ручная проверка всех границ и независимости чеков. Начальная сетка контроля — только предложения детектора, не эталон.</p><p><b>Полнота: {before['matched']}/{before['reference']} ({before['recall']:.1%}) → {after['matched']}/{after['reference']} ({after['recall']:.1%}).</b> Несопоставленных предложенных горизонталей: {before['false']} → {after['false']}. Порог сопоставления 8 px. Это результаты диагностики, не независимая точность.</p><p>Средняя ошибка сопоставленных горизонталей: {before['mean_matched_line_error_px']:.2f} → {after['mean_matched_line_error_px']:.2f} px. Она измерена только для найденных пар; пропуски учитываются отдельно. Регрессии: {regressions}.</p><table><tr><th>Фото</th><th>Подтверждено</th><th>Найдено до</th><th>Найдено после</th><th>Пропуски после</th><th>Лишние после</th><th>Ошибка пар, px</th></tr>{''.join(rows)}</table><h2>Grid Mesh и TPS на автоматических линиях</h2><p>Исходники методов не изменены. Одни и те же автоматические контроли переданы обоим методам; карты не могут восстановить отсутствующие строки. Отсутствие переворота не означает сохранность всего чека или точность координат.</p><table><tr><th>Метод</th><th>Построено</th><th>Фото с переворотом карты</th><th>Сопоставленные горизонтали после преобразования</th></tr>{''.join(method_rows)}</table><h2>Разбор каждой пропущенной линии</h2><p>Причины физического качества — визуальная диагностика агента по оригинальным фрагментам; названия складок/разрывов не выводились автоматически из одного низкого контраста. Измеренные причины отсечения сохранены отдельно. Проверены все 43 прежних пропуска, а также новая регрессия.</p><table><tr><th>Линия</th><th>Итог</th><th>Наблюдения на фото</th><th>Механизм потери</th><th>Оригинал</th></tr>{''.join(audit_rows)}</table><p><b>К внедрению не готово.</b> Остались пропуски, независимая проверка ожидает подтверждённую разметку.</p>'''
    (OUT/'index.html').write_text(page,encoding='utf-8')
    findings=f'''# Physical Grid Graph, этап 3

Диагностический набор: 10 ранее подтверждённых человеком оригиналов. Разметки не менялись. Входы детектора — только изображение, без номеров чеков, ожидаемого количества строк, OCR или координат эталона. Отдельный V2 модуль: geometry_lab/physical_grid_graph_local.py. Предыдущий граф сохранён без изменений для сравнения.

## Измерения

При пороге сопоставления 8 px полнота горизонталей выросла с {before['matched']}/{before['reference']} ({before['recall']:.1%}) до {after['matched']}/{after['reference']} ({after['recall']:.1%}). Несопоставленных предложений после: {after['false']}. Средняя ошибка найденных пар: {after['mean_matched_line_error_px']:.3f} px; 95-й процентиль СРЕДНИХ ошибок линий: {after['p95_matched_line_mean_error_px']:.3f} px. Это не p95 всех точек. Все пропуски/FР считаются отдельно; ошибка найденных пар не заменяет полноту. Чувствительность к 4/8/12 px — summary.json.

Разобраны 43 прежних пропуска: оригинальный фрагмент, наложение подтверждённой линии, измерение контраста, изгиба, ближайшие принятые/отсечённые предложения V1. Причины и исходные координаты — misses_audit.json; удобная таблица — index.html и misses.csv. Визуальные физические причины записаны агентом, они не заменяют человеческое заключение о качестве печати. Новая регрессия отдельно: {regressions}.

## Причины

На верхней части длинных чеков преобладают слабая печать и тонкие прерывистые линии. Глобальный Hough терял короткие/изогнутые фрагменты; V1 дополнительно требовал почти полных концов у границ таблицы. Часть отчётливо видимых линий отбрасывалась по покрытию колонок и области наблюдения вертикалей. У помятого заголовка и лестницы существенен изгиб около складки. Надписи рядом с линиями мешают группировке; сами надписи также способны образовать ложный путь. По одному контрасту невозможно достоверно различить складку и повреждение печати — это не скрыто автоматической меткой.

## Изменения только детектора

Локальный отклик тёмного горизонтального штриха вместо единственного порога всей страницы; семена из реально измеренных фрагментов и коротких полос; ограниченное прослеживание пути с сохранением исходного семени. Гладкая модель следа применяется только как предложение, затем повторно проверяется по исходным чернилам. Прямая наблюдаемая линия сохраняется, если она лучше подтверждается распределёнными чернилами, чем прослеженный путь. Ни шаг строк, ни желаемое количество не используются.

Пять вертикалей проверяются по повторяющимся наблюдаемым пересечениям. Вертикальные фрагменты дополнительно подгоняются в области таблицы, чтобы не включать продолжения штрихкода. Внешние границы уточняются по печатным T-пересечениям: длина края бумаги больше не является достаточным признаком. Приём горизонтали требует распределённого следа во всех четырёх колонках, наблюдаемых подходов к минимум трём вертикалям и ограничения большого непроверенного разрыва. Промежутки помечены observed=false; интерполяция сама по себе не подтверждает существование линии.

Строгое подтверждение убрало найденные при разработке ложные линии через заголовки и цифры цен. Цена этого компромисса — оставшиеся слабые линии и одна регрессия. Полнота вертикалей, семантическое распознавание НДС/итогов и размер кадра с полным блоком итогов не объявляются решёнными.

## Преобразования

Grid Mesh и TPS повторно запущены на автоматических линиях каждой фотографии с workers=4. Их исходники не изменялись; fingerprints входов одинаковы для пары методов. Метрики пропусков/лишних линий после обратной проекции, карты безопасности и время — cases/<job>/result.json. Неравное количество линий не превращалось в нулевую ошибку. Подробные картинки показывают обрезку при пропущенных верхних/нижних границах.

## Независимая проверка

Новые 8 фотографий зафиксированы в control_frozen.json до работы над V2. Они не входят ни в десять диагностических, ни в прежний резерв из восьми фото. До формирования черновых сеток зафиксирован код control_code_frozen.json; после просмотра контрольных снимков код не подстраивается. Различие SHA ещё не доказывает независимость физических чеков: редактор требует подтвердить, что это другой чек. При повторе нужен заменяющий снимок, а не включение повтора в статистику.

control_review.html хранит одну матрицу пересечений: узел одновременно принадлежит вертикали и горизонтали. Обычный drag двигает узел, Shift — строку, Alt — колонку. Можно добавить/удалить границу, скрыть сетку, менять масштаб и координаты. JSON строит обе семьи линий из одной матрицы и проверяет порядок узлов. Начальная автоматическая сетка НЕ считается эталоном: пользователь должен найти пропущенные и удалить ложные границы.

Подтверждённые файлы сохранять в Standalone_Torgsoft_Geometry_Lab_V3/control_confirmed. Пока они не получены, независимая точность PENDING. Предыдущие подтверждённые JSON и рабочий OCR не изменены. Решение о внедрении не принимается.
'''
    (OUT/'FINDINGS.md').write_text(findings,encoding='utf-8')
    protected=json.loads((OUT/'protected_before.json').read_text());changed=[]
    for name,old in protected.items():
        p=ROOT/name
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=old['sha256'] or p.stat().st_mtime_ns!=old['mtime_ns']:changed.append(name)
    frozen=json.loads((OUT/'control_code_frozen.json').read_text());code_changed=[name for name,digest in frozen.items() if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest]
    write(OUT/'integrity.json',dict(protected_files=len(protected),changed=changed,control_candidate_changed=code_changed))
    assert not changed and not code_changed,(changed,code_changed)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
