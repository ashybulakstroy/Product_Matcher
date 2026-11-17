"""Reports development performance without relabelling it independent accuracy."""
import json,csv,hashlib,statistics
from pathlib import Path
import cv2,numpy as np
from stage4 import ROOT,LAB,OUT,cases,write


def aggregate(rows,version,axis):
    tp=fn=fp=0;errors=[];max_errors=[];rejected=0
    for r in rows:
        m=r[version]['match'];ref=r['reference'] if axis=='horizontal' else 5
        if m:
            x=m[axis];tp+=len(x['matches']);fn+=len(x['missing']);fp+=len(x['false']);errors.extend(p['mean_px'] for p in x['matches'])
            if x.get('max_sample_error_px') is not None:max_errors.append(x['max_sample_error_px'])
        else:fn+=ref;rejected+=1
    full=sum(bool(r[version]['match'] and not any(r[version]['match'][a]['missing'] or r[version]['match'][a]['false'] for a in ['horizontal','vertical'])) for r in rows)
    return dict(matched=tp,missing=fn,false=fp,reference=tp+fn,recall=tp/(tp+fn),
        mean_matched_line_error_px=statistics.mean(errors) if errors else None,
        max_matched_sample_error_px=max(max_errors) if max_errors else None,
        full_table_cases=full,cases=len(rows),rejections=rejected,
        total_case_seconds=sum(r[version]['seconds'] for r in rows))


def main():
    results=json.loads((OUT/'results.json').read_text());groups={}
    for group in ['diagnostic10','former_control8_now_regression','all18']:
        subset=results if group=='all18' else [r for r in results if r['group']==group]
        groups[group]={v:{a:aggregate(subset,v,a) for a in ['horizontal','vertical']} for v in ['v2','v3']}
    write(OUT/'summary.json',dict(groups=groups,final_independent_status='PENDING_SEPARATE_CONTROL_LABELS',workers=4,match_tolerance_px=8))
    fail_notes={
        '112ea4224c734413b9e61ddb8be08f95':'V2 выбрал фоновые вертикали слева от чека, а не таблицу (позиции кандидатов 25, 44, 192, 235, 312 вместо границ таблицы около 123, 158, 338, 395, 475). Короткие физические вертикали потерялись в глобальном лимите сильнейших кандидатов. V3 использует многомасштабные фрагменты в локальном окне горизонтали. Дополнительно учитывает скошенные окончания вертикальных фрагментов: один общий интервал Y не обязан содержать все горизонтали. Восстановлены 5 вертикалей и 4 горизонтали по чернилам исходника.',
        '49ad1a1d9502487aa959e1fe0d41d39f':'V2 принял край бумаги и первую печатную границу за первые две колонки, потеряв границу номера и наименования. Поэтому покрытие первой колонки было 0, хотя остальные колонки имели покрытие 1. Многомасштабные наблюдения восстановили короткую вертикаль у имени товара; V3 нашёл все 15 горизонталей и 5 вертикалей.',
        'ada2876336214faf87457e6b257df3f3':'Та же ошибка структуры: V2 выбрал край бумаги x≈78 вместо печатной границы x≈117 и границу x≈118 вместо x≈165. Все хорошие горизонтали отклонялись по первой колонке. V3 выбрал наблюдаемые печатные вертикали, найдено 6 горизонталей и 5 вертикалей.',
        '1dde6b99bad74ac98ed63e3da4634c69':'Левая внешняя печатная граница находится за левым краем фотографии. На исходнике видны четыре физические вертикали, но не пятая. Ручная разметка ставит левую границу около x=0; это координата края кадра, не доказательство печатной линии. V3 честно отказывает в полной пятивертикальной модели. Устранить этот отказ без предположения о невидимой части нельзя; нужен более полный снимок либо отдельная явно частичная модель. Счёт отказов/пропусков в сырых метриках оставлен, не исключён ради улучшения результата.'}
    audit=[];byjob={c['job']:c for c in cases()}
    for job,note in fail_notes.items():
        c=byjob[job];gt=json.loads((LAB/c['labels']/f"{c['sha256']}.json").read_text(encoding='utf-8'))
        im=cv2.imread(str(ROOT/c['image']));points=np.array([p for line in gt['horizontal_polylines'] for p in line]);x0=max(0,int(points[:,0].min())-20);x1=min(im.shape[1],int(points[:,0].max())+21);y0=max(0,int(points[:,1].min())-20);y1=min(im.shape[0],int(points[:,1].max())+21)
        folder=OUT/'cases'/job;cv2.imwrite(str(folder/'problem_source.jpg'),im[y0:y1,x0:x1])
        audit.append(dict(job=job,diagnosis=note,problem_crop=f'cases/{job}/problem_source.jpg'))
    write(OUT/'four_failures.json',audit)
    with (OUT/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['job','group','version','axis','reference','matched','missing','false','mean_matched_px','max_matched_sample_px','seconds','refusal'])
        for r in results:
            for version in ['v2','v3']:
                for axis in ['horizontal','vertical']:
                    m=r[version]['match'];a=m[axis] if m else None
                    w.writerow([r['job'],r['group'],version,axis,a['reference'] if a else r['reference'] if axis=='horizontal' else 5,len(a['matches']) if a else 0,a['missing'] if a else 'ALL_NOT_RETURNED',a['false'] if a else None,a['mean_matched_px'] if a else None,a.get('max_sample_error_px') if a else None,r[version]['seconds'],r[version].get('reason')])
    rows=[]
    for r in results:
        v2=r['v2']['match'];v3=r['v3']['match'];h=v3['horizontal'] if v3 else None;v=v3['vertical'] if v3 else None
        rows.append(f'<tr><td><a href="cases/{r["job"]}/index.html">{r["job"][:10]}</a></td><td>{r["reference"]}</td><td>{len(v2["horizontal"]["matches"]) if v2 else "отказ"}</td><td>{len(h["matches"]) if h else "отказ"}</td><td>{h["false"] if h else "—"}</td><td>{len(v["matches"]) if v else 0}/5</td><td>{h["mean_matched_px"] if h else "—"}</td><td>{round(h["max_sample_error_px"],2) if h else "—"}</td><td>{r["v3"]["seconds"]:.2f}</td></tr>')
        folder=OUT/'cases'/r['job'];blocks=[f'<h1>{r["job"]}</h1><p>Разработка/регрессия. Не независимая проверка V3.</p>']
        if r['job'] in fail_notes:blocks.append('<p>'+fail_notes[r['job']]+'</p>')
        for name in ['original','reference','v2','v3','problem_source','grid_mesh','tps_scipy']:
            if (folder/(name+'.jpg')).exists():blocks.append(f'<figure><figcaption>{name}</figcaption><a href="{name}.jpg"><img src="{name}.jpg"></a></figure>')
        blocks.append('<p><a href="result.json">Результат</a> · <a href="v3_report.json">Физические доказательства и причины отказа V3</a></p>')
        (folder/'index.html').write_text('<meta charset="utf-8"><style>body{font:16px system-ui}figure{display:inline-block;vertical-align:top;width:30%}img{max-width:100%}</style>'+''.join(blocks),encoding='utf-8')
    table_rows=[]
    for group,versions in groups.items():
        for version in ['v2','v3']:
            h=versions[version]['horizontal'];v=versions[version]['vertical']
            table_rows.append(f'<tr><td>{group}</td><td>{version}</td><td>{h["matched"]}/{h["reference"]} ({h["recall"]:.1%})</td><td>{h["false"]}</td><td>{v["matched"]}/{v["reference"]}</td><td>{h["full_table_cases"]}/{h["cases"]}</td><td>{h["rejections"]}</td><td>{h["mean_matched_line_error_px"]:.2f}</td><td>{h["max_matched_sample_error_px"]:.2f}</td><td>{h["total_case_seconds"]:.2f}</td></tr>')
    new=groups['former_control8_now_regression']['v3']['horizontal'];old=groups['former_control8_now_regression']['v2']['horizontal']
    failures=''.join(f'<li><a href="{a["problem_crop"]}">{a["job"][:10]}: фрагмент оригинала</a> — {a["diagnosis"]}</li>' for a in audit)
    page=f'''<meta charset="utf-8"><title>Geometry Lab: V3 и новый редактор</title><style>body{{font:16px system-ui;margin:25px}}p{{max-width:1200px}}td,th{{border:1px solid #aaa;padding:7px}}table{{border-collapse:collapse}}li{{margin:14px 0;max-width:1250px}}</style><h1>Physical Grid Graph V3 + редактор сетки</h1><p><a href="editor.html"><b>Редактор 18 размеченных фотографий</b></a> · <a href="final_control_editor.html"><b>Отдельный финальный контроль V3: 8 фото</b></a> · <a href="FINDINGS.md">Заключение</a> · <a href="metrics.csv">CSV</a> · <a href="summary.json">JSON</a> · <a href="integrity.json">Неизменность рабочего OCR и методов</a></p><p>Три из четырёх отказов V2 исправлены. На прежних восьми контрольных фото полнота горизонталей выросла с {old['matched']}/{old['reference']} до {new['matched']}/{new['reference']}, лишних горизонталей 0. Но эти восемь фото участвовали в разработке V3 и теперь являются регрессионными. Независимая точность V3 пока НЕ установлена.</p><h2>Сравнение</h2><p>Порог сопоставления 8 px. Ошибки считаются только для сопоставленных линий на взаимно наблюдаемом участке. Max — максимум ошибок координат выборки, не только максимальное среднее. Время — сумма длительностей кейсов, не время стены параллельного прогона. Время V3 включает консервативную ветку V2.</p><table><tr><th>Набор</th><th>Версия</th><th>Горизонтали</th><th>Лишние H</th><th>Вертикали</th><th>Полная таблица</th><th>Отказы</th><th>Средняя H, px</th><th>Max H, px</th><th>Сумма секунд</th></tr>{''.join(table_rows)}</table><h2>Четыре отказа V2</h2><ul>{failures}</ul><h2>По фотографиям</h2><table><tr><th>Фото</th><th>H эталон</th><th>V2 H</th><th>V3 H</th><th>V3 лишние H</th><th>V3 V</th><th>Средняя H, px</th><th>Max H, px</th><th>Время V3, s</th></tr>{''.join(rows)}</table><p>Mesh и TPS запущены на одинаковых автоматических линиях V3, исходники методов не изменены. Результаты и безопасность карт — в по-чековых JSON; отсутствие карты при отказе не заменялось ручными контролями.</p><h2>Редактор</h2><p>Непрерывные клики рисуют горизонтали или вертикали, Esc завершает режим, Ctrl+Z отменяет. Общие узлы можно двигать; двойной клик по линии добавляет изгиб. Скачать результат сохраняет JSON сразу, Скачать все — один ZIP с manifest.json. Галочек, имени и диалогов подтверждения нет. Простое открытие/сохранение не меняет координаты. Новое сохранение не получает подтверждения автоматически.</p><p><b>К внедрению не готово:</b> необходимы отдельный контроль, проверка обрезанных оригиналов и оценка скорости.</p>'''
    (OUT/'index.html').write_text(page,encoding='utf-8')
    findings=f'''# Physical Grid Graph V3 и редактор

## Итог разработки

На прежнем контрольном наборе V2 находил 24/53 горизонталей, V3 — 49/53. Полных таблиц: 4/8 -> 7/8. Лишних горизонталей: 0 -> 0. Три отказа устранены; один оставлен честным отказом из-за обрезанной физической границы. На диагностических 10 фотографиях сохранены 102/115 горизонталей без новых пропусков/ложных границ. Суммарно на 18 development/regression-фото: 126/168 -> 151/168. Это НЕ независимый результат V3.

## Детектор

Новый модуль geometry_lab/physical_grid_graph_v3.py. Изображения исследуются на масштабах 1 и 1,5, используются сегменты LSD после локального контраста, объединение совместимых фрагментов и устойчивое приближение наблюдаемого следа. Пересечения оцениваются по исходным чернилам и подходам к вертикалям. Локальные окна горизонтальных наблюдений предотвращают вытеснение короткой таблицы длинными краями бумаги/плакатов в глобальном лимите кандидатов. Для скошенных/прерванных окончаний вертикалей используется их совместная область наблюдений, а не требование, чтобы все горизонтали помещались в один общий Y-интервал пяти фрагментов.

Порог чернил, покрытие всех четырёх колонок и минимум трёх подтверждённых подходов к пересечениям сохранены от V2. Нет глобального снижения порога ради одного чека. Принятие требует как минимум трёх горизонталей с физическими доказательствами. Число строк OCR, номера чеков и эталонные координаты детектору не передаются. Неудачная новая гипотеза возвращает прежний подтверждённый по исходным пикселям граф V2, если тот существует. Ненаблюдаемые промежутки явно отмечены, не считаются обнаруженными линиями.

## Четыре отказа

Подробные причины: four_failures.json, исходные фрагменты и визуализации — по-чековые HTML. Первичные ошибки трёх случаев были в вертикальной структуре (край бумаги/фоновая линия или первый столбец вместо границы имени). У 1dde6b99… печатная левая граница обрезана: генерировать её по ожиданию пяти вертикалей нельзя. Сырые метрики сохраняют полный отказ, недоступная граница не исключена из знаменателя ради результата.

## Измерения и пределы

Полнота горизонталей и вертикалей, ложные линии, средняя и максимальная ошибка координат, доля полных таблиц, причины отказов и время — summary.json/metrics.csv/index.html. Максимум считается по выборке исходных координат сопоставленных линий; отсутствующим линиям не присваивается нулевая ошибка. На этих наборах V3 медленнее V2 из-за двух масштабов и проверки гипотез; точность найденных пар сама по себе не доказывает полноту. Grid Mesh и SciPy TPS не переписаны; прогнаны на автоматически найденных контролях V3.

## Редактор

grid_editor.html + editor_documents.py — общий шаблон следующих редакторов. Непрерывное рисование H/V, связанные пересечения, изгибы, перетаскивание строк/колонок, Ctrl+Z, Esc, область таблицы правой кнопкой. JSON сохраняется одним кликом даже пустым и без правок; ZIP содержит все текущие документы и manifest.json. Обязательных галочек, автора и диалогов нет. Исходные координаты при открытии/сохранении неизменны. Дополнительная схема графа содержит ссылки на общие узлы, но совместимые horizontal_polylines/vertical_polylines всегда остаются. Загрузчик reference_data проверен с узлами изгиба. Сохранение не равно подтверждению: новые/изменённые данные human_verified=false; уже подтверждённые исходные документы сохраняют прежний статус только без правок.

## Финальный контроль

В final_control_frozen.json выделены 8 фото из ранее неразмеченного резерва. Они не использовались при настройке V3; старые результаты V1/V2 для этого резерва ранее существовали, поэтому это не набор, впервые увиденный системой. Разметки V3 пока нет. Код V3 зафиксирован в final_candidate_frozen.json до открытия финального редактора; после этого не подстраивался. Финальные фотографии намеренно открываются без навязанной автоматической сетки. Разметку можно сохранить без подтверждения, а для окончательной оценки требуется отдельное явное подтверждение проверки человеком и независимости физического чека.

Критерий успеха на независимом контроле ещё НЕ проверен. Внедрение в рабочий OCR не выполнялось. Рабочее приложение, исходные фото, web_outputs, прежние эталоны и математические реализации Mesh/TPS не изменены.
'''
    (OUT/'FINDINGS.md').write_text(findings,encoding='utf-8')
    protected=json.loads((OUT/'protected_before.json').read_text());changed=[]
    for name,old in protected.items():
        p=ROOT/name
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=old['sha256'] or p.stat().st_mtime_ns!=old['mtime_ns']:changed.append(name)
    frozen=json.loads((OUT/'final_candidate_frozen.json').read_text());mutated=[n for n,v in frozen.items() if hashlib.sha256((ROOT/n).read_bytes()).hexdigest()!=v]
    write(OUT/'integrity.json',dict(protected_files=len(protected),changed=changed,frozen_candidate_changed=mutated));assert not changed and not mutated,(changed,mutated)
    print('Report written. Protected files unchanged:',len(protected))


if __name__=='__main__':main()
