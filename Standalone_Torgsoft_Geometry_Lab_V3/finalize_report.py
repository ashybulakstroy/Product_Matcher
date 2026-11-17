from pathlib import Path
import sys,json,hashlib,csv,collections,statistics,html,platform,importlib.metadata
import numpy as np,cv2
R=Path.cwd();L=R/'Standalone_Torgsoft_Geometry_Lab_V3';P=R/'experiment_outputs/standalone_geometry_v3';run=P/'real_visual_references';d=json.loads((run/'run.json').read_text(encoding='utf-8'));records=d['records']
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sig(f):
 st=f.stat();return dict(sha256=hashlib.sha256(f.read_bytes()).hexdigest(),size=st.st_size,mtime_ns=st.st_mtime_ns)
source_before=load(P/'source_before.json');protected_before=load(P/'protected_before.json')['protected']
changed_source=[name for name,v in source_before.items() if not (R/name).exists() or sig(R/name)!=v]
changed_code=[name for name,v in protected_before.items() if not (R/name).exists() or sig(R/name)!=v]
assert not changed_source,changed_source
assert not changed_code,changed_code
integrity=dict(source_files_checked=len(source_before),protected_files_checked=len(protected_before),changed_sources=changed_source,changed_protected=changed_code,status='UNCHANGED',before_file='source_before.json',protected_before_file='protected_before.json')
(P/'SOURCE_INTEGRITY.json').write_text(json.dumps(integrity,indent=2),encoding='utf-8')
summary={}
for method in d['metadata']['methods']:
 rows=[x for x in records if x['method']==method]
 summary[method]=dict(statuses=dict(collections.Counter(x['status'] for x in rows)),reference_statuses=dict(collections.Counter(x.get('ground_truth',{}).get('status','NO_RESULT') for x in rows)),mean_seconds=round(statistics.mean(x['duration_sec'] for x in rows),3))
(P/'results_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
with (P/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
 fields=['job','method','status','seconds','valid_fraction','folded_fraction','gt_status','mean_px','mean_px_at_width_1500','missing_horizontal','extra_horizontal']
 writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
 for row in records:
  m=row.get('metrics',{});g=row.get('ground_truth',{})
  writer.writerow(dict(job=row['job_id'],method=row['method'],status=row['status'],seconds=row['duration_sec'],valid_fraction=m.get('valid_fraction'),folded_fraction=m.get('folded_fraction'),gt_status=g.get('status','NO_RESULT'),mean_px=g.get('mean_px'),mean_px_at_width_1500=g.get('mean_px_at_width_1500'),missing_horizontal=g.get('missing_horizontal'),extra_horizontal=g.get('extra_horizontal')))
# Render independent visual source tracing, separately from model lines.
references={}
for f in L.joinpath('ground_truth').glob('*.json'):
 g=load(f)
 if g.get('annotation_origin')=='agent_visual_source_only':references[g['job_id']]=g
for case in sorted((run/'cases').iterdir()):
 meta=load(case/'case.json');job=meta['job_id'];g=references.get(job)
 if g:
  im=cv2.imread(str(case/'original.png'))
  for line in g['vertical_polylines']+g['horizontal_polylines']:
   cv2.polylines(im,[np.rint(line).astype(np.int32)],False,(0,170,0),2)
  cv2.imwrite(str(case/'independent_visual_reference.png'),im)
 e=html.escape
 sections=[]
 for row in [x for x in records if x['job_id']==job]:
  method=row['method'];pics=''
  for key,label in [('rectified','Выпрямление'),('overlay','Сетка и повторный детектор'),('source_overlay','Та же карта на оригинале')]:
   if row.get(key):pics+=f'<div>{label}<a href="{Path(row[key]).name}"><img src="{Path(row[key]).name}"></a></div>'
  sections.append(f'<section><h2>{method}: {row["status"]}</h2><p>{e(str(row.get("reason","")))}</p><p>Эталон: {e(str(row.get("ground_truth",{})))}</p><div class="grid">{pics}</div></section>')
 ref='<a href="independent_visual_reference.png">Независимая визуальная разметка (не проверена человеком)</a>' if g else 'Независимая разметка отсутствует: GT_MISSING'
 page='<!doctype html><html lang="ru"><meta charset="utf-8"><title>Geometry case '+job+'</title><style>body{font:15px system-ui;margin:25px;background:#f3f5f8;color:#203040}section{background:white;padding:16px;margin:16px 0}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:15px}img{width:100%;max-height:1100px;object-fit:contain;object-position:top}p{overflow-wrap:anywhere}</style><h1>'+job+'</h1><a href="../../summary.html">Сводка прогона</a> · <a href="original.png">Исходное фото</a><p>'+ref+'</p>'+''.join(sections)+'</html>'
 (case/'case.html').write_text(page,encoding='utf-8')
# Single entry point with direct links to all ten visual comparisons.
style='body{font:16px/1.5 system-ui;color:#203040;background:#f3f5f8;margin:25px;max-width:1500px}section{background:white;padding:22px;margin:18px 0;border-radius:10px}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}table{width:100%;border-collapse:collapse}img{width:180px;max-height:230px;object-fit:contain}a{color:#165aba}'
method_rows=''.join(f'<tr><td>{m}</td><td>{html.escape(str(s["statuses"]))}</td><td>{s["mean_seconds"]}</td><td>{html.escape(str(s["reference_statuses"]))}</td></tr>' for m,s in summary.items())
case_rows=''
for case in sorted((run/'cases').iterdir()):
 meta=load(case/'case.json');job=meta['job_id'];path='real_visual_references/cases/'+case.name
 case_rows+=f'<tr><td><a href="{path}/case.html"><img src="{path}/original.png"></a></td><td><a href="{path}/case.html">{job}</a><br>{meta["sha256"][:18]}…</td><td>{references.get(job,{}).get("stratum","Без визуальной разметки")}</td></tr>'
page='''<!doctype html><html lang="ru"><meta charset="utf-8"><title>Geometry Lab V3 — реальные испытания</title><style>'''+style+'''</style><h1>Standalone Geometry Lab V3: 10 реальных фото, 4 метода</h1><section><p><b>Решение: NONE_PROVEN — готовность новых методов не доказана.</b></p><p>Для Perspective, Grid Mesh и SciPy TPS получены безопасные карты на 7 из 10 фотографий, остальные 3 отклонены. Однако на всех трёх размеченных фото, где новые методы рассчитали карту, количество горизонталей не совпало с визуальной разметкой печатной таблицы. На двух других размеченных фото новые методы отказали. Нельзя считать SUCCESS доказательством правильных товарных строк.</p><p>Основная программа, её окружение, настройки и исходные фото не изменялись. Это изолированный геометрический Trial: OCR не запускался и не требуется заданием V3.</p></section><section><h2>Честность сравнения и разметка</h2><p>Найдено 50 уникальных SHA256 загрузок. Для первого прогона заранее выбраны 10 разных фотографий: короткие/длинные, волнистые, повреждённая, смятая, с перспективой. Это уникальные байты фото, не доказательство 50 разных физических чеков. Сильный нижний ракурс и резкий надлом в выбранном наборе не подтверждены.</p><p>На 5 оригиналах агент визуально отметил физические линии и проверил экспорт через annotate.html. Эта предварительная разметка независима от детектора и синей сетки, <b>но не подтверждена человеком</b>. Поэтому человеческих GT = 0; численные ошибки по этим линиям имеют диагностический смысл и не дают основания объявлять метод точным. У остальных 5 — GT_MISSING, а не 0% качества.</p><p>Baseline — существующие файлы конкретных заданий web_outputs. Это сохранённые результаты, часть которых относится к ранним обработкам; рабочий OCR заново не запускался. На повреждённом чеке архивный baseline покрывает только правую часть таблицы. Сравнение не является заявлением о качестве последней версии сервера.</p></section><section><h2>Фактические результаты</h2><table><tr><th>Метод</th><th>Статусы на 10 фото</th><th>Среднее, с</th><th>Статусы визуальных эталонов</th></tr>'''+method_rows+'''</table><p>Время включает обработку, измерение и запись артефактов при 4 воркерах, но baseline загружается из сохранённых файлов. По этим цифрам нельзя сравнивать скорость OCR или объявлять baseline быстрее.</p><a href="real_visual_references/summary.html">Подробная таблица методов</a> · <a href="metrics.csv">CSV метрик</a></section><section><h2>Откройте фото для сравнения методов</h2><table>'''+case_rows+'''</table></section><section><h2>Что доработано только внутри лаборатории</h2><p>Добавлены 4–8 воркеров и выбор фотографий. Вертикали сравниваются в общей наблюдаемой области, горизонтали требуют поддержки по ширине таблицы после объединения фрагментов. Первоначальный прогон без доработки сохранён: новые методы строили карту лишь на 1 из 10 фото.</p><p>Сохраняются обратные карты NPZ и координаты сетки. При уменьшении исходника обратные координаты пересчитываются с учётом центров пикселей, а финальное изображение выбирается непосредственно из оригинала одной картой. Отдельный тест проверил точное совпадение изображения и remap на уменьшенном входе. Значения метрик без наблюдений не подменяются нулями.</p><p>При этом детектор ещё путает печатные разделители и текст/частичные линии, теряет и добавляет строки. Сохранность названий и итогового блока визуально не гарантирована. Пороговые изменения использовали этот же исследовательский набор: независимого holdout-теста нет.</p><p><b>19 тестов прошли.</b> Для реальной приёмки нужна проверенная человеком разметка, улучшение детектора и отдельный отложенный набор. Синтетика не заменяет эти проверки.</p></section><section><h2>Артефакты</h2><a href="FINAL_FINDINGS.md">Выводы и ограничения</a> · <a href="SOURCE_INTEGRITY.json">Неизменность исходников</a> · <a href="real_initial/summary.html">Первоначальный прогон</a> · <a href="environment.txt">Изолированные зависимости</a><p>Лаборатория: Standalone_Torgsoft_Geometry_Lab_V3. Результаты сохранены в experiment_outputs по ранее указанному предпочтению пользователя; data/output не засорялся.</p></section></html>'''
(P/'summary.html').write_text(page,encoding='utf-8')
findings=f'''# Standalone Geometry Lab V3: NONE_PROVEN

Дата: 16.11.2026 (Asia/Qyzylorda).

Реальный набор: 10 заранее выбранных уникальных SHA256 из 50 доступных загрузок, 4 метода = 40 попыток. Воркеров: 4. По SHA исключены точные дубли; разные байты могут представлять один физический документ.

{json.dumps(summary,ensure_ascii=False,indent=2)}

## Окружение
Отдельное .venv в лаборатории. Python {platform.python_version()}; Windows {platform.platform()}. CPU: Intel Core i5-13400, 10 логических процессоров по Win32_Processor; RAM около 32 ГБ. Версии: numpy {importlib.metadata.version('numpy')}, OpenCV {cv2.__version__}, SciPy {importlib.metadata.version('scipy')}. BLAS/OpenMP ограничены одним потоком на воркер. Нейросетевых весов и внешних репозиториев нет.

## Разметка и ограничения
Пять предварительных agent_visual_source_only аннотаций по оригиналам, human_verified=false. Все экспортированы офлайн-редактором. Это не человеческий GT, готовность не доказана. На 5 фото GT_MISSING; на 3 рассчитанных новых кандидатах с разметкой GT_MISMATCH по числу горизонталей. Сильный нижний ракурс и резкий надлом не подтверждены. Пороговые доработки проверены на том же наборе: независимого holdout нет.

## Рекомендация
Не переносить новые методы в App. Сначала исправить независимое выделение физических разделителей, ручная проверка пяти разметок и новый отложенный набор. Mesh и TPS пока нельзя ранжировать: они используют один ошибающийся детектор. Безопасный Jacobian не доказывает правильные ячейки.

Baseline использует только существующие артефакты выбранных web_outputs, включая ранние обработки. Это не новый прогон текущей версии OCR. На повреждённом снимке сохранённая сетка baseline покрывает только правый фрагмент — это показано в отчёте, файлы не исправлялись.

## Изоляция
Проверено {len(source_before)} исходных файлов фото/карт/разметки и {len(protected_before)} файлов App/настроек: SHA256, размер и mtime не изменились. Рабочий сервер не перезапускался. Данные не отправлялись во внешние API. Основная среда Python не менялась. Все изменения кода внутри новой лаборатории, отчёты в experiment_outputs.

## Команды
- .venv/Scripts/python.exe -m pytest -q tests (из каталога лаборатории): 19 passed.
- run.py discover --project-root .. --max-cases 0
- run.py benchmark --project-root .. --output ../experiment_outputs/standalone_geometry_v3 --run-id real_visual_references --case-ids <10 IDs в selected.json> --max-cases 10 --workers 4

Первые два прогона сохранены как real_initial и real_supported, итоговый — real_visual_references. Метрики не импортировались из V1/V2 или прежних Trial.
'''
(P/'FINAL_FINDINGS.md').write_text(findings,encoding='utf-8')
(P/'GT_STATUS.json').write_text(json.dumps(dict(visual_agent_annotations=5,human_verified=0,missing=5,decision='NONE_PROVEN'),indent=2),encoding='utf-8')
print('Report generated; protected and source hashes unchanged')
