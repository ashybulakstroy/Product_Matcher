"""Run the frozen V3 detectors/solvers against human-reviewed source lines.

Keeps stage2 exploratory outputs intact. No tuning on these annotations.
"""
from pathlib import Path
import concurrent.futures,csv,hashlib,html,json,os,statistics,time
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import cv2
import stage2
from geometry_lab.structure_diagnostics import reference_data,draw_lines

ROOT=stage2.ROOT;LAB=stage2.LAB
OUT=ROOT/'experiment_outputs/standalone_geometry_v3_confirmed'
CONFIRMED=LAB/'confirmed'


def init_worker():
    stage2.OUT=OUT


def snapshot(files):
    return {str(p.relative_to(ROOT)):dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                mtime_ns=p.stat().st_mtime_ns) for p in files if p.is_file()}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cases=json.loads((stage2.PREVIOUS/'selected.json').read_text())
    validated=[];manifest=[]
    for c in cases:
        p=CONFIRMED/(c['sha256']+'.json');gt=json.loads(p.read_text(encoding='utf-8'))
        if not stage2.confirmed_reference(gt,c):raise ValueError('Invalid human confirmation: '+p.name)
        reference_data(gt,c['shape'])
        validated.append((c,gt))
        manifest.append(dict(job=c['job'],file=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            horizontal=len(gt['horizontal_polylines']),vertical=len(gt['vertical_polylines']),reviewed_at=gt['reviewed_at']))
        source=ROOT/'experiment_outputs/standalone_geometry_v3_stage2/references'/p.name
        target=OUT/'references'/p.name;target.parent.mkdir(exist_ok=True)
        target.write_bytes(source.read_bytes())
    stage2.write(OUT/'confirmed_manifest.json',manifest)
    frozen=json.loads((ROOT/'experiment_outputs/standalone_geometry_v3_stage2/code_before_holdout.json').read_text())
    changes=[name for name,digest in frozen.items() if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest]
    if changes:raise RuntimeError('Frozen benchmark changed: '+repr(changes))
    files=[ROOT/name for name in json.loads((ROOT/'experiment_outputs/standalone_geometry_v3_stage2/protected_before.json').read_text())]
    before=snapshot(files);stage2.write(OUT/'protected_before.json',before)
    t=time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=4,initializer=init_worker) as pool:
        results=list(pool.map(stage2.run_case,[(c,str(CONFIRMED)) for c in cases]))
    stage2.write(OUT/'results.json',results)
    summary=[];rows=[]
    for r,(c,gt) in zip(results,validated):
        if r['human_confirmed_comparison']!='AVAILABLE':raise RuntimeError(r['human_confirmed_comparison'])
        folder=OUT/'cases'/r['job'];im=cv2.imread(str(ROOT/c['image']))
        cv2.imwrite(str(folder/'confirmed_reference.jpg'),draw_lines(im,dict(vertical=gt['vertical_polylines'],horizontal=gt['horizontal_polylines'])))
        page=folder/'index.html';text=page.read_text(encoding='utf-8')
        text=text.replace('Все ошибки относительно разметки предварительные до подтверждения человеком.','Разметка проверена человеком. Автоматические детекторы оцениваются относительно неё; human_confirmed проверяет соблюдение заданных контрольных линий самим преобразованием.')
        text=text.replace('</h1>','</h1><p><a href="confirmed_reference.jpg">Подтверждённая разметка на оригинале</a></p>',1)
        page.write_text(text,encoding='utf-8')
        legacy=r['detector_matches']['legacy_observations']['horizontal']
        graph=r['detector_matches'].get('physical_graph_auto',{}).get('horizontal')
        graph_text=str(graph['detected']) if graph else 'отказ'
        rows.append(f'<tr><td><a href="cases/{r["job"]}/index.html">{r["job"][:10]}</a></td><td>{len(gt["horizontal_polylines"])}</td><td>{legacy["detected"]}</td><td>{legacy["missing"]}</td><td>{legacy["false"]}</td><td>{graph_text}</td><td>{graph["missing"] if graph else "—"}</td><td>{graph["false"] if graph else "—"}</td></tr>')
    for method in stage2.METHODS:
        entries=[e for r in results for e in r['methods'] if e['input']=='human_confirmed' and e['method']==method]
        good=[e for e in entries if 'reference_geometry' in e]
        values=[e['reference_geometry']['mean_px'] for e in good]
        summary.append(dict(method=method,calculated=len(good),mean_source_error_px=round(statistics.mean(values),3) if values else None,
            max_case_mean_px=max(values) if values else None,
            cases_with_folds=sum(e['safety']['folded_fraction']>0 for e in good),
            cases_with_invalid_jacobian=sum(e['safety']['jacobian_min']<=.02 for e in good),
            p95_line_error_max_px=max(e['reference_geometry'].get('p95_line_px',0) for e in good) if good else None))
    stage2.write(OUT/'summary.json',dict(workers=4,seconds=round(time.perf_counter()-t,2),human_reviewed_photos=len(validated),
        methods=summary,note='Human-confirmed input controls measure solver fidelity, not independent generalisation. Holdout still unlabelled.'))
    with (OUT/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['job','input','method','status','horizontal','reference_mean_px','missing_h','false_h','folded_fraction','valid_source_fraction','seconds'])
        for r in results:
            for e in r['methods']:
                match=e.get('against_reference',{}).get('horizontal',{})
                writer.writerow([r['job'],e['input'],e['method'],e['status'],e.get('horizontal'),e.get('reference_geometry',{}).get('mean_px'),match.get('missing'),match.get('false'),e.get('safety',{}).get('folded_fraction'),e.get('safety',{}).get('source_valid_fraction'),e.get('seconds')])
    method_rows=''.join(f'<tr><td>{s["method"]}</td><td>{s["calculated"]}</td><td>{s["mean_source_error_px"]}</td><td>{s["max_case_mean_px"]}</td><td>{s["cases_with_folds"]}</td></tr>' for s in summary)
    page='''<meta charset="utf-8"><title>V3: подтверждённая разметка</title><style>body{font:16px system-ui;margin:30px}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:8px}p{max-width:1100px}</style><h1>V3 — сравнение на подтверждённых линиях</h1><p>Проверены человеком все 10 фотографий. Детектор и преобразование проверены отдельно. Контрольные линии одинаковы для Perspective, Mesh и TPS. Ошибка обратной проекции измерена в пикселях исходной фотографии.</p><p><a href="FINDINGS.md">Заключение</a> · <a href="metrics.csv">CSV</a> · <a href="results.json">Все измерения</a> · <a href="integrity.json">Неизменность исходников</a> · <a href="../standalone_geometry_v3_stage2/holdout.html">Контрольный набор, пока без разметки</a></p><h2>Автоматическое обнаружение</h2><p>Индексы с нуля. «Пропущены» и «лишние» означают отсутствие пары в пределах 8 px: сильно смещённая настоящая линия также может оказаться несопоставленной. Отказ графа не означает отсутствие линий на фотографии.</p><table><tr><th>Фото</th><th>Подтверждено</th><th>Старые кандидаты</th><th>Пропущены</th><th>Лишние</th><th>Граф</th><th>Пропущены</th><th>Лишние</th></tr>'''+''.join(rows)+'''</table><h2>Преобразование подтверждённых линий</h2><p>Это соблюдение входных контролей, а не независимая точность нового детектора. Среднее — по десяти средним ошибок каждого снимка. Погрешность ручных узлов сохраняется.</p><table><tr><th>Метод</th><th>Построено</th><th>Среднее, px</th><th>Макс. среднего по фото, px</th><th>Фото с переворотом карты</th></tr>'''+method_rows+'''</table><p>Независимый контрольный набор не размечен. К внедрению пока не готово.</p>'''
    (OUT/'index.html').write_text(page,encoding='utf-8')
    after=snapshot(files);modified=[name for name in before if before[name]!=after.get(name)]
    stage2.write(OUT/'integrity.json',dict(protected_files=len(before),changed=modified,frozen_code_changed=changes))
    assert not modified,modified
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
