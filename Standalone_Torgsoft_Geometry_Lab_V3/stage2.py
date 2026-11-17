"""Reproducible detector/solver isolation, four receipt workers by default."""
from pathlib import Path
import argparse,concurrent.futures,hashlib,html,json,os,time
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import cv2
import numpy as np
from geometry_lab.structure_diagnostics import (audit_detector,serialize,fingerprint,
    reference_data,match_lines,draw_lines)
from geometry_lab.physical_grid_graph import detect_physical_graph
from geometry_lab.algorithms import METHODS
from geometry_lab.metrics import safety,projected_lines,evaluate_ground_truth

LAB=Path(__file__).resolve().parent;ROOT=LAB.parent
PREVIOUS=ROOT/'experiment_outputs/standalone_geometry_v3'
OUT=ROOT/'experiment_outputs/standalone_geometry_v3_stage2'


def write(path,data):
    path.parent.mkdir(exist_ok=True,parents=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')


def prepare():
    cases=json.loads((PREVIOUS/'selected.json').read_text())
    # Source-only visual traces of the remaining five development photographs.
    # Endpoints observed visually; intermediate column intersections approximate.
    # Never load detector outputs here; never replace the five previous references.
    nodes={
    cases[0]['job']:([149,204,484,573,700],[96,157,476,575,720],
        [[510,526],[565,581],[646,668],[707,725],[770,789],[835,852],[935,956],[990,1008]]),
    cases[2]['job']:([233,273,488,555,650],[142,184,463,546,665],
        [[233,256],[277,298],[335,360],[397,422],[460,484],[505,533],[552,580],[624,650],
         [722,747],[826,848],[906,930],[963,987],[1022,1046],[1082,1107],[1128,1152]]),
    cases[3]['job']:([196,228,364,405,463],[191,229,370,414,478],
        [[237,257],[271,287],[314,330],[391,402],[461,472],[545,555],[623,629],
         [713,708],[810,801],[860,846],[938,922],[1017,999],[1070,1051]]),
    cases[5]['job']:([350,374,500,539,594],[342,373,538,579,661],
        [[138,143],[165,170],[213,216],[262,266],[285,289],[307,312],[332,336],
         [385,386],[422,422],[449,447],[471,470],[497,496],[523,522],[551,549],
         [596,593],[625,621],[653,649],[710,704],[755,748],[801,795],[848,842],
         [907,903],[955,951],[1006,1001],[1040,1035],[1106,1099],[1139,1133],
         [1173,1167],[1228,1222],[1286,1281],[1343,1339],[1383,1378],[1423,1418],[1453,1447]]),
    cases[7]['job']:([42,109,420,515,659],[44,111,424,520,665],
        [[433,431],[505,502],[615,607],[671,662]])}
    manifest=[]
    for c in cases:
        previous=LAB/'ground_truth'/f"{c['sha256']}.json"
        if previous.exists():gt=json.loads(previous.read_text(encoding='utf-8'))
        else:
            top,bottom,ends=nodes[c['job']];hh=[]
            for k,(a,b) in enumerate(ends):
                t=k/(len(ends)-1);xs=np.array(top)*(1-t)+np.array(bottom)*t
                ys=a+(xs-xs[0])/(xs[-1]-xs[0])*(b-a)
                hh.append(np.column_stack((xs,ys)).round(2).tolist())
            gt=dict(sha256=c['sha256'],job_id=c['job'],source_shape=c['shape'],
                    vertical_polylines=[[row[i] for row in hh] for i in range(5)],
                    horizontal_polylines=hh,human_verified=False,
                    annotation_origin='agent_visual_source_only_stage2',estimated_uncertainty_px=6,
                    limitations='Unverified source tracing; internal nodes approximate, inspect/correct in review UI.')
        write(OUT/'references'/f"{c['sha256']}.json",gt)
        manifest.append(dict(job=c['job'],image=c['job']+'.jpg',reference=gt))
    template=(LAB/'review_template.html').read_text(encoding='utf-8')
    (OUT/'review.html').write_text(template.replace('__MANIFEST__',json.dumps(manifest,ensure_ascii=False)),encoding='utf-8')
    # Freeze before running the graph. These are not previously used by V3.
    # SHA uniqueness does not exclude a second shot of the same receipt.
    discovered=json.loads((PREVIOUS/'discovered.json').read_text())
    holdout=[c for c in discovered if c['index'] in [0,1,11,14,16,22,26,28]]
    path=OUT/'holdout_frozen.json'
    if not path.exists():write(path,dict(cases=holdout,selection='V3-unseen indices frozen before graph benchmark',
        limitations='Not human-labelled. Receipt identity/near-duplicate independence still needs human confirmation.',
        tuning_prohibited=True))
    return cases


def confirmed_reference(gt,c):
    """Do not turn an exported or automated draft into verified ground truth."""
    if not (gt.get('human_verified') is True and gt.get('source_checked') is True
            and gt.get('reviewer','').strip() and gt.get('reviewed_at')
            and gt.get('sha256')==c['sha256'] and gt.get('source_shape')==c['shape']):return False
    return all(len(gt.get('approvals',{}).get(a,[]))==len(gt.get(a+'_polylines',[]))
               and len(gt.get(a+'_polylines',[]))>=3
               and all(x is True for x in gt['approvals'][a]) for a in ['vertical','horizontal'])


def run_case(payload):
    c,confirmed_dir=payload;cv2.setNumThreads(1);start=time.perf_counter()
    im=cv2.imread(str(ROOT/c['image']));folder=OUT/'cases'/c['job'];folder.mkdir(parents=True,exist_ok=True)
    # Never display a stale success image after a later rejection.
    for name in ['after_graph.jpg','reference.jpg']:
        (folder/name).unlink(missing_ok=True)
    cv2.imwrite(str(folder/'original.jpg'),im)
    old,observations,trace=audit_detector(im)
    graph,greport=detect_physical_graph(im)
    write(folder/'detector_stages.json',trace);write(folder/'physical_graph.json',greport)
    cv2.imwrite(str(folder/'before.jpg'),draw_lines(im,serialize(observations)))
    if graph is not None:cv2.imwrite(str(folder/'after_graph.jpg'),draw_lines(im,serialize(graph)))
    gtpath=OUT/'references'/f"{c['sha256']}.json"
    gt=json.loads(gtpath.read_text(encoding='utf-8')) if gtpath.exists() else None
    inputs={'legacy_auto':old,'physical_graph_auto':graph}
    if gt:
        inputs['provisional_reference']=reference_data(gt,im.shape)
        cv2.imwrite(str(folder/'reference.jpg'),draw_lines(im,dict(vertical=gt['vertical_polylines'],horizontal=gt['horizontal_polylines'])))
    pending='No human-confirmed reference supplied'
    if confirmed_dir:
        path=Path(confirmed_dir)/f"{c['sha256']}.json"
        if path.exists():
            checked=json.loads(path.read_text(encoding='utf-8'))
            if confirmed_reference(checked,c):
                inputs['human_confirmed']=reference_data(checked,im.shape);gt=checked;pending=None
            else:pending='Invalid confirmation provenance or incomplete per-line approvals'
    result=dict(job=c['job'],sha256=c['sha256'],stages=trace,graph_status=greport['status'],
                reference_human_verified=gt.get('human_verified',False) if gt else False,
                human_confirmed_comparison=pending or 'AVAILABLE',methods=[],detector_matches={})
    if gt:
        for name,data in [('legacy_observations',observations),('physical_graph_auto',graph)]:
            if data:result['detector_matches'][name]=match_lines(serialize(data),gt,im.shape)
    for input_name,data in inputs.items():
        if data is None:
            for method in METHODS:result['methods'].append(dict(input=input_name,method=method,status='DETECTOR_REJECTED'))
            continue
        digest=fingerprint(data);write(folder/(input_name+'_controls.json'),serialize(data))
        for method,solver in METHODS.items():
            t=time.perf_counter();entry=dict(input=input_name,method=method,controls_sha256=digest)
            try:
                r=solver(im,data);entry.update(status='CALCULATED_NOT_ACCURACY',horizontal=len(r.ys),
                    safety=safety(r,im.shape),seconds=round(time.perf_counter()-t,3))
                v,h=projected_lines(r)
                projected=dict(vertical=[p.tolist() for p in v],horizontal=[p.tolist() for p in h])
                # Solver-only error: comparison to its EXACT input curves.
                ctrl=serialize(data);ctrl_gt=dict(vertical_polylines=ctrl['vertical'],horizontal_polylines=ctrl['horizontal'])
                entry['solver_control_residual']=match_lines(projected,ctrl_gt,im.shape,tolerance=1e4)
                if gt:entry['against_reference']=match_lines(projected,gt,im.shape)
                if gt and len(r.ys)==len(gt['horizontal_polylines']):entry['reference_geometry']=evaluate_ground_truth(r,gt,im.shape)
                prefix=input_name+'_'+method
                overlay=r.image.copy()
                for x in r.xs:cv2.line(overlay,(round(x),round(r.ys[0])),(round(x),round(r.ys[-1])),(255,0,0),1)
                for y in r.ys:cv2.line(overlay,(round(r.xs[0]),round(y)),(round(r.xs[-1]),round(y)),(255,0,0),1)
                cv2.imwrite(str(folder/(prefix+'.jpg')),overlay)
                cv2.imwrite(str(folder/(prefix+'_source.jpg')),draw_lines(im,projected))
                entry['image']=prefix+'.jpg';entry['source_projection']=prefix+'_source.jpg'
            except Exception as e:entry.update(status='REJECTED',reason=f'{type(e).__name__}: {e}')
            if fingerprint(data)!=digest:raise RuntimeError('solver mutated shared controls')
            result['methods'].append(entry)
    result['seconds']=round(time.perf_counter()-start,2);write(folder/'result.json',result)
    chunks=[f"<h1>{c['job']}</h1><p>Все ошибки относительно разметки предварительные до подтверждения человеком. Синие линии — проекция геометрии, не эталон.</p>"]
    for file,label in [('original.jpg','Оригинал'),('before.jpg','До: старый детектор'),('after_graph.jpg','Physical Grid Graph'),('reference.jpg','Предварительные линии для проверки')]:
        if (folder/file).exists():chunks.append(f'<figure><figcaption>{label}</figcaption><a href="{file}"><img src="{file}"></a></figure>')
    for e in result['methods']:
        if 'image' in e:chunks.append(f'<figure><figcaption>{e["input"]} / {e["method"]}</figcaption><a href="{e["image"]}"><img src="{e["image"]}"></a><a href="{e["source_projection"]}">Проекция на оригинал</a></figure>')
    chunks.append('<p><a href="result.json">Полные измерения JSON</a> · <a href="detector_stages.json">Стадии детектора</a> · <a href="physical_graph.json">Доказательства графа</a></p>')
    (folder/'index.html').write_text('<meta charset="utf-8"><style>body{font:16px system-ui}figure{display:inline-block;vertical-align:top;width:30%}img{max-width:100%}</style>'+''.join(chunks),encoding='utf-8')
    return result


def report(results):
    rows=[]
    for r in results:
        old=r['detector_matches'].get('legacy_observations',{}).get('horizontal',{})
        graph=r['detector_matches'].get('physical_graph_auto',{}).get('horizontal',{})
        rows.append(f'<tr><td><a href="cases/{r["job"]}/index.html">{r["job"][:10]}</a></td><td>{old.get("reference","не размечено")}</td><td>{r["stages"]["selected_horizontal_count"]}</td><td>{old.get("missing","—")}</td><td>{old.get("false","—")}</td><td>{graph.get("detected",r["graph_status"])}</td><td>{graph.get("missing","—")}</td><td>{graph.get("false","—")}</td></tr>')
    text='''<meta charset="utf-8"><title>V3: диагностика структуры</title><style>body{font:16px system-ui;margin:30px}td,th{padding:9px;border:1px solid #bbb}table{border-collapse:collapse}p{max-width:1100px}</style><h1>V3 — детектор отдельно от выпрямления</h1><p>Ни одна предварительная линия не считается подтверждённой человеком. Индексы пропущенных и ложных линий начинаются с 0; видны на по-чековых изображениях. Порог сопоставления 8 пикселей, это настройка диагностики, не доказанная точность.</p><p><a href="review.html">Ручная проверка и исправление разметки</a> · <a href="results.json">Все измерения</a> · <a href="FINDINGS.md">Заключение</a> · <a href="holdout_frozen.json">Зафиксированный контрольный набор</a></p><table><tr><th>Фото</th><th>Предв. полных границ</th><th>Старый детектор</th><th>Пропущены</th><th>Ложные</th><th>Граф</th><th>Пропущены</th><th>Ложные</th></tr>'''
    (OUT/'index.html').write_text(text+''.join(rows)+'</table><p>Короткие линии блока итога не считаются полными границами товарной сетки. Геометрическое определение НДС без текста остаётся UNKNOWN. Граф пока использует общие вертикальные предложения V3.</p>',encoding='utf-8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4,choices=range(4,9));p.add_argument('--confirmed-dir');p.add_argument('--prepare-only',action='store_true');p.add_argument('--holdout',action='store_true');a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);cases=prepare()
    if a.prepare_only:return
    if a.holdout:cases=json.loads((OUT/'holdout_frozen.json').read_text())['cases']
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
        results=list(pool.map(run_case,[(c,a.confirmed_dir) for c in cases]))
    name='holdout_results.json' if a.holdout else 'results.json';write(OUT/name,results)
    if not a.holdout:report(results)
    else:
        links=''.join(f'<li><a href="cases/{r["job"]}/index.html">{r["job"]}</a> — {r["graph_status"]}; разметка отсутствует</li>' for r in results)
        (OUT/'holdout.html').write_text('<meta charset="utf-8"><h1>Контрольный набор V3</h1><p>Не использован для настройки. Нужна проверка независимости фотографий и ручная разметка. Отказ/построение карты не является точностью.</p><ul>'+links+'</ul>',encoding='utf-8')
    print(json.dumps([dict(job=r['job'][:10],old=r['stages']['selected_horizontal_count'],graph=r['graph_status'],seconds=r['seconds']) for r in results],indent=2))


if __name__=='__main__':main()
