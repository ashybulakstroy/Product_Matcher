"""Frozen V3 final-set evaluation on user-confirmed intersection-bounded segments."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import concurrent.futures,hashlib,json,time
from pathlib import Path
import cv2
from geometry_lab.reference_segments import intersection_bounded_reference
from geometry_lab.linked_grid import validate_linked_annotation
from geometry_lab.structure_diagnostics import serialize,match_lines,draw_lines,fingerprint
from geometry_lab.physical_grid_graph_v3 import detect_physical_graph
from geometry_lab.physical_grid_graph_local import detect_physical_graph as v2
from geometry_lab.algorithms import METHODS
from geometry_lab.metrics import projected_lines,safety

LAB=Path(__file__).resolve().parent;ROOT=LAB.parent;STAGE=ROOT/'experiment_outputs/standalone_geometry_v3_stage4';OUT=ROOT/'experiment_outputs/standalone_geometry_v3_final'


def write(p,data):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')


def snapshot(names):
    return {n:dict(sha256=hashlib.sha256((ROOT/n).read_bytes()).hexdigest(),mtime_ns=(ROOT/n).stat().st_mtime_ns) for n in names if (ROOT/n).is_file()}


def run(c):
    cv2.setNumThreads(1);image=cv2.imread(str(ROOT/c['image']));folder=OUT/'cases'/c['job'];folder.mkdir(parents=True,exist_ok=True)
    result=dict(job=c['job'],sha256=c['sha256'],methods=[])
    # Reference is loaded only after both image-only detector calls.
    detected={}
    for version,detector in [('v2',v2),('v3',detect_physical_graph)]:
        t=time.perf_counter();data,report=detector(image);detected[version]=data
        result[version]=dict(status=report['status'],reason=report.get('reason'),seconds=time.perf_counter()-t)
        write(folder/(version+'_report.json'),report)
    gt=json.loads((OUT/'references'/f"{c['sha256']}.json").read_text(encoding='utf-8'))
    result.update(reference_horizontal=len(gt['horizontal_polylines']),reference_vertical=len(gt['vertical_polylines']),complete_reference_matrix=gt['complete_intersection_matrix'])
    cv2.imwrite(str(folder/'original.jpg'),image)
    cv2.imwrite(str(folder/'reference.jpg'),draw_lines(image,dict(horizontal=gt['horizontal_polylines'],vertical=gt['vertical_polylines'])))
    raw=json.loads((LAB/'final_control_saved'/f"{c['sha256']}.json").read_text(encoding='utf-8'))
    cv2.imwrite(str(folder/'reference_with_editor_tails.jpg'),draw_lines(image,dict(horizontal=raw['horizontal_polylines'],vertical=raw['vertical_polylines'])))
    for version,data in detected.items():
        result[version]['match']=match_lines(serialize(data),gt,image.shape) if data else None
        if data:
            write(folder/(version+'_controls.json'),serialize(data));cv2.imwrite(str(folder/(version+'.jpg')),draw_lines(image,serialize(data)))
    data=detected['v3']
    if data:
        digest=fingerprint(data)
        for method in ['grid_mesh','tps_scipy']:
            entry=dict(method=method,controls_sha256=digest)
            try:
                r=METHODS[method](image,data);v,h=projected_lines(r)
                projection=dict(vertical=[p.tolist() for p in v],horizontal=[p.tolist() for p in h])
                entry.update(status='CALCULATED',safety=safety(r,image.shape),match=match_lines(projection,gt,image.shape))
                overlay=r.image.copy()
                for x in r.xs:cv2.line(overlay,(round(x),round(r.ys[0])),(round(x),round(r.ys[-1])),(255,0,0),1)
                for y in r.ys:cv2.line(overlay,(round(r.xs[0]),round(y)),(round(r.xs[-1]),round(y)),(255,0,0),1)
                cv2.imwrite(str(folder/(method+'.jpg')),overlay)
            except Exception as error:entry.update(status='REJECTED',reason=repr(error))
            assert fingerprint(data)==digest;result['methods'].append(entry)
    write(folder/'result.json',result)
    blocks=[f'<h1>{c["job"]}</h1><p>Оригинальная разметка сохранена. Для оценки взяты только участки между существующими перпендикулярными связями, координаты узлов не изменены.</p>']
    for name in ['original','reference_with_editor_tails','reference','v2','v3','grid_mesh','tps_scipy']:
        if (folder/(name+'.jpg')).exists():blocks.append(f'<figure><figcaption>{name}</figcaption><a href="{name}.jpg"><img src="{name}.jpg"></a></figure>')
    blocks.append('<p><a href="result.json">Все метрики</a></p>')
    (folder/'index.html').write_text('<meta charset="utf-8"><style>body{font:16px system-ui}figure{width:30%;display:inline-block;vertical-align:top}img{max-width:100%}</style>'+''.join(blocks),encoding='utf-8')
    return result


def main():
    cases=json.loads((STAGE/'final_control_frozen.json').read_text())['cases'];frozen=json.loads((STAGE/'final_candidate_frozen.json').read_text())
    changed=[n for n,h in frozen.items() if hashlib.sha256((ROOT/n).read_bytes()).hexdigest()!=h]
    if changed:raise RuntimeError('Frozen candidate changed: '+repr(changed))
    names=list(json.loads((STAGE/'protected_before.json').read_text()));manifest=[]
    for c in cases:
        path=LAB/'final_control_saved'/f"{c['sha256']}.json";source=json.loads(path.read_text(encoding='utf-8'))
        if source.get('sha256')!=c['sha256'] or source.get('source_shape')!=c['shape']:raise ValueError('Reference/source mismatch')
        ref=intersection_bounded_reference(source);validate_linked_annotation(ref)
        write(OUT/'references'/path.name,ref);names.append(str(path.relative_to(ROOT)))
        manifest.append(dict(job=c['job'],source_annotation=str(path.relative_to(ROOT)),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            tails_removed=ref['tail_normalization']['removed_node_occurrences'],coordinates_moved=0,complete_intersection_matrix=ref['complete_intersection_matrix']))
    write(OUT/'reference_manifest.json',manifest);before=snapshot(names);write(OUT/'protected_before.json',before)
    started=time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:results=list(pool.map(run,cases))
    write(OUT/'results.json',results);after=snapshot(names)
    changed=[n for n,v in before.items() if after.get(n)!=v];write(OUT/'integrity.json',dict(protected_files=len(before),changed=changed,frozen_code_changed=[]));assert not changed
    write(OUT/'run.json',dict(workers=4,seconds=time.perf_counter()-started,reference_scope='user_confirmed_intersection_bounded_segments',editor_changed=False))
    for r in results:print(r['job'][:10],[(v,len(r[v]['match']['horizontal']['matches']) if r[v]['match'] else 0,r[v]['match']['horizontal']['false'] if r[v]['match'] else []) for v in ['v2','v3']],flush=True)


if __name__=='__main__':main()
