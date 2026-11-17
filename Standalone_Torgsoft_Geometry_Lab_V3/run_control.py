"""One frozen independent evaluation, using human-confirmed linked source grids."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import concurrent.futures,csv,hashlib,json,statistics,time
import cv2,numpy as np
from stage2 import confirmed_reference
from geometry_lab.linked_grid import validate_linked_annotation
from geometry_lab.physical_grid_graph_local import detect_physical_graph
from geometry_lab.physical_grid_graph import detect_physical_graph as detect_v1
from geometry_lab.structure_diagnostics import serialize,fingerprint,match_lines,draw_lines,reference_data
from geometry_lab.algorithms import METHODS
from geometry_lab.metrics import safety,projected_lines,evaluate_ground_truth

LAB=Path(__file__).resolve().parent;ROOT=LAB.parent
STAGE=ROOT/'experiment_outputs/standalone_geometry_v3_stage3'
OUT=ROOT/'experiment_outputs/standalone_geometry_v3_independent'


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')


def snap(names):
    return {n:dict(sha256=hashlib.sha256((ROOT/n).read_bytes()).hexdigest(),mtime_ns=(ROOT/n).stat().st_mtime_ns)
            for n in names if (ROOT/n).is_file()}


def run_case(c):
    cv2.setNumThreads(1);started=time.perf_counter();image=cv2.imread(str(ROOT/c['image']))
    # Detector receives image only. No reference coordinates enter either call.
    data,report=detect_physical_graph(image);old,old_report=detect_v1(image)
    gt=json.loads((LAB/'control_confirmed'/f"{c['sha256']}.json").read_text(encoding='utf-8'))
    folder=OUT/'cases'/c['job'];folder.mkdir(parents=True,exist_ok=True)
    write(folder/'detector_v2.json',report);write(folder/'detector_v1.json',old_report)
    cv2.imwrite(str(folder/'original.jpg'),image)
    cv2.imwrite(str(folder/'confirmed.jpg'),draw_lines(image,dict(vertical=gt['vertical_polylines'],horizontal=gt['horizontal_polylines'])))
    result=dict(job=c['job'],sha256=c['sha256'],reference_horizontal=len(gt['horizontal_polylines']),
                detector_status=report['status'],v1=match_lines(serialize(old),gt,image.shape) if old else None,
                v2=match_lines(serialize(data),gt,image.shape) if data else None,methods=[])
    if data:
        cv2.imwrite(str(folder/'detected.jpg'),draw_lines(image,serialize(data)))
        write(folder/'auto_controls.json',serialize(data))
    for scenario,controls in [('v1_auto',old),('v2_auto',data),('human_controls',reference_data(gt,image.shape))]:
        if controls is None:continue
        digest=fingerprint(controls)
        for method in ['grid_mesh','tps_scipy']:
            t=time.perf_counter();entry=dict(input=scenario,method=method,controls_sha256=digest)
            try:
                r=METHODS[method](image,controls);v,h=projected_lines(r)
                projected=dict(vertical=[p.tolist() for p in v],horizontal=[p.tolist() for p in h])
                entry.update(status='CALCULATED',horizontal=len(r.ys),seconds=time.perf_counter()-t,safety=safety(r,image.shape),
                    against_confirmed=match_lines(projected,gt,image.shape))
                if len(r.ys)==len(gt['horizontal_polylines']):entry['geometry']=evaluate_ground_truth(r,gt,image.shape)
                prefix=scenario+'_'+method;overlay=r.image.copy()
                for x in r.xs:cv2.line(overlay,(round(x),round(r.ys[0])),(round(x),round(r.ys[-1])),(255,0,0),1)
                for y in r.ys:cv2.line(overlay,(round(r.xs[0]),round(y)),(round(r.xs[-1]),round(y)),(255,0,0),1)
                cv2.imwrite(str(folder/(prefix+'.jpg')),overlay)
                cv2.imwrite(str(folder/(prefix+'_source.jpg')),draw_lines(image,projected))
                entry['image']=prefix+'.jpg';entry['source_projection']=prefix+'_source.jpg'
            except Exception as e:entry.update(status='REJECTED',reason=repr(e))
            assert fingerprint(controls)==digest
            result['methods'].append(entry)
    result['seconds']=time.perf_counter()-started;write(folder/'result.json',result)
    blocks=[f'<h1>{c["job"]}</h1><p>Независимый контроль. Разметка подтверждена человеком до запуска.</p>']
    for name in ['original','confirmed','detected']:
        if (folder/(name+'.jpg')).exists():blocks.append(f'<figure><figcaption>{name}</figcaption><a href="{name}.jpg"><img src="{name}.jpg"></a></figure>')
    for e in result['methods']:
        if 'image'in e:blocks.append(f'<figure><figcaption>{e["input"]} / {e["method"]}</figcaption><a href="{e["image"]}"><img src="{e["image"]}"></a><a href="{e["source_projection"]}">Обратно на оригинале</a></figure>')
    blocks.append('<p><a href="result.json">Полные измерения</a></p>')
    (folder/'index.html').write_text('<meta charset="utf-8"><style>body{font:16px system-ui}figure{width:30%;display:inline-block;vertical-align:top}img{width:100%}</style>'+''.join(blocks),encoding='utf-8')
    return result


def aggregate(results,key,axis='horizontal'):
    tp=fp=fn=0;errors=[]
    for r in results:
        ref=r['reference_horizontal'] if axis=='horizontal' else 5;m=r[key][axis] if r[key] else None
        tp+=len(m['matches']) if m else 0;fp+=len(m['false']) if m else 0;fn+=len(m['missing']) if m else ref
        if m:errors.extend(p['mean_px'] for p in m['matches'])
    return dict(reference=tp+fn,matched=tp,missing=fn,false=fp,recall=tp/(tp+fn),precision=tp/(tp+fp) if tp+fp else None,
        mean_matched_line_error_px=statistics.mean(errors) if errors else None,
        p95_of_line_mean_errors_px=float(np.percentile(errors,95)) if errors else None)


def main():
    cases=json.loads((STAGE/'control_frozen.json').read_text())['cases']
    frozen=json.loads((STAGE/'control_code_frozen.json').read_text());changed=[name for name,digest in frozen.items() if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest]
    if changed:raise RuntimeError('Frozen candidate changed: '+repr(changed))
    manifest=[]
    diagnostic=json.loads((ROOT/'experiment_outputs/standalone_geometry_v3/selected.json').read_text())
    for c in cases:
        path=LAB/'control_confirmed'/f"{c['sha256']}.json";gt=json.loads(path.read_text(encoding='utf-8'))
        if not confirmed_reference(gt,c) or not gt.get('identity_checked'):raise ValueError('Missing human source/identity verification: '+path.name)
        validate_linked_annotation(gt)
        if c['sha256'] in [d['sha256'] for d in diagnostic]:raise ValueError('Control overlaps diagnostic set')
        if list(cv2.imread(str(ROOT/c['image'])).shape[:2])!=gt['source_shape']:raise ValueError('Image shape mismatch')
        manifest.append(dict(job=c['job'],annotation=str(path.relative_to(ROOT)),annotation_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_sha256=hashlib.sha256((ROOT/c['image']).read_bytes()).hexdigest(),identity_checked=True,reviewed_at=gt['reviewed_at']))
    OUT.mkdir(parents=True,exist_ok=True);write(OUT/'input_manifest.json',manifest);write(OUT/'candidate_frozen.json',frozen)
    names=list(json.loads((STAGE/'protected_before.json').read_text()))+[m['annotation'] for m in manifest]
    before=snap(names);write(OUT/'protected_before.json',before);started=time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:results=list(pool.map(run_case,cases))
    write(OUT/'results.json',results)
    summary=dict(photos=len(cases),workers=4,seconds=time.perf_counter()-started,
        v1_horizontal=aggregate(results,'v1'),v2_horizontal=aggregate(results,'v2'),v2_vertical=aggregate(results,'v2','vertical'),
        candidate_unchanged=True,matching_tolerance_px=8,
        full_horizontal_cases=sum(r['v2'] is not None and not r['v2']['horizontal']['missing'] and not r['v2']['horizontal']['false'] for r in results))
    write(OUT/'summary.json',summary)
    with (OUT/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['job','detector','reference_h','matched_h','missing_h','false_h','mean_matched_h_px'])
        for r in results:
            for key in ['v1','v2']:
                m=r[key]['horizontal'] if r[key] else None
                w.writerow([r['job'],key,r['reference_horizontal'],len(m['matches']) if m else 0,m['missing'] if m else 'ALL_REJECTED',m['false'] if m else None,m['mean_matched_px'] if m else None])
    after=snap(names);modified=[name for name in before if before[name]!=after.get(name)]
    code_changed=[name for name,digest in frozen.items() if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest]
    write(OUT/'integrity.json',dict(protected_files=len(before),changed=modified,candidate_changed=code_changed))
    assert not modified and not code_changed
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
