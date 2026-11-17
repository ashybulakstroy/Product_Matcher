"""Diagnostic-only detector development. Independent controls are never GT here."""
from pathlib import Path
import concurrent.futures,csv,hashlib,html,json,os,time
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import cv2,numpy as np
from geometry_lab.physical_grid_graph_local import detect_physical_graph,ridge_response
from geometry_lab.structure_diagnostics import reference_data,serialize,match_lines,draw_lines,fingerprint
from geometry_lab.algorithms import METHODS
from geometry_lab.metrics import safety,projected_lines,evaluate_ground_truth

LAB=Path(__file__).resolve().parent;ROOT=LAB.parent
OUT=ROOT/'experiment_outputs/standalone_geometry_v3_stage3'


def write(p,data):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')


def audit_miss(image,gt,line_id,old_report,folder):
    H,W=image.shape[:2];gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY);response,dark=ridge_response(gray)
    rule=reference_data(gt,image.shape)['horizontal'][line_id]
    xs=np.linspace(*rule.domain,200);ys=rule.eval((xs-W/2)/W)
    xx=np.rint(xs).astype(int).clip(0,W-1);yy=np.rint(ys).astype(int).clip(0,H-1)
    vals=response[yy,xx];observed=vals>=7
    chord=np.interp(xs,[xs[0],xs[-1]],[ys[0],ys[-1]])
    curvature=float(np.max(abs(ys-chord)))
    candidates=old_report.get('edges',[])+old_report.get('footer',[])+old_report.get('rejected',[])
    near=sorted(candidates,key=lambda e:abs(e['position']-rule.position))[:3]
    reasons=[]
    if float(observed.mean())<.7:reasons.append('weak_or_interrupted_print_evidence')
    if curvature>2:reasons.append('curve_not_represented_by_straight_fragments')
    if near and abs(near[0]['position']-rule.position)<8:
        e=near[0]
        if sum(c>=.5 for c in e['column_coverage'])<3:reasons.append('v1_column_coverage_gate')
        if sum(e['intersection_evidence'])<3:reasons.append('v1_intersection_or_observed_domain_gate')
        if abs(e['domain'][0]-rule.domain[0])>12 or abs(e['domain'][1]-rule.domain[1])>12:reasons.append('v1_full_width_fragment_endpoint_gate')
    else:reasons.append('no_v1_proposal_near_physical_rule_fragment_loss_or_grouping')
    if not reasons:reasons=['v1_selection_or_dedup_gate_needs_visual_review']
    y0=max(0,int(ys.min())-22);y1=min(H,int(ys.max())+23)
    x0=max(0,int(xs.min())-15);x1=min(W,int(xs.max())+16)
    crop=image[y0:y1,x0:x1].copy();path=folder/f'missing_{line_id:02d}.jpg';cv2.imwrite(str(path),crop)
    annotated=crop.copy();p=np.column_stack((xs-x0,ys-y0)).round().astype(np.int32)
    cv2.polylines(annotated,[p],False,(255,170,0),1,cv2.LINE_AA);cv2.imwrite(str(folder/f'missing_{line_id:02d}_marked.jpg'),annotated)
    return dict(reference_line=line_id,source_ridge_mean=float(vals.mean()),observed_sample_fraction=float(observed.mean()),
        chord_deviation_px=curvature,algorithmic_reasons=reasons,nearest_v1_candidates=near,
        physical_cause_note='Weak print/break/curve evidenced by measurements; fold vs text overlap requires source visual review, not inferred from low contrast alone.',
        original_crop=path.name,marked_crop=f'missing_{line_id:02d}_marked.jpg')


def run_case(c):
    cv2.setNumThreads(1);started=time.perf_counter()
    folder=OUT/'cases'/c['job'];folder.mkdir(parents=True,exist_ok=True)
    image=cv2.imread(str(ROOT/c['image']));gt=json.loads((LAB/'confirmed'/f"{c['sha256']}.json").read_text(encoding='utf-8'))
    old=json.loads((ROOT/'experiment_outputs/standalone_geometry_v3_confirmed/cases'/c['job']/'result.json').read_text())
    old_graph=json.loads((ROOT/'experiment_outputs/standalone_geometry_v3_confirmed/cases'/c['job']/'physical_graph.json').read_text())
    data,report=detect_physical_graph(image);write(folder/'physical_graph_v2.json',report)
    old_match=old['detector_matches'].get('physical_graph_auto')
    old_missing=old_match['horizontal']['missing'] if old_match else list(range(len(gt['horizontal_polylines'])))
    audit=[audit_miss(image,gt,i,old_graph,folder) for i in old_missing]
    result=dict(job=c['job'],sha256=c['sha256'],before=old_match,after=match_lines(serialize(data),gt,image.shape) if data else None,
        detector_status=report['status'],miss_audit=audit,methods=[])
    cv2.imwrite(str(folder/'original.jpg'),image)
    ref=dict(vertical=gt['vertical_polylines'],horizontal=gt['horizontal_polylines'])
    cv2.imwrite(str(folder/'confirmed.jpg'),draw_lines(image,ref))
    if data:
        cv2.imwrite(str(folder/'detected.jpg'),draw_lines(image,serialize(data)))
        write(folder/'controls.json',serialize(data));digest=fingerprint(data)
        for method in ['grid_mesh','tps_scipy']:
            entry=dict(method=method,controls_sha256=digest)
            try:
                t=time.perf_counter();r=METHODS[method](image,data);v,h=projected_lines(r)
                projected=dict(vertical=[p.tolist() for p in v],horizontal=[p.tolist() for p in h])
                entry.update(status='CALCULATED_NOT_VALIDATED',seconds=time.perf_counter()-t,safety=safety(r,image.shape),
                    against_confirmed=match_lines(projected,gt,image.shape))
                if len(r.ys)==len(gt['horizontal_polylines']):entry['geometry']=evaluate_ground_truth(r,gt,image.shape)
                overlay=r.image.copy()
                for x in r.xs:cv2.line(overlay,(round(x),round(r.ys[0])),(round(x),round(r.ys[-1])),(255,0,0),1)
                for y in r.ys:cv2.line(overlay,(round(r.xs[0]),round(y)),(round(r.xs[-1]),round(y)),(255,0,0),1)
                cv2.imwrite(str(folder/(method+'.jpg')),overlay)
                cv2.imwrite(str(folder/(method+'_source.jpg')),draw_lines(image,projected))
            except Exception as e:entry.update(status='REJECTED',reason=repr(e))
            assert fingerprint(data)==digest
            result['methods'].append(entry)
    result['seconds']=time.perf_counter()-started;write(folder/'result.json',result)
    chunks=[f'<h1>{c["job"]}</h1><p>Подтверждённый диагностический набор. Не независимый тест.</p>']
    for name in ['original','confirmed','detected','grid_mesh','tps_scipy']:
        if (folder/(name+'.jpg')).exists():chunks.append(f'<figure><figcaption>{name}</figcaption><a href="{name}.jpg"><img src="{name}.jpg"></a></figure>')
    chunks.append('<h2>Каждая ранее пропущенная линия</h2>')
    for a in audit:
        chunks.append(f'<p>Линия {a["reference_line"]}: {html.escape(", ".join(a["algorithmic_reasons"]))}</p><a href="{a["original_crop"]}"><img style="max-width:100%" src="{a["original_crop"]}"></a><br><img style="max-width:100%" src="{a["marked_crop"]}">')
    chunks.append('<p><a href="result.json">Метрики и причины</a> · <a href="physical_graph_v2.json">Доказательства каждой линии</a></p>')
    (folder/'index.html').write_text('<meta charset="utf-8"><style>body{font:16px system-ui}figure{display:inline-block;width:30%;vertical-align:top}figure img{width:100%}</style>'+''.join(chunks),encoding='utf-8')
    return result


def main():
    cases=json.loads((ROOT/'experiment_outputs/standalone_geometry_v3/selected.json').read_text())
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:results=list(pool.map(run_case,cases))
    write(OUT/'results.json',results)
    for r in results:
        before=r['before']['horizontal'] if r['before'] else None;after=r['after']['horizontal'] if r['after'] else None
        print(r['job'][:10], 'before',before['detected'] if before else 'reject','after',after['detected'] if after else 'reject',
            'missing',after['missing'] if after else None,'false',after['false'] if after else None,flush=True)


if __name__=='__main__':main()
