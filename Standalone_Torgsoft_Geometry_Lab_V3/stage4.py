"""V2 versus V3 on 18 development/regression photos. Not an independent test."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import concurrent.futures,json,time
from pathlib import Path
import cv2
from geometry_lab.physical_grid_graph_v3 import detect_physical_graph
from geometry_lab.physical_grid_graph_local import detect_physical_graph as v2
from geometry_lab.structure_diagnostics import serialize,match_lines,draw_lines
from geometry_lab.structure_diagnostics import fingerprint
from geometry_lab.algorithms import METHODS
from geometry_lab.metrics import projected_lines,safety
import numpy as np
ROOT=Path(__file__).resolve().parent.parent;LAB=ROOT/'Standalone_Torgsoft_Geometry_Lab_V3';OUT=ROOT/'experiment_outputs/standalone_geometry_v3_stage4'

def write(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')

def cases():
    first=json.loads((ROOT/'experiment_outputs/standalone_geometry_v3/selected.json').read_text())
    second=json.loads((ROOT/'experiment_outputs/standalone_geometry_v3_stage3/control_frozen.json').read_text())['cases']
    return [dict(c,labels='confirmed',group='diagnostic10') for c in first]+[dict(c,labels='control_confirmed',group='former_control8_now_regression') for c in second]

def run(c):
    cv2.setNumThreads(1);im=cv2.imread(str(ROOT/c['image']));folder=OUT/'cases'/c['job'];folder.mkdir(parents=True,exist_ok=True)
    gt=json.loads((LAB/c['labels']/f"{c['sha256']}.json").read_text(encoding='utf-8'))
    result=dict(job=c['job'],sha256=c['sha256'],group=c['group'],reference=len(gt['horizontal_polylines']))
    cv2.imwrite(str(folder/'original.jpg'),im);cv2.imwrite(str(folder/'reference.jpg'),draw_lines(im,dict(horizontal=gt['horizontal_polylines'],vertical=gt['vertical_polylines'])))
    for name,detector in [('v2',v2),('v3',detect_physical_graph)]:
        start=time.perf_counter();data,report=detector(im);result[name]=dict(seconds=time.perf_counter()-start,status=report['status'],reason=report.get('reason'),selection=report.get('selection'),match=match_lines(serialize(data),gt,im.shape) if data else None)
        write(folder/(name+'_report.json'),report)
        if data:
            controls=serialize(data);write(folder/(name+'_controls.json'),controls);cv2.imwrite(str(folder/(name+'.jpg')),draw_lines(im,controls))
            for axis,along,across in [('horizontal',0,1),('vertical',1,0)]:
                samples=[]
                for pair in result[name]['match'][axis]['matches']:
                    p=np.array(controls[axis][pair['detected']]);q=np.array(gt[axis+'_polylines'][pair['reference']]);q=q[np.argsort(q[:,along])]
                    lo=max(p[:,along].min(),q[:,along].min());hi=min(p[:,along].max(),q[:,along].max());t=np.linspace(lo,hi,150)
                    error=abs(np.interp(t,p[:,along],p[:,across])-np.interp(t,q[:,along],q[:,across]));samples.extend(error.tolist())
                result[name]['match'][axis]['max_sample_error_px']=max(samples) if samples else None
            if name=='v3':
                result['methods']=[];digest=fingerprint(data)
                for method in ['grid_mesh','tps_scipy']:
                    entry=dict(method=method,controls_sha256=digest)
                    try:
                        r=METHODS[method](im,data);v,h=projected_lines(r)
                        projection=dict(vertical=[p.tolist() for p in v],horizontal=[p.tolist() for p in h])
                        entry.update(status='CALCULATED',safety=safety(r,im.shape),match=match_lines(projection,gt,im.shape))
                        overlay=r.image.copy()
                        for x in r.xs:cv2.line(overlay,(round(x),round(r.ys[0])),(round(x),round(r.ys[-1])),(255,0,0),1)
                        for y in r.ys:cv2.line(overlay,(round(r.xs[0]),round(y)),(round(r.xs[-1]),round(y)),(255,0,0),1)
                        cv2.imwrite(str(folder/(method+'.jpg')),overlay);cv2.imwrite(str(folder/(method+'_source.jpg')),draw_lines(im,projection))
                    except Exception as error:entry.update(status='REJECTED',reason=repr(error))
                    assert fingerprint(data)==digest
                    result['methods'].append(entry)
    write(folder/'result.json',result)
    return result

def main():
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:results=list(pool.map(run,cases()))
    write(OUT/'results.json',results)
    for r in results:print(r['job'][:8],[(k,len(r[k]['match']['horizontal']['matches']) if r[k]['match'] else 0,r[k]['match']['horizontal']['false'] if r[k]['match'] else []) for k in ['v2','v3']],flush=True)

if __name__=='__main__':main()
