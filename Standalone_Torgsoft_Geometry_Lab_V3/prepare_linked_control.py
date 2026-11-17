"""Freeze the candidate, create unverified source drafts on a separate set.

No labels from this set are read, no thresholds are changed, and no accuracy is
computed here. Humans must add missed rules, remove false rules and verify ID.
"""
from pathlib import Path
import concurrent.futures,hashlib,json,os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import cv2,numpy as np
from stage3 import ROOT,LAB,OUT,write
from geometry_lab.physical_grid_graph_local import detect_physical_graph
from geometry_lab.algorithms import intersection
from geometry_lab.detection import _group_and_fit,_choose_verticals
from editor_documents import render_editor


def draft(c):
    cv2.setNumThreads(1);image=cv2.imread(str(ROOT/c['image']));H,W=image.shape[:2]
    data,report=detect_physical_graph(image)
    nodes=[]
    if data:
        nodes=[[list(intersection(v,h,image.shape)) for v in data['vertical']] for h in data['horizontal']]
        column_x=[float(v.position) for v in data['vertical']]
    else:
        vv=_choose_verticals(_group_and_fit(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),'v'),W,H)
        column_x=[v.position for v in vv] if vv else [W*f for f in [.15,.22,.6,.72,.9]]
    return dict(job=c['job'],sha256=c['sha256'],image='../../'+c['image'].replace('\\','/'),height=H,width=W,
                nodes=nodes,column_x=column_x,seed_origin='automatic_proposals_unverified_not_ground_truth',
                no_detected_rows=data is None)


def main():
    control=json.loads((OUT/'control_frozen.json').read_text())
    files=[LAB/'geometry_lab'/name for name in ['algorithms.py','physical_grid_graph_local.py','physical_grid_graph.py','detection.py','structure_diagnostics.py']]
    freeze={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    path=OUT/'control_code_frozen.json'
    if path.exists() and json.loads(path.read_text())!=freeze:raise RuntimeError('Candidate changed after control draft; do not reuse as independent test silently')
    write(path,freeze)
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:manifest=list(pool.map(draft,control['cases']))
    write(OUT/'control_drafts_unverified.json',manifest)
    (OUT/'control_review.html').write_text(render_editor(manifest),encoding='utf-8')
    (LAB/'control_confirmed').mkdir(exist_ok=True)
    print('Linked control editor prepared for',len(manifest),'new photographs. No accuracy claim.')


if __name__=='__main__':main()
