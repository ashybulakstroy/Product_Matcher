"""V3: multiscale physical segments and jointly scored five-column hypotheses.

Image-only experimental detector. Frozen V2 remains a conservative fallback.
No OCR, reference coordinates, receipt ID, or expected row count is accepted.
"""
import cv2,numpy as np
from itertools import combinations
from .detection import Rule
from .physical_grid_graph import connected_fragments,graph_verticals
from .physical_grid_graph_local import (detect_physical_graph as v2,ridge_response,
                                       trace_seed,junction_outer_boundaries)
from .structure_diagnostics import PolylineRule
from .algorithms import intersection


def multiscale_segments(gray):
    H,W=gray.shape;parts={'h':[],'v':[]}
    for scale in [1.,1.5]:
        sized=cv2.resize(gray,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
        sized=cv2.createCLAHE(clipLimit=2.,tileGridSize=(8,8)).apply(sized)
        found=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(sized)[0]
        if found is None:continue
        for item in found.reshape(-1,4):
            x0,y0,x1,y1=(item+.5)/scale-.5;dx=x1-x0;dy=y1-y0
            if abs(dx)>=max(18,W*.025) and abs(dy)<=abs(dx)*.35:
                if x1<x0:x0,y0,x1,y1=x1,y1,x0,y0
                ref=y0+(W/2-x0)*(y1-y0)/max(x1-x0,1)
                parts['h'].append((float(ref),tuple(map(float,(x0,y0,x1,y1))),float(x1-x0)))
            elif abs(dy)>=max(30,H*.025) and abs(dx)<=abs(dy)*.35:
                # Transposed coordinates use the same fragment-connectivity test.
                if y1<y0:x0,y0,x1,y1=x1,y1,x0,y0
                ref=x0+(H/2-y0)*(x1-x0)/max(y1-y0,1)
                parts['v'].append((float(ref),tuple(map(float,(y0,x0,y1,x1))),float(y1-y0)))
    return parts


def fit_segments(parts,axis,shape):
    H,W=shape;extent=W if axis=='h' else H
    rules=[]
    for group in connected_fragments(parts,extent,tolerance=3.):
        p=[]
        for _,(a,b,c,d),length in group:
            t=np.linspace(0,1,max(3,int(length/8)));p.extend(zip(a+(c-a)*t,b+(d-b)*t))
        p=np.array(p);span=np.ptp(p[:,0])
        if span<max(35,extent*(.12 if axis=='h' else .035)):continue
        norm=(p[:,0]-extent/2)/extent;A=np.vander(norm,3);weight=np.ones(len(p))
        for _ in range(4):
            coeff=np.linalg.lstsq(A*weight[:,None],p[:,1]*weight,rcond=None)[0]
            weight=np.minimum(1.,2./np.maximum(abs(p[:,1]-A@coeff),.001))
        rules.append(Rule(axis,coeff,float(np.polyval(coeff,0)),float(span/extent),
                          (float(p[:,0].min()),float(p[:,0].max()))))
    return sorted(rules,key=lambda r:r.position)


def evidence(rule,vv,shape,response,dark):
    H,W=shape;nodes=[intersection(v,rule,shape) for v in vv]
    left,right=nodes[0][0],nodes[-1][0]
    if right-left<50:return None
    xs=np.linspace(left,right,max(70,int(right-left)));ys=rule.eval((xs-W/2)/W)
    valid=(xs>=0)&(xs<W)&(ys>=1)&(ys<H-1)
    ix=np.rint(xs).astype(int).clip(0,W-1);iy=np.rint(ys).astype(int).clip(0,H-1)
    pixel=np.maximum.reduce([dark[(iy+d).clip(0,H-1),ix] for d in [-1,0,1]])
    observed=valid&(response[iy,ix]>=7)&(pixel>=9)
    columns=[];junctions=[]
    for i in range(4):
        mask=(xs>=nodes[i][0]+3)&(xs<=nodes[i+1][0]-3)
        columns.append(float(observed[mask].mean()) if mask.any() else 0.)
    for i,(x,y) in enumerate(nodes):
        offsets=np.arange(3,11) if i==0 else -np.arange(3,11) if i==4 else np.r_[-np.arange(3,11),np.arange(3,11)]
        xx=x+offsets;yy=rule.eval((xx-W/2)/W);okay=(xx>=0)&(xx<W)&(yy>=1)&(yy<H-1)
        xx=np.rint(xx[okay]).astype(int).clip(0,W-1);yy=np.rint(yy[okay]).astype(int).clip(0,H-1)
        arms=np.maximum.reduce([dark[(yy+d).clip(0,H-1),xx] for d in [-1,0,1]]) if len(xx) else np.array([])
        junctions.append(bool(len(arms)>=5 and (arms>=9).mean()>=.65))
    cuts=np.r_[0,np.where(observed[1:]!=observed[:-1])[0]+1,len(xs)]
    intervals=[dict(x0=float(xs[a]),x1=float(xs[b-1]),observed=bool(observed[a])) for a,b in zip(cuts,cuts[1:])]
    gap=max([r['x1']-r['x0'] for r in intervals if not r['observed']]+[0])/(right-left)
    accepted=(min(columns)>=.55 and observed.mean()>=.64 and sum(junctions)>=3 and gap<=.18)
    return dict(position=rule.position,domain=list(rule.domain),column_coverage=columns,
        ink_coverage=float(observed.mean()),intersection_evidence=junctions,intervals=intervals,
        longest_unsupported_fraction=float(gap),accepted=bool(accepted),
        reason='physical_source_arms_and_four_columns' if accepted else 'insufficient_source_arms_or_column_ink')


def detect_physical_graph(image):
    baseline,previous=v2(image)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY);H,W=gray.shape
    response,dark=ridge_response(gray);parts=multiscale_segments(gray)
    vertical=fit_segments(parts['v'],'v',(H,W));horizontal=fit_segments(parts['h'],'h',(H,W))
    report=dict(version='physical_grid_graph_v3',status='REJECTED',vertical_candidates=len(vertical),
        horizontal_candidates=len(horizontal),edges=[],rejected=[],v2_status=previous['status'],
        evidence_policy='unchanged pixel thresholds; new multiscale observations, no row-count prior')
    # Local source windows prevent long poster/paper edges from consuming the
    # global top-20 candidate budget and hiding a short table's five borders.
    hypotheses=[]
    initial=graph_verticals(vertical,horizontal,W,H)
    if initial is not None:hypotheses.append(initial)
    seen=set()
    for h in horizontal:
        if h.domain[1]-h.domain[0]<W*.2:continue
        y=float(h.eval(((h.domain[0]+h.domain[1])/2-W/2)/W));near=[]
        for v in vertical:
            x=float(v.eval((y-H/2)/H))
            if v.domain[0]-10<=y<=v.domain[1]+10 and h.domain[0]-8<=x<=h.domain[1]+8:
                near.append((x,v))
        unique=[]
        for x,v in sorted(near,key=lambda p:-p[1].support):
            if all(abs(x-a)>4 for a,b in unique):unique.append((x,v))
        vv0=graph_verticals([v for x,v in unique],horizontal,W,H) if len(unique)>=5 else None
        if vv0 is None and 5<=len(unique)<=12:
            # A single common Y interval cannot describe staggered segment ends
            # at a slanted table or a broken VAT boundary. Bootstrap from five
            # observed vertical fragments and >=3 bounded horizontal proposals;
            # final source-arm/ink acceptance below remains unchanged.
            options=[]
            for combo in combinations(sorted(unique,key=lambda p:p[0]),5):
                xx=np.array([x for x,v in combo]);span=xx[-1]-xx[0]
                if not W*.18<span<W*.92:continue
                ratios=np.diff(xx)/span
                if not (.035<ratios[0]<.22 and .32<ratios[1]<.73 and .07<ratios[2]<.30 and .09<ratios[3]<.38):continue
                vv1=[v for x,v in combo];lo=max(v.domain[0] for v in vv1);hi=min(v.domain[1] for v in vv1)
                if hi-lo<40:continue
                bounded=sum(abs(r.domain[0]-xx[0])<span*.10 and abs(r.domain[1]-xx[-1])<span*.10 and
                    min(v.domain[0] for v in vv1)-12<=r.position<=max(v.domain[1] for v in vv1)+12 for r in horizontal)
                if bounded>=3:options.append((bounded-2*float(np.sum(abs(ratios-[.10,.52,.16,.22]))),vv1))
            if options:vv0=max(options,key=lambda p:p[0])[1]
        if vv0 is not None:
            sig=tuple(round(v.position/4) for v in vv0)
            if sig not in seen:hypotheses.append(vv0);seen.add(sig)
    scored=[]
    for hypothesis in hypotheses:
        vv0=junction_outer_boundaries(hypothesis,vertical,horizontal,response)
        count=0;coverage=0.
        for h in horizontal:
            e=evidence(h,vv0,(H,W),response,dark)
            if e and e['accepted']:count+=1;coverage+=e['ink_coverage']
        scored.append((count,coverage,vv0))
    vv=max(scored,key=lambda p:(p[0],p[1]))[2] if scored else None
    report['local_hypotheses']=len(hypotheses)
    if vv is None:
        if baseline is not None:return baseline,dict(report,status='OBSERVED_NOT_VERIFIED',selection='frozen_v2_fallback',edges=previous['edges'])
        report['reason']='no_evidence_backed_five_vertical_hypothesis';return None,report
    vv=junction_outer_boundaries(vv,vertical,horizontal,response)
    accepted=[]
    for rule in horizontal:
        ev=evidence(rule,vv,(H,W),response,dark)
        if ev is None:continue
        # Do not extrapolate an unrelated underline across the table.
        sample=np.linspace(rule.domain[0],rule.domain[1],30)
        y=rule.eval((sample-W/2)/W)
        if not any(v.domain[0]-10<=float(np.median(y))<=v.domain[1]+10 for v in vv):continue
        if ev['accepted']:accepted.append((rule,ev))
        else:report['rejected'].append(ev)
    picked=[]
    for r,e in sorted(accepted,key=lambda p:-p[1]['ink_coverage']):
        duplicate=False
        for old,_ in picked:
            lo=max(r.domain[0],old.domain[0]);hi=min(r.domain[1],old.domain[1])
            if hi>lo:
                t=(np.linspace(lo,hi,50)-W/2)/W
                if np.percentile(abs(r.eval(t)-old.eval(t)),90)<5:duplicate=True;break
        if not duplicate:picked.append((r,e))
    picked.sort(key=lambda p:p[0].position)
    sample=(np.linspace(vv[0].position,vv[-1].position,100)-W/2)/W
    clean=[]
    for r,e in picked:
        if clean and np.min(r.eval(sample)-clean[-1][0].eval(sample))<4:continue
        clean.append((r,e))
    if baseline is not None and len(clean)<=len(baseline['horizontal']):
        return baseline,dict(report,status='OBSERVED_NOT_VERIFIED',selection='frozen_v2_fallback',edges=previous['edges'])
    report['edges']=[e for r,e in clean];report['nodes']=[dict(horizontal=j,vertical=i,point=intersection(v,r,(H,W))) for j,(r,e) in enumerate(clean) for i,v in enumerate(vv)]
    if len(clean)<3:report['reason']='insufficient_jointly_confirmed_multiscale_edges';return None,report
    report.update(status='OBSERVED_NOT_VERIFIED',selection='multiscale_joint_graph',horizontal=len(clean),vertical=5)
    return dict(vertical=vv,horizontal=[r for r,e in clean],shape=(H,W)),report
