"""Experimental source-only printed-grid graph. No OCR or expected row count.

Vertical proposals currently reuse V3's detector (a declared shared limitation).
Horizontal fragments are connected by local geometric agreement, never by page
height or desired row count. Unsupported gaps remain explicitly inferred.
"""
from __future__ import annotations
import cv2
import numpy as np
from .detection import (_candidate_lines, _fit_group, _group_and_fit,
                        _choose_verticals)
from .algorithms import intersection


def graph_verticals(rules,horizontals,width,height):
    """Score five-column hypotheses by shared bounded horizontal edges.

    Paper edges are long but do not terminate the repeated printed rules. No
    row-count prior: count only measured fragments with both observed endpoints.
    """
    from itertools import combinations
    candidates=sorted(sorted(rules,key=lambda r:-r.support)[:20],key=lambda r:r.position)
    best=None
    for vv in combinations(candidates,5):
        top=max(v.domain[0] for v in vv);bottom=min(v.domain[1] for v in vv)
        if bottom-top<height*.06:continue
        samples=(np.linspace(top,bottom,12)-height/2)/height
        observed=np.stack([v.eval(samples) for v in vv]);xs=np.median(observed,axis=1)
        if np.min(np.diff(observed,axis=0))<5:continue
        span=xs[-1]-xs[0]
        if not width*.18<span<width*.92:continue
        ratios=np.diff(xs)/span
        if not (.035<ratios[0]<.22 and .32<ratios[1]<.73 and .07<ratios[2]<.30 and .09<ratios[3]<.38):continue
        bounded=[];endpoint_errors=[]
        for r in horizontals:
            if not top-8<=r.position<=bottom+8:continue
            left=float(vv[0].eval((r.position-height/2)/height));right=float(vv[-1].eval((r.position-height/2)/height))
            if abs(r.domain[0]-left)<span*.12 and abs(r.domain[1]-right)<span*.12:
                bounded.append(r)
                endpoint_errors.append((abs(r.domain[0]-left)+abs(r.domain[1]-right))/span)
        # De-duplicate thickness without collapsing neighbouring physical rows.
        positions=[]
        for r in sorted(bounded,key=lambda r:r.position):
            if not positions or r.position-positions[-1]>4:positions.append(r.position)
        score=len(positions)-2*np.sum(abs(ratios-[.10,.52,.16,.22]))-5*(float(np.mean(endpoint_errors)) if endpoint_errors else 1.)
        if best is None or score>best[0]:best=(score,list(vv),len(positions))
    if best is None or best[2]<3:return None
    return best[1]


def connected_fragments(parts, width, tolerance=3.0):
    """Join collinear fragments only where they overlap or have a short gap.

    Complete-link compatibility prevents transitive chains joining adjacent rows.
    """
    def compatible(a,b):
        _,(ax,ay,bx,by),_=a; _,(cx,cy,dx,dy),_=b
        if bx<ax: ax,ay,bx,by=bx,by,ax,ay
        if dx<cx: cx,cy,dx,dy=dx,dy,cx,cy
        ma=(by-ay)/max(bx-ax,1); mb=(dy-cy)/max(dx-cx,1)
        lo=max(ax,cx); hi=min(bx,dx)
        if lo-hi > width*.18 or abs(ma-mb)>.12:return False
        sample=np.linspace(min(lo,hi),max(lo,hi),3)
        return bool(np.max(abs(ay+ma*(sample-ax)-cy-mb*(sample-cx)))<=tolerance)
    groups=[]
    for p in sorted(parts,key=lambda p:-p[2]):
        possible=[g for g in groups if any(compatible(p,q) for q in g)
                  and all(compatible(p,q) for q in g
                          if min(max(p[1][0],p[1][2]),max(q[1][0],q[1][2])) >=
                             max(min(p[1][0],p[1][2]),min(q[1][0],q[1][2])))]
        if possible:max(possible,key=lambda g:sum(q[2] for q in g)).append(p)
        else:groups.append([p])
    return groups


def detect_physical_graph(image):
    H,W=image.shape[:2];gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    groups=connected_fragments(_candidate_lines(gray,'h'),W)
    horizontal_rules=[r for g in groups if (r:=_fit_group(g,'h',(H,W))) is not None]
    vv=graph_verticals(_group_and_fit(gray,'v'),horizontal_rules,W,H)
    report=dict(status='REJECTED',vertical_source='V3 proposals, graph-scored five-boundary selection',
                rows_semantics='UNKNOWN: geometry cannot identify VAT text',
                footer=[],edges=[],rejected=[],nodes=[])
    if vv is None:
        report['reason']='five_verticals_not_observed';return None,report
    # Directional ink evidence: a word/fold alone is insufficient without multiple
    # column crossings. Morphology is horizontal only; no synthetic gap closing.
    boosted=cv2.createCLAHE(clipLimit=2.,tileGridSize=(8,8)).apply(gray)
    binary=cv2.adaptiveThreshold(cv2.GaussianBlur(boosted,(3,3),0),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV,31,8)
    ink=cv2.morphologyEx(binary,cv2.MORPH_OPEN,np.ones((1,9),np.uint8))
    dilated=cv2.dilate(ink,np.ones((5,3),np.uint8))
    proposals=[]
    ylo=min(v.domain[0] for v in vv)-8;yhi=max(v.domain[1] for v in vv)+8
    for group in groups:
        rule=_fit_group(group,'h',(H,W))
        if rule is None or not ylo<=rule.position<=yhi:continue
        nodes=[intersection(v,rule,image.shape) for v in vv]
        left,right=nodes[0][0],nodes[-1][0]
        if right-left<30:continue
        xs=np.linspace(left,right,max(30,int(right-left)))
        ys=rule.eval((xs-W/2)/W)
        valid=(xs>=0)&(xs<W)&(ys>=0)&(ys<H)
        supported=np.zeros(len(xs),bool)
        supported[valid]=dilated[np.rint(ys[valid]).astype(int).clip(0,H-1),
                                 np.rint(xs[valid]).astype(int).clip(0,W-1)]>0
        col_support=[]
        for i in range(4):
            region=(xs>nodes[i][0]+3)&(xs<nodes[i+1][0]-3)
            col_support.append(float(supported[region].mean()) if region.any() else 0.)
        confirmed=[]
        for vi,(x,y) in zip(vv,nodes):
            observed_domain=(vi.domain[0]-8<=y<=vi.domain[1]+8 and rule.domain[0]-8<=x<=rule.domain[1]+8)
            x,y=round(x),round(y)
            confirmed.append(bool(observed_domain and 0<=x<W and 0<=y<H and
                ink[max(0,y-4):min(H,y+5),max(0,x-6):min(W,x+7)].any()))
        ev=dict(position=rule.position,domain=list(rule.domain),
                ink_coverage=round(float(supported.mean()),3),
                column_coverage=col_support,intersection_evidence=confirmed,
                fragments=[list(map(int,g[1])) for g in group],
                inferred_gap_samples=int((~supported).sum()))
        transitions=np.r_[0,np.where(supported[1:]!=supported[:-1])[0]+1,len(xs)]
        ev['intervals']=[dict(x0=round(float(xs[a]),2),x1=round(float(xs[b-1]),2),
            evidence='observed_horizontal_ink' if supported[a] else 'inferred_gap_not_detected')
            for a,b in zip(transitions,transitions[1:])]
        columns=sum(c>=.5 for c in col_support)
        bounded=abs(rule.domain[0]-left)<(right-left)*.15 and abs(rule.domain[1]-right)<(right-left)*.15
        if columns>=3 and sum(confirmed)>=3 and supported.mean()>=.58 and bounded:
            proposals.append((rule,ev))
        elif columns>=1:
            ev['kind']=('right_total_box_candidate_semantics_unverified'
                        if max(col_support[:2])<.25 and min(col_support[2:])>.6
                        else 'partial_rule_or_text_unresolved')
            report['footer'].append(ev)
        else:report['rejected'].append(ev)
    # Duplicate strokes differ by ink thickness only. Never average close rows.
    picked=[]
    for r,e in sorted(proposals,key=lambda p:-p[1]['ink_coverage']):
        duplicate=False
        for old,_ in picked:
            lo=max(r.domain[0],old.domain[0]);hi=min(r.domain[1],old.domain[1])
            if hi>lo:
                sample=(np.linspace(lo,hi,20)-W/2)/W
                if np.max(abs(r.eval(sample)-old.eval(sample)))<=3:
                    duplicate=True;break
        if not duplicate:picked.append((r,e))
    picked.sort(key=lambda p:p[0].position)
    report['edges']=[e for _,e in picked]
    for j,(r,e) in enumerate(picked):
        for i,v in enumerate(vv):
            report['nodes'].append(dict(horizontal=j,vertical=i,
                point=intersection(v,r,image.shape),ink_confirmed=e['intersection_evidence'][i]))
    if len(picked)<3:
        report['reason']='insufficient_evidence_backed_horizontal_edges';return None,report
    hh=[r for r,_ in picked]
    x=np.linspace(vv[0].position,vv[-1].position,80)
    if any(np.min(b.eval((x-W/2)/W)-a.eval((x-W/2)/W))<4 for a,b in zip(hh,hh[1:])):
        report['reason']='crossing_horizontal_proposals';return None,report
    report.update(status='OBSERVED_NOT_VERIFIED',horizontal=len(hh),vertical=5,
                  footer_semantics='partial rules separate; VAT/total need text or human confirmation')
    return dict(vertical=vv,horizontal=hh,shape=(H,W)),report
