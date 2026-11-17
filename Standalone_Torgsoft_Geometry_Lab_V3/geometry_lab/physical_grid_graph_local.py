"""Physical Grid Graph V2: source-evidenced local ridge tracking.

No GT, OCR, target row count or receipt identifiers enter this module. Existing
warpers and V1 are retained verbatim for the diagnostic comparison.
"""
import cv2
import numpy as np
from scipy.signal import find_peaks
from .detection import _candidate_lines,_group_and_fit,_choose_verticals,_fit_group,_clusters,Rule
from .physical_grid_graph import connected_fragments,graph_verticals
from .structure_diagnostics import PolylineRule
from .algorithms import intersection


def ridge_response(gray):
    """Local dark horizontal ridges, not page-level thresholding.

    Vertical black-hat suppresses long vertical strokes. Short horizontal means
    retain faint printed rules without requiring a full-page straight segment.
    """
    boost=cv2.createCLAHE(clipLimit=2.,tileGridSize=(8,8)).apply(gray)
    dark=cv2.morphologyEx(boost,cv2.MORPH_BLACKHAT,np.ones((11,1),np.uint8)).astype(np.float32)
    return cv2.boxFilter(dark,-1,(9,1),normalize=True),dark


def table_local_verticals(gray,initial):
    """Refit observed vertical fragments only over the internal table span.

    This removes barcode continuation from a column fit and avoids rejecting a
    short table border merely because it is less than 13% of the full photo.
    """
    H,W=gray.shape
    top=max(initial[2].domain[0],initial[3].domain[0])
    bottom=min(initial[2].domain[1],initial[3].domain[1])
    if bottom-top<40:return []
    mid=(top+bottom)/2;parts=[]
    for _,s,length in _candidate_lines(gray,'v'):
        x0,y0,x1,y1=map(float,s)
        if y1<y0:x0,y0,x1,y1=x1,y1,x0,y0
        lo=max(y0,top);hi=min(y1,bottom)
        if hi-lo<max(30,(bottom-top)*.25):continue
        slope=(x1-x0)/max(y1-y0,1)
        parts.append((x0+slope*(mid-y0),(x0+slope*(lo-y0),lo,x0+slope*(hi-y0),hi),hi-lo))
    rules=[]
    for group in _clusters(parts,3.):
        points=[]
        for _,(x0,y0,x1,y1),length in group:
            t=np.linspace(0,1,max(3,int(length/10)));points.extend(zip(x0+(x1-x0)*t,y0+(y1-y0)*t))
        p=np.array(points);domain=(float(p[:,1].min()),float(p[:,1].max()))
        if domain[1]-domain[0]<max(35,(bottom-top)*.4):continue
        norm=(p[:,1]-H/2)/H
        coeff=np.polyfit(norm,p[:,0],2 if len(p)>10 else 1)
        rules.append(Rule('v',coeff,float(np.polyval(coeff,0)),(domain[1]-domain[0])/H,domain))
    return rules


def junction_outer_boundaries(vv,candidates,horizontals,response):
    """Resolve paper-vs-table ambiguity by repeated observed T junctions."""
    H,W=response.shape
    def score(v):
        hits=[]
        for h in horizontals:
            x,y=intersection(v,h,(H,W))
            if not (v.domain[0]-8<=y<=v.domain[1]+8 and h.domain[0]-8<=x<=h.domain[1]+8):continue
            xx=x+np.arange(-9,10);yy=h.eval((xx-W/2)/W)
            valid=(xx>=0)&(xx<W)&(yy>=0)&(yy<H)
            if valid.sum()<8:continue
            vals=response[np.rint(yy[valid]).astype(int).clip(0,H-1),np.rint(xx[valid]).astype(int).clip(0,W-1)]
            if (vals>=7).mean()>=.42 and vals.mean()>=7:hits.append(y)
        unique=[]
        for y in sorted(hits):
            if not unique or y-unique[-1]>5:unique.append(y)
        return len(unique)
    result=list(vv)
    for side in [0,4]:
        eligible=[]
        for v in candidates:
            combo=result.copy();combo[side]=v
            lo=max(r.domain[0] for r in combo);hi=min(r.domain[1] for r in combo)
            if hi-lo<40:continue
            xs=np.array([float(r.eval(((lo+hi)/2-H/2)/H)) for r in combo])
            if np.min(np.diff(xs))<5:continue
            span=xs[-1]-xs[0];ratio=np.diff(xs)/span
            if not (.035<ratio[0]<.22 and .32<ratio[1]<.73 and .07<ratio[2]<.30 and .09<ratio[3]<.38):continue
            eligible.append((score(v),-abs(ratio[0 if side==0 else 3]-(.10 if side==0 else .22)),v))
        if eligible:
            best=max(eligible,key=lambda a:(a[0],a[1]))
            if best[0]>=3:result[side]=best[2]
    return result


def trace_seed(response,seed,xs,dark=None):
    """Track local evidence within a bounded band around an observed seed.

    The seed fixes a source location in the middle, preventing a neighbouring
    stronger rule from replacing it. Returned paths are proposals until checked.
    """
    x0,y0,x1,y1=seed;H,W=response.shape
    slope=(y1-y0)/max(abs(x1-x0),1)
    if x1<x0:slope=-slope
    base=y0+slope*(xs-x0)
    radius=max(10,min(24,int((xs[-1]-xs[0])*.045)))
    offsets=np.arange(-radius,radius+1);K=len(offsets);N=len(xs)
    coords=np.rint(base[:,None]+offsets).astype(int)
    valid=(coords>=0)&(coords<H)
    samples=response[coords.clip(0,H-1),np.rint(xs).astype(int).clip(0,W-1)[:,None]]
    rewards=np.minimum(samples/35.,1.8)-.012*abs(offsets)[None,:]
    rewards[~valid]=-20
    anchor=int(np.argmin(abs(xs-(x0+x1)/2)))
    rewards[anchor,abs(offsets)>3]=-1e4
    score=rewards[0].copy();parents=np.zeros((N,K),np.int16)
    for i in range(1,N):
        candidates=[]
        for shift in range(-2,3):
            shifted=np.full(K,-1e6)
            if shift<0:shifted[:shift]=score[-shift:]
            elif shift>0:shifted[shift:]=score[:-shift]
            else:shifted=score.copy()
            candidates.append(shifted-.13*abs(shift))
        candidates=np.array(candidates);choice=np.argmax(candidates,axis=0)
        parents[i]=np.arange(K)-(choice-2)
        score=np.max(candidates,axis=0)+rewards[i]
    index=int(np.argmax(score));indices=np.empty(N,int);indices[-1]=index
    for i in range(N-1,0,-1):indices[i-1]=parents[i,indices[i]]
    ys=base+offsets[indices]
    # Smooth subpixel noise only; no polynomial extrapolation beyond fragments.
    ys=cv2.GaussianBlur(ys.reshape(1,-1).astype(np.float32),(5,1),.8).ravel()
    # A printed edge varies smoothly across the sheet. Robust local ridge fitting
    # prevents a path from zigzagging through aligned glyphs and price digits.
    norm=(xs-(xs[0]+xs[-1])/2)/max(xs[-1]-xs[0],1)
    A=np.vander(norm,4);weight=np.ones(len(xs))
    for _ in range(4):
        coeff=np.linalg.lstsq(A*weight[:,None],ys*weight,rcond=None)[0]
        residual=ys-A@coeff
        weight=np.minimum(1.,2.5/np.maximum(abs(residual),.001))
    ys=A@coeff
    if dark is not None:
        ix=np.rint(xs).astype(int).clip(0,W-1)
        def physical_coverage(path):
            iy=np.rint(path).astype(int).clip(0,H-1)
            pixel=np.maximum.reduce([dark[(iy+d).clip(0,H-1),ix] for d in [-1,0,1]])
            return float(((response[iy,ix]>=7)&(pixel>=9)).mean())
        # Keep a directly observed straight fragment when it has more distributed
        # ink than the tracked path. This prevents nearby glyphs stealing a faint
        # straight rule. Both alternatives face the same junction tests below.
        if physical_coverage(base)>physical_coverage(ys)+.025:ys=base
    vals=response[np.rint(ys).astype(int).clip(0,H-1),np.rint(xs).astype(int).clip(0,W-1)]
    return ys,vals


def detect_physical_graph(image):
    H,W=image.shape[:2];gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    response,dark=ridge_response(gray)
    segments=_candidate_lines(gray,'h')
    groups=connected_fragments(segments,W)
    fitted=[r for g in groups if (r:=_fit_group(g,'h',(H,W))) is not None]
    vertical_candidates=_group_and_fit(gray,'v')
    vv=graph_verticals(vertical_candidates,fitted,W,H)
    fallback=False
    if vv is None:
        vv=_choose_verticals(vertical_candidates,W,H);fallback=True
    report=dict(status='REJECTED',version='local_physical_graph_v2',edges=[],rejected=[],nodes=[],
                vertical_fallback=fallback,semantics='VAT/total unknown without text or human labels')
    if vv is None:report['reason']='five_vertical_proposals_missing';return None,report
    local=table_local_verticals(gray,vv)
    refined=graph_verticals(local,fitted,W,H) if len(local)>=5 else None
    if refined is not None:vv=refined;report['vertical_table_local_refit']=True
    vv=junction_outer_boundaries(vv,local+vertical_candidates,fitted,response)
    ylo=max(0,int(min(v.domain[0] for v in vv))-8)
    yhi=min(H-1,int(max(v.domain[1] for v in vv))+8)
    centers=np.linspace(ylo,yhi,30)
    left=float(np.median(vv[0].eval((centers-H/2)/H)))
    right=float(np.median(vv[-1].eval((centers-H/2)/H)))
    left=max(0,left);right=min(W-1,right)
    if right-left<50:report['reason']='narrow_table';return None,report
    xs=np.linspace(left,right,max(40,int((right-left)/2)+1))
    seeds=[]
    for _,s,length in segments:
        x0,y0,x1,y1=map(float,s)
        midx=(x0+x1)/2;midy=(y0+y1)/2
        if left-5<=midx<=right+5 and ylo<=midy<=yhi and length>max(18,(right-left)*.12):
            seeds.append((length,(x0,y0,x1,y1),'hough_fragment'))
    # Independent short-strip observations near inner column boundaries. These
    # recover rules too faint/curved for global Hough; no regular row spacing.
    for frac in [.10,.32,.60,.78,.92]:
        x=left+(right-left)*frac;half=max(6,int((right-left)*.035))
        strip=response[ylo:yhi+1,max(0,round(x)-half):min(W,round(x)+half+1)]
        profile=np.mean(strip,axis=1)
        peaks,_=find_peaks(profile,height=6.,prominence=3.,distance=5)
        for p in peaks:seeds.append((2*half,(x-half,ylo+p,x+half,ylo+p),'local_strip_ridge'))
    # Prefer long physical fragments; dedup only seed coordinates, not nearby rows.
    seeds.sort(key=lambda s:-s[0]);used=set();proposals=[]
    for length,seed,origin in seeds:
        x0,y0,x1,y1=seed;slope=(y1-y0)/max(x1-x0,1)
        signature=(round((y0+slope*((left+right)/2-x0))/3),round(slope/.04))
        if signature in used:continue
        used.add(signature)
        ys,vals=trace_seed(response,seed,xs,dark)
        points=np.column_stack((xs,ys));rule=PolylineRule(points,'h',(H,W))
        nodes=[intersection(v,rule,image.shape) for v in vv]
        columns=[];confirmed=[]
        ix=np.rint(xs).astype(int).clip(0,W-1);iy=np.rint(ys).astype(int).clip(0,H-1)
        pixel=np.maximum.reduce([dark[(iy+d).clip(0,H-1),ix] for d in [-1,0,1]])
        observed=(vals>=7.)&(pixel>=9.)
        for i in range(4):
            mask=(xs>nodes[i][0]+3)&(xs<nodes[i+1][0]-3)
            columns.append(float(observed[mask].mean()) if mask.any() else 0.)
        for column,(v,(x,y)) in enumerate(zip(vv,nodes)):
            ix,iy=round(x),round(y)
            domain=v.domain[0]-10<=y<=v.domain[1]+10
            # Horizontal stroke approaching the junction is required; darkness
            # at a vertical stroke alone cannot establish a horizontal edge.
            offsets=np.arange(3,11) if column==0 else -np.arange(3,11) if column==4 else np.r_[-np.arange(3,11),np.arange(3,11)]
            armx=x+offsets;army=rule.eval((armx-W/2)/W)
            valid=(armx>=0)&(armx<W)&(army>=1)&(army<H-1)
            ax=np.rint(armx[valid]).astype(int).clip(0,W-1);ay=np.rint(army[valid]).astype(int).clip(0,H-1)
            arms=np.maximum.reduce([dark[(ay+d).clip(0,H-1),ax] for d in [-1,0,1]]) if len(ax) else np.array([])
            horizontal=bool(len(arms)>=5 and (arms>=9).mean()>=.65)
            confirmed.append(domain and horizontal)
        gaps=np.where(np.r_[True,observed[1:]!=observed[:-1],True])[0]
        runs=[dict(x0=float(xs[a]),x1=float(xs[b-1]),observed=bool(observed[a])) for a,b in zip(gaps,gaps[1:])]
        longest_gap=max([r['x1']-r['x0'] for r in runs if not r['observed']]+[0.])/(right-left)
        evidence=dict(position=rule.position,domain=list(rule.domain),origin=origin,seed=list(map(float,seed)),
            ink_coverage=float(observed.mean()),column_coverage=columns,intersection_evidence=confirmed,
            longest_unsupported_fraction=longest_gap,intervals=runs,mean_ridge_contrast=float(vals.mean()))
        # Require distributed physical evidence. Neither topology nor a row-count
        # prior can create a separator in a blank or text-only interval.
        supported_cols=sum(c>=.55 for c in columns)
        accepted=(supported_cols>=4 and sum(confirmed)>=3 and observed.mean()>=.64
                  and longest_gap<=.18 and min(columns[2:])>=.5)
        if not accepted:
            evidence['reason']='insufficient_distributed_physical_evidence';report['rejected'].append(evidence);continue
        # A trace seeded in text must agree with source ink where the seed was.
        local=(xs>=min(x0,x1))&(xs<=max(x0,x1))
        if local.any() and np.median(abs(ys[local]-(y0+slope*(xs[local]-x0))))>4:
            evidence['reason']='path_abandoned_seed';report['rejected'].append(evidence);continue
        proposals.append((rule,evidence))
    picked=[]
    for r,e in sorted(proposals,key=lambda p:-(p[1]['ink_coverage']+.002*p[1]['mean_ridge_contrast'])):
        duplicate=False
        for old,_ in picked:
            distance=abs(r.eval((xs-W/2)/W)-old.eval((xs-W/2)/W))
            if np.median(distance)<4 and np.percentile(distance,90)<6:duplicate=True;break
        if not duplicate:picked.append((r,e))
    picked.sort(key=lambda p:p[0].position)
    # Crossing alternatives represent ambiguous observations, not separate rows.
    # Keep the stronger edge rather than rejecting all valid physical boundaries.
    clean=[]
    for r,e in picked:
        if clean and np.min(r.eval((xs-W/2)/W)-clean[-1][0].eval((xs-W/2)/W))<4:
            old,oe=clean[-1]
            if e['ink_coverage']>oe['ink_coverage']:clean[-1]=(r,e)
            continue
        clean.append((r,e))
    report['edges']=[e for r,e in clean]
    report['vertical_proposals']=[dict(position=v.position,domain=v.domain) for v in vv]
    for j,(r,e) in enumerate(clean):
        for i,v in enumerate(vv):report['nodes'].append(dict(horizontal=j,vertical=i,point=intersection(v,r,image.shape),ink_confirmed=e['intersection_evidence'][i]))
    if len(clean)<3:report['reason']='fewer_than_three_supported_edges';return None,report
    report.update(status='OBSERVED_NOT_VERIFIED',vertical=5,horizontal=len(clean),seeds_traced=len(used))
    return dict(vertical=vv,horizontal=[r for r,e in clean],shape=(H,W)),report
