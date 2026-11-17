"""Non-destructive diagnostics and reference adapters for the existing solvers."""
import hashlib,json
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from .detection import (_candidate_lines,_clusters,_fit_group,_group_and_fit,
                        _choose_verticals,_choose_horizontals,detect_grid)


class PolylineRule:
    def __init__(self,points,axis,shape):
        self.axis=axis;self.points=np.asarray(points,float)
        along=1 if axis=='v' else 0; across=1-along
        p=self.points[np.argsort(self.points[:,along])]
        if len(p)<2 or not np.isfinite(p).all() or np.any(np.diff(p[:,along])<=0):
            raise ValueError('invalid reference polyline')
        self.extent=shape[0] if axis=='v' else shape[1]
        self.t=p[:,along];self.values=p[:,across]
        self.domain=(float(self.t[0]),float(self.t[-1]))
        self.support=(self.t[-1]-self.t[0])/self.extent
        self.position=float(self.eval(0.))

    def eval(self,coord):
        q=np.asarray(coord)*self.extent+self.extent/2
        result=np.interp(q,self.t,self.values)
        # Existing solvers request a small margin outside support. Linear
        # extrapolation is explicit, and excluded from reference scoring.
        result=np.where(q<self.t[0],self.values[0]+(q-self.t[0])*
                        (self.values[1]-self.values[0])/(self.t[1]-self.t[0]),result)
        result=np.where(q>self.t[-1],self.values[-1]+(q-self.t[-1])*
                        (self.values[-1]-self.values[-2])/(self.t[-1]-self.t[-2]),result)
        return result


def reference_data(gt,shape):
    return dict(shape=shape[:2],vertical=[PolylineRule(p,'v',shape) for p in gt['vertical_polylines']],
                horizontal=[PolylineRule(p,'h',shape) for p in gt['horizontal_polylines']])


def serialize(data):
    if data is None:return None
    H,W=data['shape'];out={}
    for key,axis,extent in [('vertical','v',H),('horizontal','h',W)]:
        out[key]=[]
        for r in data[key]:
            t=np.linspace(*r.domain,80);a=r.eval((t-extent/2)/extent)
            p=np.column_stack((a,t) if axis=='v' else (t,a))
            out[key].append(p.round(4).tolist())
    return out


def fingerprint(data):
    return hashlib.sha256(json.dumps(serialize(data),sort_keys=True).encode()).hexdigest()


def audit_detector(image):
    H,W=image.shape[:2];gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    v=_group_and_fit(gray,'v');five=_choose_verticals(v,W,H)
    segments=_candidate_lines(gray,'h')
    groups=_clusters(segments,max(5,H*.012))
    fitted=[r for g in groups if (r:=_fit_group(g,'h',(H,W))) is not None]
    fitted.sort(key=lambda r:r.position)
    hh=_choose_horizontals(fitted,five,(H,W)) if five else []
    merge_groups=[]
    if five:
        ylo=max(0.,min(r.domain[0] for r in five)-H*.035)
        yhi=min(float(H),max(r.domain[1] for r in five)+H*.035)
        for r in fitted:
            if not (ylo<=r.position<=yhi and r.support>=.13):continue
            if merge_groups and r.position-merge_groups[-1][-1]<max(12,H*.024):merge_groups[-1].append(r.position)
            else:merge_groups.append([r.position])
    data,status=detect_grid(image)
    observations=dict(vertical=five or [],horizontal=hh,shape=(H,W))
    stages=dict(raw_horizontal_segments=len(segments),initial_group_count=len(groups),
                fitted_horizontal_count=len(fitted),selected_horizontal_count=len(hh),
                initial_cluster_tolerance_px=max(5,H*.012),
                second_merge_tolerance_px=max(12,H*.024),
                fitted_positions=[r.position for r in fitted],
                final_positions=[r.position for r in hh],status=status,
                merge_group_positions=merge_groups,
                fitted_observations=serialize(dict(vertical=five or [],horizontal=fitted,shape=(H,W))),
                observations=serialize(observations))
    return data,observations,stages


def match_lines(predicted,reference,shape,tolerance=8.):
    """One-to-one match with unmatched lists; count equality is not accuracy.

    Match by mean perpendicular-axis displacement only within mutual observed
    domain. Threshold is an evaluation setting, never detector feedback.
    """
    result={}
    for key,along,across in [('horizontal',0,1),('vertical',1,0)]:
        pp=predicted.get(key,[]);rr=reference.get(key+'_polylines',[])
        costs=np.full((len(pp),len(rr)),1e6)
        for i,p in enumerate(pp):
            p=np.array(p);p=p[np.argsort(p[:,along])]
            for j,r in enumerate(rr):
                r=np.array(r);r=r[np.argsort(r[:,along])]
                lo=max(p[:,along].min(),r[:,along].min());hi=min(p[:,along].max(),r[:,along].max())
                if hi-lo<5:continue
                t=np.linspace(lo,hi,100)
                costs[i,j]=float(np.mean(abs(np.interp(t,p[:,along],p[:,across])-
                                           np.interp(t,r[:,along],r[:,across]))))
        pairs=[]
        if costs.size:
            ii,jj=linear_sum_assignment(np.minimum(costs,tolerance*3))
            pairs=[dict(detected=int(i),reference=int(j),mean_px=round(float(costs[i,j]),3))
                   for i,j in zip(ii,jj) if costs[i,j]<=tolerance]
        result[key]=dict(detected=len(pp),reference=len(rr),matches=pairs,
            missing=[j for j in range(len(rr)) if not any(p['reference']==j for p in pairs)],
            false=[i for i in range(len(pp)) if not any(p['detected']==i for p in pairs)],
            mean_matched_px=round(float(np.mean([p['mean_px'] for p in pairs])),3) if pairs else None)
    result.update(tolerance_px=tolerance,human_verified=reference.get('human_verified',False))
    return result


def draw_lines(image,lines,color=(255,150,0),labels=True):
    out=image.copy()
    for key in ['vertical','horizontal']:
        for i,line in enumerate(lines.get(key,[])):
            p=np.rint(line).astype(np.int32)
            cv2.polylines(out,[p],False,color,2,cv2.LINE_AA)
            if labels:
                xy=tuple(p[len(p)//2]);cv2.putText(out,f'{key[0]}{i}',xy,cv2.FONT_HERSHEY_SIMPLEX,.45,(255,0,255),1,cv2.LINE_AA)
    return out
