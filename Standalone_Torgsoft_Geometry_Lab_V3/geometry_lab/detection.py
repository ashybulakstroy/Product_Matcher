"""Independent source-image grid observation. Never reads baseline layout."""
from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class Rule:
    axis: str
    coeff: np.ndarray  # polynomial in normalized full-image coordinate
    position: float
    support: float
    domain: tuple[float, float]

    def eval(self, coord):
        return np.polyval(self.coeff, coord)


def _clusters(parts, tolerance):
    parts = sorted(parts, key=lambda p: p[0])
    groups=[]
    for item in parts:
        if not groups or abs(item[0] - np.median([p[0] for p in groups[-1]])) > tolerance:
            groups.append([item])
        else:
            groups[-1].append(item)
    return groups


def _candidate_lines(gray, axis):
    h,w=gray.shape
    # Both adaptive and global contrast; expose missing rather than inventing separators.
    boosted=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(gray)
    binary=cv2.adaptiveThreshold(cv2.GaussianBlur(boosted,(3,3),0),255,
              cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,8)
    if axis=="v":
        k=cv2.getStructuringElement(cv2.MORPH_RECT,(1,max(9,h//120)))
        min_length=max(24,h//22)
    else:
        k=cv2.getStructuringElement(cv2.MORPH_RECT,(max(9,w//120),1))
        min_length=max(24,w//22)
    m=cv2.morphologyEx(binary,cv2.MORPH_OPEN,k)
    segments=cv2.HoughLinesP(m,1,np.pi/720,
                    threshold=max(20,min_length//3),minLineLength=min_length,
                    maxLineGap=max(8,int(max(h,w)*.014)))
    if segments is None:
        return []
    result=[]
    for x0,y0,x1,y1 in segments.reshape(-1,4):
        dx=float(x1-x0);dy=float(y1-y0)
        if axis=="v":
            if abs(dy)<min_length or abs(dx)>abs(dy)*.65:continue
            # ref position at center of image; distance from line endpoint
            ref=float(x0 + (w*.0 + h/2-y0)*dx/dy)
            result.append((ref,(x0,y0,x1,y1),abs(dy)))
        else:
            if abs(dx)<min_length or abs(dy)>abs(dx)*.65:continue
            ref=float(y0 + (w/2-x0)*dy/dx)
            result.append((ref,(x0,y0,x1,y1),abs(dx)))
    return result


def _fit_group(group, axis, shape):
    h,w=shape
    points=[]
    for _,(x0,y0,x1,y1),length in group:
        ns=max(3,int(length/12))
        a=np.linspace(0,1,ns)
        x=x0+(x1-x0)*a; y=y0+(y1-y0)*a
        points.extend(zip(x,y))
    pts=np.array(points,dtype=float)
    along=pts[:,1] if axis=="v" else pts[:,0]
    across=pts[:,0] if axis=="v" else pts[:,1]
    extent=h if axis=="v" else w
    if np.ptp(along)<extent*.13:
        return None
    norm=(along-extent/2)/max(extent,1)
    degree=2 if len(norm)>8 and np.ptp(norm)>.25 else 1
    # Least-squares robust iteration: downweight outlying short segments.
    A=np.vander(norm,degree+1)
    weight=np.ones_like(norm)
    for _ in range(4):
        co=np.linalg.lstsq(A*weight[:,None],across*weight,rcond=None)[0]
        resid=across-A@co
        scale=max(1.5,float(np.median(abs(resid)))*2.5)
        weight=np.clip(scale/np.maximum(abs(resid),.001),.1,1.)
    center=float(np.polyval(co,0.))
    return Rule(axis,co,center,float(np.ptp(along)/extent),
                (float(np.min(along)),float(np.max(along))))


def _group_and_fit(gray,axis):
    h,w=gray.shape
    grouped=_clusters(_candidate_lines(gray,axis),max(5,(w if axis=="v" else h)*.012))
    output=[]
    for g in grouped:
        rule=_fit_group(g,axis,(h,w))
        if rule is not None:
            output.append(rule)
    return sorted(output,key=lambda p:p.position)


def _choose_verticals(rules, width, height):
    # Five physical boundaries of four printed columns. No coordinate leakage from baseline.
    from itertools import combinations
    if len(rules)<5:return None
    # Cap excessive clutter/combination explosion; preserve most supported lines.
    candidates=sorted(sorted(rules,key=lambda r:-r.support)[:20],key=lambda r:r.position)
    best=None
    for combo in combinations(candidates,5):
        top=max(c.domain[0] for c in combo)
        bottom=min(c.domain[1] for c in combo)
        if bottom-top < height*.08: continue
        sample=(np.linspace(top,bottom,30)-height/2)/height
        observed=np.stack([c.eval(sample) for c in combo])
        if np.min(np.diff(observed,axis=0)) < 5: continue
        xs=np.median(observed,axis=1)
        span=xs[-1]-xs[0]
        if not width*.18<span<width*.92:continue
        ratios=np.diff(xs)/span
        if not (.035<ratios[0]<.22 and .32<ratios[1]<.73 and .07<ratios[2]<.30 and .09<ratios[3]<.38):continue
        support=sum(min(c.support,(bottom-top)/height) for c in combo)
        score=support-4*np.sum(abs(ratios-[.10,.52,.16,.22]))
        if best is None or score>best[0]:best=(score,list(combo))
    return best[1] if best else None


def _choose_horizontals(rules, verticals, shape):
    # Filter by actual intersection with grid X span and overlap in Y; no invented rows.
    h,w=shape
    y_low=max(0.,min(r.domain[0] for r in verticals)-h*.035)
    y_hi=min(float(h),max(r.domain[1] for r in verticals)+h*.035)
    picked=[r for r in rules if y_low<=r.position<=y_hi and r.support>=.13]
    # Wavy rules are often split into several near-parallel Hough groups.
    # Merge nearby observations instead of counting one curved rule as 2-3 rows.
    # Only merge within a fraction of the image height; never insert new rows.
    groups=[]
    for rule in sorted(picked,key=lambda q:q.position):
        if groups and rule.position-groups[-1][-1].position < max(12.,h*.024):
            groups[-1].append(rule)
        else:
            groups.append([rule])
    result=[]
    for group in groups:
        if len(group)==1:
            result.append(group[0]);continue
        weights=np.array([max(r.support,.05) for r in group])
        coeff=np.average(np.stack([np.pad(r.coeff,(3-len(r.coeff),0)) for r in group]),
                         axis=0,weights=weights)
        result.append(Rule('h',coeff,float(np.polyval(coeff,0)),
                           max(r.support for r in group),
                           (min(r.domain[0] for r in group),max(r.domain[1] for r in group))))
    supported=[]
    for r in result:
        left=float(verticals[0].eval((r.position-h/2)/h))
        right=float(verticals[-1].eval((r.position-h/2)/h))
        slack=max(8.,(right-left)*.16)
        if r.domain[0] <= left+slack and r.domain[1] >= right-slack:
            supported.append(r)
    return supported


def detect_grid(image):
    h,w=image.shape[:2]
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    vertical=_group_and_fit(gray,"v")
    five=_choose_verticals(vertical,w,h)
    if five is None:return None,{"status":"REJECTED","reason":"five_printed_verticals_not_confirmed","detected_vertical":len(vertical)}
    horizontal=_choose_horizontals(_group_and_fit(gray,"h"),five,(h,w))
    if len(horizontal)<3:
        return None,{"status":"REJECTED","reason":"three_printed_horizontals_not_confirmed","vertical":5,"horizontal":len(horizontal)}
    # Reject crossing boundaries everywhere in their supported Y/X ranges.
    yy=np.linspace(max(0,max(r.domain[0] for r in five)),min(h,min(r.domain[1] for r in five)),40)
    xx=np.linspace(five[0].position,five[-1].position,40)
    if any(np.min(five[i+1].eval((yy-h/2)/h)-five[i].eval((yy-h/2)/h))<5 for i in range(4)):
        return None,{"status":"REJECTED","reason":"crossed_vertical_lines"}
    if any(np.min(horizontal[i+1].eval((xx-w/2)/w)-horizontal[i].eval((xx-w/2)/w))<4 for i in range(len(horizontal)-1)):
        return None,{"status":"REJECTED","reason":"crossed_horizontal_lines"}
    data={"vertical":five,"horizontal":horizontal,"shape":(h,w)}
    return data,{"status":"OBSERVED","vertical":len(five),"horizontal":len(horizontal),
                 "note":"candidate observations, NOT ground truth"}
