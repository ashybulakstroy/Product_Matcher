"""Experimental perspective + two-axis printed-grid geometry candidate.

Works on PIXELS ONLY (no OCR, no totals, no fixture references). This module
is deliberately conservative: it can reject a candidate instead of reshaping
text when printed rules are insufficient. Image coordinates are BGR OpenCV.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import cv2
import numpy as np
from dotenv import dotenv_values

VERSION = "perspective-2d-grid-dewarp-v1.0"
ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Rule:
    level: float
    profile: np.ndarray
    start: int
    end: int
    coverage: float
    raw_p95: float
    width: int


def setting_mode(root=None):
    root = Path(root) if root is not None else ROOT
    config = dotenv_values(root / '.env', encoding='utf-8-sig')
    mode = str(config.get('OCR_PERSPECTIVE_2D_MODE') or 'shadow').strip().lower()
    if mode not in {'off', 'shadow', 'select'}:
        raise ValueError('OCR_PERSPECTIVE_2D_MODE: ожидается off, shadow или select')
    return mode


def _line_mask(image, vertical=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    src = gray.T.copy() if vertical else gray
    binary = cv2.adaptiveThreshold(cv2.GaussianBlur(src, (3,3), 0), 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 9)
    # For vertical rules, the transposed mask is processed the same way.
    dilated = cv2.dilate(binary, np.ones((3,1),np.uint8))
    opened = cv2.morphologyEx(dilated, cv2.MORPH_OPEN,
                              np.ones((1,max(13,int(src.shape[1]*.055))),np.uint8))
    joined = cv2.morphologyEx(opened, cv2.MORPH_CLOSE,
                              np.ones((3,max(8,int(src.shape[1]*.022))),np.uint8))
    return joined


def _fit_profile(xs, ys, length, degree_limit=3):
    # Fit a robust low-order polynomial; avoid overfitting print defects.
    x = np.asarray(xs,np.float64)
    y = np.asarray(ys,np.float64)
    if len(x) < 8:
        raise ValueError('too_few_rule_points')
    u = (x-length/2)/max(1.,length/2)
    degree = min(degree_limit,len(x)-1)
    weights = np.ones(len(x))
    for _ in range(5):
        co = np.polynomial.polynomial.polyfit(u,y,degree,w=weights)
        resid = y-np.polynomial.polynomial.polyval(u,co)
        scale=max(.8,1.4826*np.median(np.abs(resid-np.median(resid))))
        weights=np.minimum(1.,1.5*scale/np.maximum(np.abs(resid),1e-6))
    full=np.polynomial.polynomial.polyval((np.arange(length)-length/2)/max(1.,length/2),co)
    return full.astype(np.float32),float(np.median(np.abs(resid)))


def detect_rules(image, vertical=False):
    mask = _line_mask(image,vertical)
    h,w = mask.shape
    count,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
    lines=[]
    for lab in range(1,count):
        x,y,dx,dy,area=(int(v) for v in stats[lab])
        mincoverage=.73 if vertical else .18
        if dx < mincoverage*w or dy > .095*h or area < .23*dx:
            continue
        ys,xs=np.where(labels[y:y+dy,x:x+dx]==lab)
        xs=xs+x;ys=ys+y
        samples=[]
        for start in range(x,x+dx,6):
            vals=ys[(xs>=start)&(xs<start+6)]
            if vals.size:
                samples.append((start+2.5,float(np.median(vals))))
        if len(samples)<8:continue
        try:
            profile,fiterr=_fit_profile(*np.asarray(samples).T,w,
                           degree_limit=3 if vertical else (2 if dx > w*.4 else 1))
            # Never extrapolate a short broken line with a high-order curve.
            left=max(0,x);right=min(w-1,x+dx-1)
            deriv_l=float(np.clip(profile[min(w-1,left+2)]-profile[left],-.09,.09))/max(1,min(w-1,left+2)-left)
            deriv_r=float(np.clip(profile[right]-profile[max(0,right-2)],-.09,.09))/max(1,right-max(0,right-2))
            if left>0:profile[:left]=profile[left]+deriv_l*(np.arange(left)-left)
            if right<w-1:profile[right+1:]=profile[right]+deriv_r*(np.arange(right+1,w)-right)
        except ValueError:
            continue
        if fiterr > max(3.8,.008*h):continue
        if not np.isfinite(profile).all():continue
        level=float(np.median(profile[max(0,x):min(w,x+dx)]))
        # A component must actually represent a rule, not a thick letter blob.
        if not 0<level<h:continue
        p95=float(np.percentile(np.abs(profile[x:x+dx]-level),95))
        lines.append(Rule(level,profile,x,x+dx,dx/w,p95,dx))
    lines.sort(key=lambda r:r.level)
    cleaned=[]
    for rule in lines:
        if cleaned and rule.level-cleaned[-1].level < max(7.,h*.006):
            if rule.width>cleaned[-1].width:cleaned[-1]=rule
        else:cleaned.append(rule)
    if not vertical:
        cleaned=_merge_horizontal_fragments(cleaned,w,h)
    return cleaned,mask


def _merge_horizontal_fragments(rules, width, height):
    """Join disjoint fragments of one curved printed rule.

    A curved horizontal line can be split by text into components with
    different median levels.  The polynomial profiles of those components
    meet at the fragment boundary, but their full-width extrapolations may
    cross.  Merge only when the boundary gap is small and the levels meet;
    never merge two long parallel rules.
    """
    rules=list(rules)
    level_limit=max(14.,height*.02)
    boundary_limit=max(6.,height*.01)
    gap_limit=max(10.,width*.02)
    merged=True
    while merged:
        merged=False
        rules.sort(key=lambda r:r.level)
        for i in range(len(rules)-1):
            a,b=rules[i],rules[i+1]
            if b.level-a.level > level_limit:
                continue
            left,right=(a,b) if a.start<=b.start else (b,a)
            gap=right.start-left.end
            if gap>gap_limit:
                continue
            if left.coverage>.55 and right.coverage>.55:
                continue
            if gap>=0:
                xl=min(width-1,max(0,left.end-1))
                xr=min(width-1,max(0,right.start))
                delta=abs(float(right.profile[xr]-left.profile[xl]))
            else:
                lo=max(left.start,right.start);hi=min(left.end,right.end)
                if hi<=lo:
                    continue
                delta=float(np.median(np.abs(left.profile[lo:hi]-right.profile[lo:hi])))
            if delta>boundary_limit:
                continue
            lo=min(left.start,right.start);hi=max(left.end,right.end)
            profile=np.empty(width,np.float32)
            observed=np.zeros(width,bool)
            for x in range(lo,hi):
                in_left=left.start<=x<left.end
                in_right=right.start<=x<right.end
                if in_left and in_right:
                    profile[x]=(left.profile[x]+right.profile[x])*.5
                    observed[x]=True
                elif in_left:
                    profile[x]=left.profile[x];observed[x]=True
                elif in_right:
                    profile[x]=right.profile[x];observed[x]=True
            idx=np.flatnonzero(observed)
            if idx.size<8:
                continue
            profile[idx[0]:idx[-1]+1]=np.interp(
                np.arange(idx[0],idx[-1]+1),idx,profile[idx]).astype(np.float32)
            # Continue with the slope measured at each observed edge.
            left_slope=float(np.clip(profile[idx[0]+2]-profile[idx[0]],-.09,.09))/2
            right_slope=float(np.clip(profile[idx[-1]]-profile[idx[-1]-2],-.09,.09))/2
            if idx[0]>0:
                profile[:idx[0]]=profile[idx[0]]+left_slope*(np.arange(idx[0])-idx[0])
            if idx[-1]<width-1:
                profile[idx[-1]+1:]=profile[idx[-1]]+right_slope*(np.arange(idx[-1]+1,width)-idx[-1])
            level=float(np.median(profile[idx[0]:idx[-1]+1]))
            p95=float(np.percentile(np.abs(profile[idx[0]:idx[-1]+1]-level),95))
            rules[i]=Rule(level,profile,int(idx[0]),int(idx[-1]+1),
                          (idx[-1]+1-idx[0])/width,max(p95,left.raw_p95,right.raw_p95),
                          int(idx[-1]+1-idx[0]))
            rules.pop(i+1)
            merged=True
            break
    return rules


def _warp_axis(levels, paths, size, perpendicular, minsep):
    if len(levels)<2 or np.min(np.diff(levels))<minsep:
        raise ValueError('rule_levels_not_ordered')
    if np.min(np.diff(paths,axis=0))<minsep*.45:
        raise ValueError('curved_rules_intersect')
    xx=np.arange(size,dtype=np.float32)
    output=np.empty((perpendicular,size),np.float32)
    for i in range(perpendicular):
        source=np.interp(xx,levels,paths[:,i]).astype(np.float32)
        lo=xx<levels[0];hi=xx>levels[-1]
        source[lo]=xx[lo]+paths[0,i]-levels[0]
        source[hi]=xx[hi]+paths[-1,i]-levels[-1]
        output[i]=source
    return output


def _stats(image):
    hrs,_=detect_rules(image,False)
    vrs,_=detect_rules(image,True)
    def residual(rules):
        if not rules:return None
        return round(float(np.median([r.raw_p95 for r in rules])),2)
    return dict(horizontal_rules=len(hrs),vertical_rules=len(vrs),
                horizontal_p95_px=residual(hrs),vertical_p95_px=residual(vrs))


def dewarp_grid(image):
    """Return (candidate, source_x, source_y, report), or raise ValueError."""
    h,w=image.shape[:2]
    if min(h,w)<220:raise ValueError('image_too_small')
    before=_stats(image)
    vertical,_=detect_rules(image,vertical=True)
    if len(vertical)!=5 or any(r.coverage<.73 for r in vertical):
        raise ValueError('five_long_vertical_dividers_not_confirmed')
    levels=np.asarray([v.level for v in vertical],float)
    curves=np.asarray([v.profile for v in vertical],float)
    xmap=_warp_axis(levels,curves,w,h,minsep=9.)
    yy=np.broadcast_to(np.arange(h,dtype=np.float32)[:,None],(h,w)).copy()
    aligned_v=cv2.remap(image,xmap,yy,cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT,borderValue=(255,255,255))
    horizontal,_=detect_rules(aligned_v,vertical=False)
    if len(horizontal)<5:
        raise ValueError('insufficient_horizontal_rules_after_vertical_warp')
    if len(horizontal)>70:
        raise ValueError('too_many_horizontal_rule_candidates')
    ylevels=np.asarray([r.level for r in horizontal],float)
    ypaths=np.asarray([r.profile for r in horizontal],float)
    ymap=_warp_axis(ylevels,ypaths,h,w,minsep=9.).T.copy()
    px=np.broadcast_to(np.arange(w,dtype=np.float32)[None,:],(h,w)).copy()
    composed_x=cv2.remap(xmap,px,ymap,cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
    composed_y=ymap
    grad_x_y,grad_x_x=np.gradient(composed_x)
    grad_y_y,grad_y_x=np.gradient(composed_y)
    jac=grad_x_x*grad_y_y-grad_x_y*grad_y_x
    lower,upper=float(np.min(jac)),float(np.max(jac))
    if not np.isfinite(jac).all() or lower < .35 or upper>2.5:
        raise ValueError('unsafe_2d_jacobian')
    candidate=cv2.remap(image,composed_x,composed_y,cv2.INTER_CUBIC,
                       borderMode=cv2.BORDER_CONSTANT,borderValue=(255,255,255))
    after=_stats(candidate)
    return candidate,composed_x,composed_y,dict(version=VERSION,status='candidate_ready',
              before=before,after=after,vertical_levels=[round(float(x),1) for x in levels],
              horizontal_levels=[round(float(x),1) for x in ylevels],
              min_jacobian=round(lower,4),max_jacobian=round(upper,4))


def perspective_candidate(original, polygon, padding=16):
    """Homography of an OBSERVED polygon followed by grid-based 2D dewarp.

    The polygon is passed by the existing detector; no fixture-specific corners
    or expected prices/counts are used in production.
    """
    src=np.asarray(polygon,dtype=np.float32)
    if src.shape!=(4,2) or not np.isfinite(src).all():
        raise ValueError('invalid_polygon')
    tl,tr,br,bl=src
    width=max(np.linalg.norm(tr-tl),np.linalg.norm(br-bl))
    height=max(np.linalg.norm(bl-tl),np.linalg.norm(br-tr))
    if width<85 or height<180 or width*height>12_000_000:
        raise ValueError('polygon_size_unsafe')
    # No isotropic resizing of arbitrary page portions: the destination is a
    # rectangle derived from observed four corners and preserves grid density.
    scale=min(2.,max(1.,500./width))
    cw,ch=round(width*scale),round(height*scale)
    margin=padding
    ow,oh=cw+margin*2,ch+margin*2
    dst=np.float32([[margin,margin],[margin+cw,margin],
                    [margin+cw,margin+ch],[margin,margin+ch]])
    inv=cv2.getPerspectiveTransform(dst,src)
    iy,ix=np.indices((oh,ow),dtype=np.float32)
    points=cv2.perspectiveTransform(np.stack((ix,iy),axis=-1),inv)
    mx,my=points[:,:,0],points[:,:,1]
    baseline=cv2.remap(original,mx,my,cv2.INTER_CUBIC,
              borderMode=cv2.BORDER_CONSTANT,borderValue=(255,255,255))
    candidate,px,py,report=dewarp_grid(baseline)
    final_x=cv2.remap(mx,px,py,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    final_y=cv2.remap(my,px,py,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    return (candidate,final_x.astype(np.float32),final_y.astype(np.float32)),dict(report,perspective_size=[ow,oh]),baseline


def _improves(baseline,proposed):
    """Reject a misleadingly straight but less complete result."""
    b=_stats(baseline);a=_stats(proposed)
    if a['vertical_rules']!=5 or a['horizontal_rules']<5 or a['horizontal_rules']<max(5,int(b['horizontal_rules']*.7)):
        return False,dict(reason='rules_lost_or_incomplete',before=b,after=a)
    if None in (b['vertical_p95_px'],b['horizontal_p95_px'],a['vertical_p95_px'],a['horizontal_p95_px']):
        return False,dict(reason='missing_independent_metrics',before=b,after=a)
    bsum=b['vertical_p95_px']+b['horizontal_p95_px']
    asum=a['vertical_p95_px']+a['horizontal_p95_px']
    passed=(asum+0.4<bsum*.82 and a['vertical_p95_px']<=max(2.,b['vertical_p95_px']*1.15)
            and a['horizontal_p95_px']<=max(2.,b['horizontal_p95_px']*1.15))
    return passed,dict(reason='measured_grid_improvement' if passed else 'insufficient_measured_improvement',
                       before=b,after=a,baseline_error=round(bsum,2),candidate_error=round(asum,2))


def maybe_refine_perspective_grid(original,polygon,current,*,output_dir=None,mode_override=None):
    """Main app adapter. Current (image,map_x,map_y,metadata) stays unchanged on reject."""
    mode=mode_override or setting_mode()
    if mode not in {'off','shadow','select'}:raise ValueError('invalid_mode')
    audit=dict(version=VERSION,mode=mode,status='not_run',selected=False)
    if mode=='off':return current,dict(audit,status='disabled')
    # A previously validated dewarp must not be overridden by experiment.
    if current[3].get('method') in {'grid_curvature_v2','wave_grid','printed_grid_rectified',
                                    'source_grid_crop','contour_coons_patch',
                                    'perspective_grid_2d','experimental_horizontal_only'}:
        return current,dict(audit,status='not_needed',reason='existing_2d_grid_geometry')
    try:
        (proposed,mx,my),report,homography=perspective_candidate(original,polygon)
        audit.update(status='candidate_ready',candidate=report)
        # Compare against perspective-only baseline: same sampling and ROI.
        approved,quality=_improves(homography,proposed)
        current_quality=_stats(current[0])
        audit['current_geometry_quality']=current_quality
        # Also guard against replacing an already better independently rectified
        # working image (e.g. Coons / source-grid fallback).
        q_after=quality['after']
        if (current_quality['vertical_rules']==5 and current_quality['horizontal_rules']>=5 and
            current_quality['vertical_p95_px'] is not None and
            current_quality['horizontal_p95_px'] is not None and
            q_after['vertical_p95_px'] is not None and
            q_after['horizontal_p95_px'] is not None):
            ch,cw=current[0].shape[:2];ph,pw=proposed.shape[:2]
            current_norm=(current_quality['vertical_p95_px']/cw +
                          current_quality['horizontal_p95_px']/ch)
            proposed_norm=(q_after['vertical_p95_px']/pw +
                           q_after['horizontal_p95_px']/ph)
            audit['relative_to_current']=dict(current=round(current_norm,6),candidate=round(proposed_norm,6))
            if proposed_norm>=current_norm*.87:
                approved=False
                quality['reason']='existing_geometry_not_worse'
        audit['quality_gate']=quality
        if output_dir is not None:
            folder=Path(output_dir);folder.mkdir(parents=True,exist_ok=True)
            cv2.imwrite(str(folder/'perspective_only.png'),homography)
            cv2.imwrite(str(folder/'perspective_2d_candidate.png'),proposed)
        if mode=='shadow':
            return current,dict(audit,reason='shadow_mode_no_selection')
        if not approved:
            return current,dict(audit,status='rejected',reason=quality['reason'])
        margin=16
        metadata=dict(method='perspective_grid_2d',algorithm_version=VERSION,margin=margin,
                      body_size=[int(proposed.shape[1]-2*margin),int(proposed.shape[0]-2*margin)],
                      warnings=[],source_polygon=np.asarray(polygon,float).tolist(),
                      perspective_2d_assessment=report)
        return (proposed,mx,my,metadata),dict(audit,status='selected',selected=True,reason=quality['reason'])
    except (ValueError,TypeError,KeyError,IndexError,cv2.error,np.linalg.LinAlgError) as error:
        return current,dict(audit,status='rejected',reason=f'{type(error).__name__}: {error}')
