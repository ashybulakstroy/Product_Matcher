"""Geometry safety + independent manual ground-truth distance checks."""
from __future__ import annotations
import cv2
import numpy as np


def safety(result, source_shape):
    mx,my=result.map_x,result.map_y
    h,w=source_shape[:2]
    if mx.shape!=my.shape or mx.shape!=result.image.shape[:2]:
        return {'status':'invalid_map_shapes'}
    finite=np.isfinite(mx)&np.isfinite(my)
    inside=finite&(mx>=0)&(my>=0)&(mx<w)&(my<h)
    ax=np.gradient(mx.astype(float),axis=1)
    ay=np.gradient(mx.astype(float),axis=0)
    bx=np.gradient(my.astype(float),axis=1)
    by=np.gradient(my.astype(float),axis=0)
    jac=ax*by-ay*bx
    valid=inside&(jac>0.02)
    return {'status':'MEASURED','source_valid_fraction':round(float(inside.mean()),5),
            'folded_fraction':round(float(((jac<=0)&finite).mean()),5),
            'valid_fraction':round(float(valid.mean()),5),
            'jacobian_p01':round(float(np.percentile(jac[finite],1)),5) if finite.any() else None,
            'jacobian_min':round(float(np.min(jac[finite])),5) if finite.any() else None,
            'no_ground_truth_note':'Jacobian and pixel coverage are not print-grid accuracy'}


def _sample_mapping(mx,my,points):
    coords=np.asarray(points,dtype=np.float32)
    qx=coords[:,0].reshape(1,-1).clip(0,mx.shape[1]-1)
    qy=coords[:,1].reshape(1,-1).clip(0,mx.shape[0]-1)
    sx=cv2.remap(mx,qx,qy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE).reshape(-1)
    sy=cv2.remap(my,qx,qy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE).reshape(-1)
    return np.column_stack((sx,sy))


def projected_lines(result, n=100):
    xs,ys=result.xs,result.ys
    vertical=[];horizontal=[]
    for x in xs:
        p=np.column_stack((np.full(n,x),np.linspace(ys[0],ys[-1],n)))
        vertical.append(_sample_mapping(result.map_x,result.map_y,p))
    for y in ys:
        p=np.column_stack((np.linspace(xs[0],xs[-1],n),np.full(n,y)))
        horizontal.append(_sample_mapping(result.map_x,result.map_y,p))
    return vertical,horizontal


def _line_mask(shape,points):
    h,w=shape[:2]
    canvas=np.full((h,w),255,np.uint8)
    for a,b in zip(points,points[1:]):
        x0,y0=np.rint(a).astype(int);x1,y1=np.rint(b).astype(int)
        cv2.line(canvas,(x0,y0),(x1,y1),0,1,cv2.LINE_8)
    return canvas


def evaluate_ground_truth(result, gt, source_shape):
    """Compare to human-annotated SOURCE polylines; no self-grading on detector output."""
    v,h=projected_lines(result)
    observed_v=gt.get('vertical_polylines',[])
    observed_h=gt.get('horizontal_polylines',[])
    if len(v)!=len(observed_v) or len(h)!=len(observed_h):
        return {'status':'GT_MISMATCH','expected_vertical':len(v),'annotated_vertical':len(observed_v),
                'expected_horizontal':len(h),'annotated_horizontal':len(observed_h),
                'missing_horizontal':max(0,len(observed_h)-len(h)),
                'extra_horizontal':max(0,len(h)-len(observed_h)),
                'reference_origin':gt.get('annotation_origin','unspecified'),
                'human_verified':gt.get('human_verified',False)}
    entries=[]
    for kind,predicted,reference in [('vertical',v,observed_v),('horizontal',h,observed_h)]:
        for i,(estimated,actual) in enumerate(zip(predicted,reference)):
            if len(actual)<2:
                return {'status':'GT_INVALID','reason':'polyline_less_than_two_points'}
            actual=np.asarray(actual,float)
            if actual.ndim!=2 or actual.shape[1]!=2 or not np.isfinite(actual).all():
                return {'status':'GT_INVALID','reason':'nonfinite_or_invalid_polyline'}
            mask=_line_mask(source_shape,actual)
            distances=cv2.distanceTransform(mask,cv2.DIST_L2,3)
            actual=np.asarray(actual,float)
            axis=1 if kind=='vertical' else 0
            mask=(estimated[:,axis]>=actual[:,axis].min()) & (estimated[:,axis]<=actual[:,axis].max())
            mask &= (estimated[:,0]>=0)&(estimated[:,0]<source_shape[1])&(estimated[:,1]>=0)&(estimated[:,1]<source_shape[0])
            if not mask.any():
                return {'status':'GT_NO_COVERAGE','axis':kind,'index':i}
            coords=np.rint(estimated[mask]).astype(int)
            coords[:,0]=np.clip(coords[:,0],0,source_shape[1]-1)
            coords[:,1]=np.clip(coords[:,1],0,source_shape[0]-1)
            vals=distances[coords[:,1],coords[:,0]]
            entries.append({'axis':kind,'index':i,'coverage':round(float(mask.mean()),3),'mean_px':round(float(np.mean(vals)),2),
                            'p95_px':round(float(np.percentile(vals,95)),2)})
    return {'status':'GT_EVALUATED','line_metrics':entries,
            'mean_px':round(float(np.mean([x['mean_px'] for x in entries])),2),
            'p95_line_px':round(float(np.percentile([x['p95_px'] for x in entries],95)),2),
            'reference_origin':gt.get('annotation_origin','unspecified'),
            'human_verified':gt.get('human_verified',False),
            'mean_px_at_width_1500':round(float(np.mean([x['mean_px'] for x in entries]))*1500/source_shape[1],2),
            'note':'Exploratory source tracing; accuracy is limited by verification, coverage and annotation uncertainty'}


def render_overlay(result, source):
    """Blue proposed grid + green independently redetected print strokes on output."""
    output=result.image.copy()
    for x in result.xs:
        cv2.line(output,(round(x),round(result.ys[0])),
                 (round(x),round(result.ys[-1])),(255,0,0),1,cv2.LINE_AA)
    for y in result.ys:
        cv2.line(output,(round(result.xs[0]),round(y)),
                 (round(result.xs[-1]),round(y)),(255,0,0),1,cv2.LINE_AA)
    # Do not use redetection as a ground-truth accuracy estimate.
    from .detection import _group_and_fit
    gray=cv2.cvtColor(result.image,cv2.COLOR_BGR2GRAY)
    count={'vertical':0,'horizontal':0}
    for axis,kind in [('v','vertical'),('h','horizontal')]:
        lines=_group_and_fit(gray,axis)
        count[kind]=len(lines)
        if axis=='v':
            for rule in lines:
                ys=np.linspace(result.ys[0],result.ys[-1],80)
                xs=rule.eval((ys-gray.shape[0]/2)/gray.shape[0])
                pts=np.rint(np.column_stack((xs,ys))).astype(np.int32)
                cv2.polylines(output,[pts],False,(0,190,0),1,cv2.LINE_AA)
        else:
            for rule in lines:
                xs=np.linspace(result.xs[0],result.xs[-1],80)
                ys=rule.eval((xs-gray.shape[1]/2)/gray.shape[1])
                pts=np.rint(np.column_stack((xs,ys))).astype(np.int32)
                cv2.polylines(output,[pts],False,(0,190,0),1,cv2.LINE_AA)
    return output,count


def render_source_overlay(result,source):
    out=source.copy()
    v,h=projected_lines(result)
    for line in v+h:
        cv2.polylines(out,[np.rint(line).astype(np.int32)],False,(255,0,0),2,cv2.LINE_AA)
    return out
