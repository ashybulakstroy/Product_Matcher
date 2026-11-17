"""Independent GT/segment coverage diagnostics; never turn missing GT into 0%."""
from __future__ import annotations
import cv2
import numpy as np


def draw_lines(image, lines, color=(255,0,0),thickness=2):
    canvas=image.copy()
    for r in lines:
        pts=np.asarray(r['points'],float)
        if len(pts)<2 or not np.isfinite(pts).all():continue
        cv2.polylines(canvas,[np.rint(pts).astype(np.int32)],False,color,thickness,cv2.LINE_AA)
    return canvas


def _mask(shape,polylines):
    mask=np.full(shape[:2],255,np.uint8)
    for p in polylines:
        q=np.asarray(p,float)
        if q.ndim!=2 or q.shape[1]!=2 or len(q)<2 or not np.isfinite(q).all():continue
        cv2.polylines(mask,[np.rint(q).astype(np.int32)],False,0,1,cv2.LINE_8)
    return mask


def _sample(polylines, shape, step=3):
    h,w=shape[:2]
    chunks=[]
    for p in polylines:
        q=np.asarray(p,float)
        if q.ndim!=2 or q.shape[1]!=2 or len(q)<2 or not np.isfinite(q).all():continue
        for a,b in zip(q,q[1:]):
            n=max(2,int(np.linalg.norm(b-a)/step)+1)
            z=a+(b-a)*np.linspace(0,1,n)[:,None]
            chunks.append(z)
    if not chunks:return np.empty((0,2),np.int32)
    coords=np.rint(np.concatenate(chunks)).astype(np.int32)
    coords[:,0]=coords[:,0].clip(0,w-1);coords[:,1]=coords[:,1].clip(0,h-1)
    return coords


def _distance_to(polys, coords, shape):
    if not len(coords) or not polys:return np.empty(0,float)
    m=_mask(shape,polys)
    dist=cv2.distanceTransform(m,cv2.DIST_L2,3)
    return dist[coords[:,1],coords[:,0]].astype(float)


def compare(gt, detected, shape, tolerance=4.0):
    if gt is None:
        return {'status':'GT_MISSING','mean_distance_px':None,'recall_sample_pct':None,
                'precision_sample_pct':None}
    if isinstance(gt,dict) and '_invalid' in gt:
        return {'status':'GT_INVALID','reason':gt['_invalid']}
    confirmed = bool(gt.get('human_confirmed') or gt.get('human_verified') or gt.get('user_confirmed_scope'))
    result={'status':'GT_DIAGNOSTIC_UNCONFIRMED' if not confirmed else 'GT_EVALUATED',
            'warning':'Pixel-sample coverage of CANDIDATE segments, not complete row accuracy'}
    for axis,key in [('h','horizontal_polylines'),('v','vertical_polylines')]:
        truth=gt.get(key,[])
        found=[r['points'] for r in detected if r.get('axis')==axis]
        p_gt=_sample(truth,shape);p_de=_sample(found,shape)
        d1=_distance_to(found,p_gt,shape)
        d2=_distance_to(truth,p_de,shape)
        result[axis]={'annotated_lines':len(truth),'candidate_segments':len(found),
                      'recall_sample_pct':round(float(100*np.mean(d1<=tolerance)),2) if len(d1) else None,
                      'precision_sample_pct':round(float(100*np.mean(d2<=tolerance)),2) if len(d2) else None,
                      'mean_gt_to_candidate_px':round(float(np.mean(d1)),2) if len(d1) else None,
                      'note':'Counts refer to heterogeneous segments and GT polylines, NOT 1-to-1 line count'}
    return result
