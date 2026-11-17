"""Implemented algorithms. No imports from App or previous Trial versions."""
from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np
from .detection import detect_grid


@dataclass
class Result:
    method: str
    image: np.ndarray
    map_x: np.ndarray
    map_y: np.ndarray
    xs: list[float]
    ys: list[float]
    info: dict


def intersection(v,h,image_shape):
    H,W=image_shape[:2]
    x=float(v.position);y=float(h.position)
    for _ in range(10):
        x=float(v.eval((y-H/2)/H))
        y=float(h.eval((x-W/2)/W))
    return x,y


def _dest(data, margin=16):
    five=data['vertical'];horizontal=data['horizontal']
    x=np.array([v.position for v in five],dtype=float)
    y=np.array([v.position for v in horizontal],dtype=float)
    if min(np.diff(x))<6 or min(np.diff(y))<5:
        raise ValueError('invalid_grid_spacing')
    # Preserve observed cell width/height rather than forcing equal columns/rows.
    xs=(x-x[0]+margin).tolist()
    ys=(y-y[0]+margin).tolist()
    W=int(np.ceil(xs[-1]+margin));H=int(np.ceil(ys[-1]+margin))
    if min(H,W)<30 or max(H,W)>3000 or H*W>5_000_000:
        raise ValueError('unreasonable_output_size')
    return xs,ys,(H,W)


def _image_sample(image, mx, my):
    out=cv2.remap(image,mx.astype(np.float32),my.astype(np.float32),
                  cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT,borderValue=(255,255,255))
    return out


def perspective(image,data):
    xs,ys,(H,W)=_dest(data)
    v=data['vertical'];h=data['horizontal']
    corners=np.array([intersection(v[0],h[0],image.shape),
             intersection(v[-1],h[0],image.shape),
             intersection(v[-1],h[-1],image.shape),
             intersection(v[0],h[-1],image.shape)],np.float32)
    dest=np.array([[xs[0],ys[0]],[xs[-1],ys[0]],
                   [xs[-1],ys[-1]],[xs[0],ys[-1]]],np.float32)
    Hmat=cv2.getPerspectiveTransform(dest,corners)
    gy,gx=np.indices((H,W),dtype=np.float32)
    pts=np.stack((gx.ravel(),gy.ravel()),axis=-1).reshape(-1,1,2)
    src=cv2.perspectiveTransform(pts,Hmat).reshape(H,W,2)
    mx,my=src[:,:,0],src[:,:,1]
    return Result('perspective',_image_sample(image,mx,my),mx,my,xs,ys,
                  {'model':'single_homography','physical_grid':'detected_source'})


def mesh(image,data):
    """Inverse curvilinear chart: enforce x=x_i(y) and y=y_j(x) jointly.

    Iterate source positions for each target coordinate. When output x or y equals
    a control boundary, it is anchored to the observed source curve (not guessed).
    """
    xs,ys,(H,W)=_dest(data)
    ih,iw=image.shape[:2]
    gy,gx=np.indices((H,W),dtype=np.float64)
    xs_arr=np.array(xs);ys_arr=np.array(ys)
    xi=np.clip(np.searchsorted(xs_arr,gx,side='right')-1,0,len(xs)-2)
    yi=np.clip(np.searchsorted(ys_arr,gy,side='right')-1,0,len(ys)-2)
    a=(gx-xs_arr[xi])/(xs_arr[xi+1]-xs_arr[xi]);
    b=(gy-ys_arr[yi])/(ys_arr[yi+1]-ys_arr[yi])
    xx=gx-xs[0]+data['vertical'][0].position
    yy=gy-ys[0]+data['horizontal'][0].position
    vv=data['vertical'];hh=data['horizontal']
    for _ in range(8):
        normy=(yy-ih/2)/ih
        vstack=np.stack([rule.eval(normy) for rule in vv],axis=0)
        x0=np.take_along_axis(vstack,xi[None,:,:],axis=0)[0]
        x1=np.take_along_axis(vstack,(xi+1)[None,:,:],axis=0)[0]
        xx=x0+a*(x1-x0)
        normx=(xx-iw/2)/iw
        hstack=np.stack([rule.eval(normx) for rule in hh],axis=0)
        y0=np.take_along_axis(hstack,yi[None,:,:],axis=0)[0]
        y1=np.take_along_axis(hstack,(yi+1)[None,:,:],axis=0)[0]
        yy=y0+b*(y1-y0)
    mx=xx.astype(np.float32);my=yy.astype(np.float32)
    return Result('grid_mesh',_image_sample(image,mx,my),mx,my,xs,ys,
                  {'model':'coupled_curvilinear_inverse','iterations':8,
                   'physical_grid':'observed_lines_source'})


def tps(image,data):
    try:
        from scipy.interpolate import RBFInterpolator
    except ImportError as e:
        raise RuntimeError('scipy_optional_dependency_missing') from e
    xs,ys,(H,W)=_dest(data)
    v=data['vertical'];h=data['horizontal']
    targets=np.array([(x,y) for y in ys for x in xs],np.float64)
    observed=np.array([intersection(vi,hi,image.shape) for hi in h for vi in v],np.float64)
    if len(targets)<10:raise ValueError('too_few_anchors_for_tps')
    model=RBFInterpolator(targets,observed,kernel='thin_plate_spline',
                          degree=1,smoothing=.10)
    y,x=np.indices((H,W),dtype=np.float32)
    coords=np.column_stack((x.ravel(),y.ravel())).astype(np.float64)
    samples=np.empty((len(coords),2),dtype=np.float32)
    for i in range(0,len(coords),16384):
        samples[i:i+16384]=model(coords[i:i+16384])
    mx=samples[:,0].reshape(H,W)
    my=samples[:,1].reshape(H,W)
    return Result('tps_scipy',_image_sample(image,mx,my),mx,my,xs,ys,
                  {'model':'scipy_RBFInterpolator_thin_plate_spline',
                   'anchors':len(targets),'physical_grid':'intersections_only'})


METHODS={'perspective':perspective,'grid_mesh':mesh,'tps_scipy':tps}


def detect_and_run(image, method):
    if method not in METHODS:raise KeyError(method)
    data,report=detect_grid(image)
    if data is None:return None,report
    try:
        result=METHODS[method](image,data)
        return result,dict(report,method=method,model=result.info)
    except Exception as error:
        return None,{'status':'REJECTED','reason':f'{type(error).__name__}: {error}',
                     'detector':report}
