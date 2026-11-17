"""Independent line-detector adapters. No source writes, no calls to OCR engines.

Each result contains physical candidate segments/polylines in processed-image
pixel coordinates, NOT inferred document row cells. Missing evidence is not zero error.
"""
from __future__ import annotations
import importlib
import hashlib
import inspect
import math
import sys
from pathlib import Path
from typing import Any
import cv2
import numpy as np

GEOMETRY_METHODS = ('baseline','perspective','grid_mesh','tps_scipy')
METHODS = GEOMETRY_METHODS + ('physical_grid_current', 'opencv_lsd', 'edge_drawing', 'img2table', 'frangi', 'sato')

class DependencyMissing(RuntimeError):
    pass

class AlgorithmUnavailable(RuntimeError):
    pass

V3_VERSION = 'Physical Grid Graph V3 / frozen final candidate'
V3_SHA256 = 'f04017f73e009cc82862a55178fb3715b1c42f246190800c44aa5e7eedd43548'

def resolve_v3(lab_path):
    """Fail closed: never silently substitute the legacy grid detector."""
    lab_path = Path(lab_path).resolve()
    expected = lab_path / 'geometry_lab/physical_grid_graph_v3.py'
    if not expected.is_file():
        raise AlgorithmUnavailable(f'Actual V3 source unavailable: {expected}')
    sys.path.insert(0, str(lab_path))
    try:
        module = importlib.import_module('geometry_lab.physical_grid_graph_v3')
        function = module.detect_physical_graph
        actual = Path(inspect.getsourcefile(function)).resolve()
        digest = hashlib.sha256(actual.read_bytes()).hexdigest()
        if (actual != expected or Path(module.__file__).resolve() != expected or
                function.__name__ != 'detect_physical_graph' or digest != V3_SHA256):
            raise AlgorithmUnavailable('Actual V3 identity differs from the frozen audited implementation')
        return function, dict(module_path=str(actual), function=function.__name__,
                              sha256=digest, version=V3_VERSION,
                              backend='geometry_lab.physical_grid_graph_v3.detect_physical_graph')
    except (ImportError, AttributeError, OSError, TypeError) as exc:
        raise AlgorithmUnavailable(f'Actual V3 cannot be loaded: {exc}') from exc
    finally:
        sys.path.pop(0)


def gray_image(bgr: np.ndarray) -> np.ndarray:
    if bgr.ndim == 2:
        return bgr
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def _segment(x0, y0, x1, y1, shape, min_fraction=0.045) -> dict | None:
    h, w = shape[:2]
    if not all(math.isfinite(float(q)) for q in (x0,y0,x1,y1)):
        return None
    length=math.hypot(float(x1-x0),float(y1-y0))
    dx=abs(x1-x0);dy=abs(y1-y0)
    if dx > 2*dy:
        axis='h'; threshold=max(10,w*min_fraction)
    elif dy > 2*dx:
        axis='v'; threshold=max(10,h*min_fraction)
    else:
        return None
    if length < threshold:
        return None
    pts = [[float(np.clip(x0,0,w-1)), float(np.clip(y0,0,h-1))],
           [float(np.clip(x1,0,w-1)), float(np.clip(y1,0,h-1))]]
    if pts[0] == pts[1]: return None
    return {'axis':axis,'points':pts,'length_px':round(length,2),'source':'observed_segment'}


def _format_lines(lines, shape, min_fraction=.045):
    if lines is None:
        return []
    a=np.asarray(lines, dtype=np.float64)
    if not a.size:return []
    a=a.reshape(-1,4)
    out=[]
    for x0,y0,x1,y1 in a:
        item=_segment(x0,y0,x1,y1,shape,min_fraction)
        if item is not None:out.append(item)
    return out


def opencv_lsd(img):
    if not hasattr(cv2,'createLineSegmentDetector'):
        raise DependencyMissing('cv2.createLineSegmentDetector unavailable')
    g=gray_image(img)
    lsd=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    found=lsd.detect(g)[0]
    return _format_lines(found,g.shape),{'backend':'cv2.createLineSegmentDetector'}


def edge_drawing(img):
    if not hasattr(cv2,'ximgproc') or not hasattr(cv2.ximgproc,'createEdgeDrawing'):
        raise DependencyMissing('cv2.ximgproc.createEdgeDrawing missing; use isolated opencv-contrib-python install')
    detector=cv2.ximgproc.createEdgeDrawing()
    detector.detectEdges(gray_image(img))
    found=detector.detectLines()
    return _format_lines(found,img.shape),{'backend':'cv2.ximgproc.createEdgeDrawing'}


def img2table_lines(img):
    """Use its line-finding algorithm WITHOUT OCR to avoid losing empty cells.

    The internal API is version-sensitive; catch and surface incompatibilities.
    This IS NOT the public image.extract_tables OCR/table recognizer.
    """
    try:
        mod=importlib.import_module('img2table.tables.bordered.lines')
    except (ImportError,ModuleNotFoundError):
        # img2table 1.4.x moved the internal bordered-lines module.
        try:
            mod=importlib.import_module('img2table.tables.processing.bordered_tables.lines')
        except (ImportError,ModuleNotFoundError) as exc:
            raise DependencyMissing(f'img2table line API unavailable: {exc}') from exc
    rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    h,w=img.shape[:2]
    min_line=max(24,int(min(h,w)*.07))
    horizontal, vertical=mod.detect_lines(img=rgb,contours=[],char_length=10.,min_line_length=min_line)
    all_found=[]
    for l in list(horizontal)+list(vertical):
        s=_segment(l.x1,l.y1,l.x2,l.y2,img.shape,min_fraction=.035)
        if s:all_found.append(s)
    return all_found, {'backend':'img2table.tables.bordered.lines.detect_lines','api':'internal, version-sensitive',
                       'note':'OCR not run; line detector only'}


def ridge_lines(img, name):
    try:
        from skimage.filters import frangi, sato
    except (ModuleNotFoundError,ImportError) as exc:
        raise DependencyMissing(f'scikit-image not installed: {exc}') from exc
    gray=gray_image(img)
    norm=np.asarray(gray,np.float32)/255.
    response=(frangi if name=='frangi' else sato)(norm,sigmas=(1,2,3),black_ridges=True)
    response=np.nan_to_num(response,nan=0.,posinf=0.,neginf=0.)
    nonzero=response[response>0]
    if not nonzero.size:
        return [], {'backend':name,'threshold':None,'reason':'no_positive_ridges'}
    threshold=float(np.percentile(nonzero,88))
    if threshold<=0:
        return [], {'backend':name,'threshold':threshold,'reason':'invalid_response_threshold'}
    mask=((response>=threshold)&(response>0)).astype(np.uint8)*255
    h,w=mask.shape
    # Multi orientation: short strokes survive even if a long rule is curved.
    found=cv2.HoughLinesP(mask,1,np.pi/720,threshold=max(10,int(min(h,w)*.012)),
                           minLineLength=max(22,int(min(h,w)*.05)),
                           maxLineGap=max(6,int(min(h,w)*.012)))
    return _format_lines(found,mask.shape,min_fraction=.04),{
        'backend':f'skimage.filters.{name}','threshold':threshold,'sigmas':[1,2,3],
        'note':'Ridge response includes text; not confirmed table boundaries'}


def physical_grid_current(img, lab_path: Path):
    function, identity = resolve_v3(lab_path)
    grid, info = function(img)
    info = dict(info, **identity)
    if grid is None:
        return [], info
    h,w=img.shape[:2]
    rules=[]
    for axis, group in [('v',grid.get('vertical',[])),('h',grid.get('horizontal',[]))]:
        for r in group:
            start,end=r.domain
            if axis=='v':
                y=np.linspace(max(0,start),min(h-1,end),max(3,int((end-start)/12)))
                xs=np.asarray(r.eval((y-h/2)/h))
                poly=np.column_stack([xs,y])
            else:
                x=np.linspace(max(0,start),min(w-1,end),max(3,int((end-start)/12)))
                ys=np.asarray(r.eval((x-w/2)/w))
                poly=np.column_stack([x,ys])
            if not np.isfinite(poly).all():continue
            poly[:,0]=poly[:,0].clip(0,w-1);poly[:,1]=poly[:,1].clip(0,h-1)
            if np.linalg.norm(poly[-1]-poly[0])<12:continue
            rules.append({'axis':axis,'points':poly.round(2).tolist(),'source':'fitted_physical_rule',
                          'support':round(float(r.support),3)})
    return rules, info



def geometry_current(method: str, img: np.ndarray, lab_path: Path, job_dir: Path, job_id: str, sha: str):
    """Geometry methods from current V3, drawn in SAME oriented-original coords.

    These are *proposed grid lines*, not independent physical observations.
    """
    lab_path=Path(lab_path).resolve()
    if not (lab_path/'geometry_lab'/'algorithms.py').is_file():
        raise DependencyMissing('V3 geometry_lab/algorithms.py missing')
    sys.path.insert(0,str(lab_path))
    try:
        from geometry_lab.metrics import projected_lines
        if method=='baseline':
            from geometry_lab.cases import Case, load_baseline
            case=Case(job_id,Path(job_dir),Path(job_dir)/'original.png',sha)
            result,reason=load_baseline(case)
        else:
            from geometry_lab.algorithms import METHODS as SOLVERS
            function, identity = resolve_v3(lab_path)
            controls, report = function(img)
            report = dict(report, **identity)
            result = None
            if controls is not None:
                try:
                    result = SOLVERS[method](img, controls)
                except Exception as exc:
                    report = dict(report, reason=f'{type(exc).__name__}: {exc}')
            reason=report.get('reason') if result is None else None
        if result is None:
            return [], {'status':'NOT_AVAILABLE' if method=='baseline' else 'REJECTED',
                        'reason':str(reason) if reason else 'no_model_result', 'model':method}, None
        vertical,horizontal=projected_lines(result,n=80)
        records=[]
        for axis,grp in [('v',vertical),('h',horizontal)]:
            for q in grp:
                arr=np.asarray(q,dtype=float)
                if arr.shape!=(80,2) or not np.isfinite(arr).all():continue
                records.append({'axis':axis,'points':arr.round(3).tolist(),
                                'source':'projected_model_grid'})
        return records, {'status':'MODEL_PROJECTED','model':method,
                         **(identity if method != 'baseline' else {}),
                         'note':'Proposed grid from V3 geometry, NOT an independent observed line detector'},result.image
    finally:
        sys.path.pop(0)


def detect(method: str, img: np.ndarray, lab_path: Path):
    cv2.setNumThreads(1)
    if hasattr(cv2,'ocl'):cv2.ocl.setUseOpenCL(False)
    if method=='opencv_lsd':return opencv_lsd(img)
    if method=='edge_drawing':return edge_drawing(img)
    if method=='img2table':return img2table_lines(img)
    if method in ('frangi','sato'):return ridge_lines(img,method)
    if method=='physical_grid_current':return physical_grid_current(img,lab_path)
    raise ValueError(f'Unknown method: {method}')
