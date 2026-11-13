"""Independent, non-destructive curved-grid detector and horizontal-only prototype.

Use as a candidate generator BEFORE the existing contour/wave decision.  No OCR
model, external API, or production switch.  All geometry and source maps are
explicit; the stepped total/footer is never treated as a full-width row rule.
"""
from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np

VERSION = 'grid-curvature-v2.0-experiment'


@dataclass
class Rule:
    curve: np.ndarray
    coverage: float
    fit_error_px: float
    curvature_p95_px: float
    level_px: float
    bounds: tuple[int, int, int, int]


def _line_mask(image: np.ndarray, *, vertical: bool = False) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if vertical:
        gray = gray.T.copy()
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 5)
    n = max(15, min(29, int(gray.shape[1] * .026) // 2 * 2 + 1))
    center = n // 2
    mask = np.zeros_like(binary)
    for angle in range(-24, 25, 4):
        delta = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))]) * center
        kernel = np.zeros((n, n), np.uint8)
        cv2.line(kernel, tuple(np.rint(center-delta).astype(int)),
                 tuple(np.rint(center+delta).astype(int)), 1, 1)
        joined = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel[center-2:center+3,center-2:center+3])
        mask |= cv2.morphologyEx(joined, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((1, 7), np.uint8))
    return mask.T.copy() if vertical else mask


def _fit_robust(xs: np.ndarray, ys: np.ndarray, along: int) -> tuple[np.ndarray, float]:
    """Huber-IRLS quartic curve; ignore character blobs attached to printed lines."""
    u = 2*xs/max(1,along-1)-1
    degree = min(4, len(xs)-1)
    matrix = np.polynomial.legendre.legvander(u, degree)
    weights = np.ones(len(xs))
    penalty = np.diag([0.0, 0.0] + [0.004 * (n-1)**4 for n in range(2,degree+1)])
    for _ in range(9):
        co = np.linalg.solve(matrix.T@(weights[:,None]*matrix) + penalty + np.eye(degree+1)*1e-8,
                             matrix.T@(weights*ys))
        error = ys - matrix @ co
        med = float(np.median(error))
        scale = max(0.7, float(1.4826*np.median(np.abs(error-med))))
        weights = np.minimum(1.0, 1.5*scale/np.maximum(np.abs(error), 1e-6))
    full = np.polynomial.legendre.legval(np.linspace(-1,1,along),co)
    return full, float(np.median(np.abs(error)))


def extract_rules(image: np.ndarray, *, vertical: bool = False) -> tuple[list[Rule], np.ndarray]:
    mask = _line_mask(image, vertical=vertical)
    active = mask.T.copy() if vertical else mask
    height,width = active.shape
    count, labels, stats, _ = cv2.connectedComponentsWithStats(active, 8)
    rules=[]
    for label in range(1,count):
        x,y,w,h,pixels=(int(v) for v in stats[label])
        if (w<width*.69 or x>width*.15 or x+w<width*.85
                or h>height*.23 or pixels<w*.48):
            continue
        ys,xs = np.where(labels==label)
        columns=np.unique(xs)
        if len(columns)/width < .69:
            continue
        points=np.array([(col, np.median(ys[xs==col])) for col in columns[::2]],float)
        if len(points)<7:
            continue
        curve,fiterr=_fit_robust(points[:,0],points[:,1],width)
        if fiterr>max(5.0, height*.013):
            continue
        line=np.polyval(np.polyfit(np.arange(width),curve,1),np.arange(width))
        curvature=float(np.percentile(np.abs(curve-line),95))
        level=float(np.median(curve))
        if level<=4 or level>=height-4:
            continue
        rules.append(Rule(curve.astype(np.float32),len(columns)/width,fiterr,
                          curvature,level,(x,y,w,h)))
    rules.sort(key=lambda r:r.level_px)
    unique=[]
    for rule in rules:
        if unique and rule.level_px-unique[-1].level_px < 7:
            if rule.coverage>unique[-1].coverage:
                unique[-1]=rule
        else:unique.append(rule)
    return unique,mask


def _profile(rule: Rule) -> np.ndarray:
    x=np.linspace(-1,1,len(rule.curve))
    series=rule.curve-np.polyval(np.polyfit(x,rule.curve,1),x)
    return np.interp(np.linspace(-1,1,64),x,series)


def diagnose(image: np.ndarray) -> dict:
    """Analyse a cropped table ROI; nothing is changed or downloaded."""
    h,w=image.shape[:2]
    if min(h,w)<90:
        return dict(version=VERSION,decision='UNCERTAIN',reason='ROI слишком мала',rules=[],coherent_rules=0)
    rules, _ = extract_rules(image)
    vertical, _ = extract_rules(image,vertical=True)
    minimum=max(2.5, 0.006*w)
    curved=[r for r in rules if r.curvature_p95_px>=minimum]
    best=[]
    for a in curved:
        p=_profile(a)
        group=[]
        for b in curved:
            q=_profile(b)
            denom=float(np.linalg.norm(p)*np.linalg.norm(q))
            c=float(p@q/denom) if denom>1e-8 else 0
            if c>=.78:
                group.append(b)
        if len(group)>len(best):best=group
    stable=all(np.min(np.stack([rules[i+1].curve-rules[i].curve],axis=0))>7
               for i in range(len(rules)-1))
    coherent=len(best)>=3 and stable
    decision='DEWARP' if coherent else ('SKIP' if len(rules)>=3 else 'UNCERTAIN')
    data=[dict(y=round(r.level_px,2),coverage=round(r.coverage,3),
               fit_error_px=round(r.fit_error_px,2),curve_p95_px=round(r.curvature_p95_px,2),
               bounds=list(r.bounds)) for r in rules]
    return dict(version=VERSION,decision=decision,needs_dewarp=coherent,
                reason=('Три или больше согласованных изогнутых границы строк' if coherent else
                        'Нет доказанного согласованного изгиба нескольких строк'),
                horizontal_rules=len(rules),vertical_rules=len(vertical),
                curved_rules=len(curved),coherent_rules=len(best),
                curvature_threshold_px=round(minimum,2),stable_row_order=stable,rules=data)


def horizontal_candidate(image: np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray,dict]:
    """Experimental y-axis-only straightening (not a full 2D rectification).

    Returns candidate and backward source coordinate maps in *image* coordinates.
    No footer is included in full-width anchor detection. The footer extends by
    the displacement of the last full-width row boundary (without extrapolated
    scaling), thus does not get stretched into the VAT row.
    """
    decision=diagnose(image)
    if decision['decision']!='DEWARP':
        raise ValueError('Недостаточно доказательств необходимости выпрямления')
    rules,_=extract_rules(image)
    h,w=image.shape[:2]
    y_levels=np.asarray([r.level_px for r in rules],float)
    paths=np.stack([r.curve for r in rules])
    steps=np.diff(paths,axis=0)
    if np.min(steps)<7:
        raise ValueError('Обнаружены пересекающиеся линии')
    yy,xx=np.indices((h,w),dtype=np.float32)
    my=np.empty((h,w),np.float32)
    for x in range(w):
        # Piecewise row-geometry preserving: each printed separator is an anchor.
        dst=np.arange(h,dtype=float)
        src=np.interp(dst,y_levels,paths[:,x])
        before=dst<y_levels[0]
        after=dst>y_levels[-1]
        src[before]=dst[before]+(paths[0,x]-y_levels[0])
        src[after]=dst[after]+(paths[-1,x]-y_levels[-1])
        my[:,x]=src
    jac=np.gradient(my,axis=0)
    if np.min(jac)<.4 or np.max(jac)>2.2 or not np.isfinite(my).all():
        raise ValueError('Unsafe deformation Jacobian')
    rectified=cv2.remap(image,xx,my,cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_CONSTANT,borderValue=(255,255,255))
    after=diagnose(rectified)
    details=dict(before=decision,after=after,min_jacobian=float(jac.min()),
                 max_jacobian=float(jac.max()),method='experimental_horizontal_only',
                 NOTE='Candidate for controlled evaluation; not automatically selected in OCR')
    return rectified,xx,my,details



def two_axis_candidate(image: np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray,dict]:
    """Try vertical straighten after horizontal; fallback to horizontal when unsafe.

    A single backward 2D warp samples original pixels once.  Do not use in
    production until an OCR+arithmetic regression gate approves the result.
    """
    horizontal,identity_x,source_y,report=horizontal_candidate(image)
    vertical,_=extract_rules(horizontal,vertical=True)
    if len(vertical)!=5 or not all(r.coverage>=.68 for r in vertical):
        report['vertical_status']='skipped_unreliable_dividers'
        return horizontal,identity_x,source_y,report
    h,w=image.shape[:2]
    dest=np.asarray([r.level_px for r in vertical],float)
    paths=np.stack([r.curve for r in vertical])
    if np.min(np.diff(paths,axis=0)) < 8:
        report['vertical_status']='skipped_intersecting_dividers'
        return horizontal,identity_x,source_y,report
    rows,cols=np.indices((h,w),dtype=np.float32)
    map_x=np.empty((h,w),np.float32)
    for y in range(h):
        dst=np.arange(w,dtype=float)
        src=np.interp(dst,dest,paths[:,y])
        before=dst<dest[0]
        after=dst>dest[-1]
        src[before]=dst[before]+paths[0,y]-dest[0]
        src[after]=dst[after]+paths[-1,y]-dest[-1]
        map_x[y,:]=src
    # The preceding horizontal map evaluates to original x and y.
    map_y=cv2.remap(source_y,map_x,rows,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    dx_y,dx_x=np.gradient(map_x)
    dy_y,dy_x=np.gradient(map_y)
    jac=dx_x*dy_y-dx_y*dy_x
    if not np.isfinite(jac).all() or float(jac.min())<=.35 or float(jac.max())>=2.5:
        report['vertical_status']='skipped_unsafe_2d_jacobian'
        return horizontal,identity_x,source_y,report
    output=cv2.remap(image,map_x,map_y,cv2.INTER_CUBIC,
                     borderMode=cv2.BORDER_CONSTANT,borderValue=(255,255,255))
    report['vertical_status']='candidate_ready'
    report['quality_2d']=dict(min_jacobian=round(float(jac.min()),4),
                              max_jacobian=round(float(jac.max()),4))
    report['after_2d']=diagnose(output)
    return output,map_x,map_y,report

def rectify_candidate(image: np.ndarray, polygon, baseline_layout: dict,
                      baseline_map_x: np.ndarray, *, margin: int = 16):
    """Build a padded V2 candidate that fits the working OCR layout contract.

    The horizontal-only warp keeps source x unchanged, so the established
    contour column dividers are carried over through its forward map.  Row
    boundaries come from the straight rules measured after the V2 warp.  The
    candidate is padded by ``margin`` to match the existing rectification
    metadata.  Raises ``ValueError`` when the measured geometry is not safe.
    """
    polygon = np.asarray(polygon, np.float32)
    lo = np.maximum(0, np.floor(polygon.min(axis=0)).astype(int) - margin)
    hi = np.minimum([image.shape[1], image.shape[0]],
                    np.ceil(polygon.max(axis=0)).astype(int) + margin + 1)
    crop = image[lo[1]:hi[1], lo[0]:hi[0]]
    if min(crop.shape[:2]) < 90:
        raise ValueError('Слишком малая область таблицы для V2')
    candidate, mx, my, report = two_axis_candidate(crop)
    before = report.get('before', {})
    after = (report.get('after_2d') if report.get('vertical_status') == 'candidate_ready'
             else report.get('after')) or report.get('after') or {}
    if before.get('decision') != 'DEWARP':
        raise ValueError('V2 не подтвердил согласованный изгиб исходной таблицы')
    if after.get('decision') == 'DEWARP':
        raise ValueError('V2 не устранил изгиб таблицы')
    rules = sorted(float(rule['y']) for rule in after.get('rules', []))
    if len(rules) < 4:
        raise ValueError('V2 не нашёл достаточно горизонтальных границ строк')
    ys = [round(margin + y, 2) for y in rules]
    padded_height = crop.shape[0] + 2 * margin
    if any(not 0 < y < padded_height for y in ys) or np.min(np.diff(ys)) < 4:
        raise ValueError('Некорректные границы строк V2')
    try:
        middle = int(round((float(baseline_layout['body_top'])
                            + float(baseline_layout['body_bottom'])) / 2))
        middle = min(max(middle, 0), baseline_map_x.shape[0] - 1)
        base_xs = [float(baseline_layout[key]) for key in
                   ('body_left', 'name_left', 'name_right', 'quantity_right', 'right')]
        source_xs = [float(baseline_map_x[middle, int(round(x))]) for x in base_xs]
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ValueError('Не удалось перенести колонки базовой таблицы в V2') from error
    xs = [round(x - lo[0] + margin, 2) for x in source_xs]
    padded_width = crop.shape[1] + 2 * margin
    if (any(not margin - 2 <= x <= margin + crop.shape[1] + 2 for x in xs)
            or np.min(np.diff(xs)) < 4):
        raise ValueError('Некорректные границы колонок V2')
    padded = cv2.copyMakeBorder(candidate, margin, margin, margin, margin,
                                cv2.BORDER_CONSTANT, value=(255, 255, 255))
    # The white margin must also remap to white, so its source coordinates are
    # deliberately outside the photo instead of replicating the table border.
    map_x = cv2.copyMakeBorder(mx, margin, margin, margin, margin,
                               cv2.BORDER_CONSTANT, value=-1e4) + lo[0]
    map_y = cv2.copyMakeBorder(my, margin, margin, margin, margin,
                               cv2.BORDER_CONSTANT, value=-1e4) + lo[1]
    grid = dict(body_left=xs[0], name_left=xs[1], name_right=xs[2],
                quantity_right=xs[3], right=xs[4], body_top=ys[0],
                body_bottom=ys[-1], row_boundaries=ys, grid_confirmed=True)
    return padded, map_x.astype(np.float32), map_y.astype(np.float32), grid, report


def main() -> int:
    """Standalone offline probe; diagnostics stay in experiment_outputs/."""
    import argparse
    import json
    from pathlib import Path
    parser=argparse.ArgumentParser(description='Grid curvature detector / unselected dewarp candidate')
    parser.add_argument('photo',type=Path)
    parser.add_argument('--roi',type=int,nargs=4,metavar=('X0','Y0','X1','Y1'),
                        help='Observed table crop in source-photo coordinates')
    parser.add_argument('--output-dir',default='experiment_outputs/grid_curvature_v2/probe')
    args=parser.parse_args()
    root=Path.cwd()
    directory=(root/args.output_dir).resolve()
    allowed=(root/'experiment_outputs').resolve()
    if not directory.is_relative_to(allowed):
        parser.error('Отчёты разрешены только внутри experiment_outputs/')
    photo=cv2.imdecode(np.fromfile(str(args.photo),dtype=np.uint8),cv2.IMREAD_COLOR)
    if photo is None:
        parser.error(f'Не удалось прочитать {args.photo}')
    if args.roi:
        x0,y0,x1,y1=args.roi
        if not(0<=x0<x1<=photo.shape[1] and 0<=y0<y1<=photo.shape[0]):
            parser.error('Недопустимый ROI')
        crop=photo[y0:y1,x0:x1].copy()
    else:
        try:
            from App.ocr.table_boundary import detect_table_boundary
            boundary=detect_table_boundary(photo)
            polygon=boundary.get('polygon')
            if polygon is None:
                parser.error('Не обнаружена область таблицы; передайте --roi')
            pts=np.asarray(polygon,float)
            lo=np.maximum([0,0],np.floor(pts.min(axis=0)-16).astype(int))
            hi=np.minimum([photo.shape[1],photo.shape[0]],np.ceil(pts.max(axis=0)+17).astype(int))
            x0,y0=map(int,lo)
            x1,y1=map(int,hi)
            crop=photo[y0:y1,x0:x1].copy()
        except ImportError:
            parser.error('Без рабочего Product_Matcher передайте --roi')
    report=diagnose(crop)
    report['roi_source']=[x0,y0,x1,y1]
    directory.mkdir(parents=True,exist_ok=True)
    (directory/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    cv2.imencode('.png',crop)[1].tofile(str(directory/'before.png'))
    if report.get('needs_dewarp'):
        try:
            output,mx,my,details=horizontal_candidate(crop)
            cv2.imencode('.png',output)[1].tofile(str(directory/'candidate_horizontal.png'))
            # Coordinates are in ORIGINAL SOURCE photo, for OCR box back-projection.
            np.savez_compressed(directory/'candidate_map.npz',map_x=mx+x0,map_y=my+y0)
            two,tx,ty,two_report=two_axis_candidate(crop)
            if two_report['vertical_status']=='candidate_ready':
                cv2.imencode('.png',two)[1].tofile(str(directory/'candidate_2d.png'))
                np.savez_compressed(directory/'candidate_2d_map.npz',map_x=tx+x0,map_y=ty+y0)
            (directory/'candidate_report.json').write_text(
                json.dumps(two_report,ensure_ascii=False,indent=2),encoding='utf-8')
        except ValueError as error:
            (directory/'candidate_error.txt').write_text(str(error),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('decision','horizontal_rules','vertical_rules','coherent_rules')},
                     ensure_ascii=False))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
