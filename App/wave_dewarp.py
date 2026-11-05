"""Nonlinear receipt-table rectification from printed horizontal and vertical rules.

Build an ordered backward map, validate it, and sample original pixels once.
Only NumPy/OpenCV from the project environment are required.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

VERSION = "wave-grid-1.0"


@dataclass
class Rule:
    values: np.ndarray
    samples: np.ndarray
    coverage: float
    fit_error: float
    curvature: float

    def public(self):
        step = max(1, len(self.values)//60)
        return dict(curve=self.values[::step].tolist(), step=step, coverage=self.coverage,
                    fit_error_px=self.fit_error, curvature_px=self.curvature,
                    samples=self.samples[::max(1,len(self.samples)//150)].tolist())


def line_mask(image, vertical=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if vertical:
        gray = gray.T.copy()
    binary = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,5)
    # Local straight segments survive even when the complete rule is a curve.
    n = max(15,min(29,int(gray.shape[1]*.026)//2*2+1))
    center = n//2
    mask = np.zeros_like(binary)
    for angle in range(-24,25,4):
        delta = np.array([np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle))])*center
        kernel = np.zeros((n,n),np.uint8)
        cv2.line(kernel,tuple(np.rint(center-delta).astype(int)),tuple(np.rint(center+delta).astype(int)),1,1)
        short = kernel[center-2:center+3,center-2:center+3]
        joined = cv2.morphologyEx(binary,cv2.MORPH_CLOSE,short)
        mask |= cv2.morphologyEx(joined,cv2.MORPH_OPEN,kernel)
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((1,7),np.uint8))
    return mask.T.copy() if vertical else mask


def robust_curve(x, y, length):
    # Robust smooth polynomial, not a fit through every noisy dark pixel.
    degree = min(5,len(x)-1)
    xx = 2*x/max(length-1,1)-1
    design = np.polynomial.legendre.legvander(xx,degree)
    weights = np.ones(len(x))
    regularizer = np.diag([0.,0.]+[.002*(i-1)**4 for i in range(2,degree+1)])
    for _ in range(10):
        coefficients = np.linalg.solve(design.T@(weights[:,None]*design)+regularizer,
                                       design.T@(weights*y))
        residual = y-design@coefficients
        sigma = max(.7,1.4826*np.median(np.abs(residual-np.median(residual))))
        weights = np.minimum(1.,1.5*sigma/np.maximum(np.abs(residual),1e-6))
    values = np.polynomial.legendre.legval(np.linspace(-1,1,length),coefficients)
    return values, float(np.median(np.abs(residual)))


def find_rules(image, vertical=False):
    mask = line_mask(image,vertical)
    working = mask.T.copy() if vertical else mask
    height,width = working.shape
    count,labels,stats,_ = cv2.connectedComponentsWithStats(working,8)
    rules = []
    for index in range(1,count):
        x,y,w,h,pixels = stats[index]
        if w < width*.58 or h > height*.22 or pixels < w*.45:
            continue
        ys,xs = np.where(labels == index)
        unique = np.unique(xs)
        if len(unique)/width < .58:
            continue
        points = np.array([(column,np.median(ys[xs==column])) for column in unique[::2]],float)
        values,error = robust_curve(points[:,0],points[:,1],width)
        if error > max(2.5,height*.008):
            continue
        if np.median(values)<5 or np.median(values)>height-6:
            continue  # Crop/image-edge artifacts are not printed grid rules.
        linear = np.polyval(np.polyfit(points[:,0],values[points[:,0].astype(int)],1),points[:,0])
        curvature = float(np.percentile(np.abs(values[points[:,0].astype(int)]-linear),95))
        rules.append(Rule(values,points,len(unique)/width,error,curvature))
    rules.sort(key=lambda rule:np.median(rule.values))
    # Close duplicate lines are commonly the two sides of a thick printed rule.
    merged = []
    for rule in rules:
        if merged and np.median(rule.values-merged[-1].values) < 7:
            if rule.coverage > merged[-1].coverage:
                merged[-1] = rule
        else:
            merged.append(rule)
    return merged, mask


def ordered_axis_map(rules, length, along):
    curves = np.array([rule.values for rule in rules])
    if len(rules)<3 or curves.shape[1] != along:
        raise ValueError("Недостаточно длинных печатных линий для надёжной сетки")
    if np.min(np.diff(curves,axis=0)) < 7:
        raise ValueError("Опорные линии пересекаются или слишком близки")
    levels = np.median(curves,axis=1)
    destination = np.arange(length,dtype=float)
    result = np.empty((length,along),np.float32)
    for column in range(along):
        source = curves[:,column]
        values = np.interp(destination,levels,source)
        before,after = destination<levels[0],destination>levels[-1]
        # Continue the local map into the padding/total strip, never stretch the
        # stepped total-cell lower border over the entire table body.
        low = np.clip((source[1]-source[0])/(levels[1]-levels[0]),.5,2.)
        high = np.clip((source[-1]-source[-2])/(levels[-1]-levels[-2]),.5,2.)
        values[before] = source[0]+(destination[before]-levels[0])*low
        values[after] = source[-1]+(destination[after]-levels[-1])*high
        result[:,column] = values
    return result,levels


def map_quality(mx,my):
    dx_y,dx_x = np.gradient(mx)
    dy_y,dy_x = np.gradient(my)
    jacobian = dx_x*dy_y-dx_y*dy_x
    return dict(min_jacobian=float(np.min(jacobian)),max_jacobian=float(np.max(jacobian)),
                finite=bool(np.isfinite(mx).all() and np.isfinite(my).all()),
                positive=bool(np.min(jacobian)>.12))


def rectify(image, polygon, margin=24):
    """Return rectified image/map/diagnostics; unchanged crop on rejection.

    The polygon is only an ROI hint. Internal rules, not its convex hull, supply
    the nonlinear constraints. The total footer remains inside the source ROI.
    """
    points = np.asarray(polygon,float).reshape(-1,2)
    if len(points)<4 or not np.isfinite(points).all():
        raise ValueError("Invalid table polygon")
    lo = np.maximum(0,np.floor(points.min(axis=0)).astype(int)-margin)
    hi = np.minimum([image.shape[1],image.shape[0]],np.ceil(points.max(axis=0)).astype(int)+margin+1)
    crop = image[lo[1]:hi[1],lo[0]:hi[0]].copy()
    if min(crop.shape[:2])<50 or max(crop.shape[:2])>3000:
        raise ValueError("Unsupported table ROI size")
    height,width = crop.shape[:2]
    yy,xx = np.indices((height,width),dtype=np.float32)
    metadata = dict(version=VERSION,status="rejected",roi=[*lo.tolist(),*hi.tolist()],
                    warnings=[])
    horizontal,hmask = find_rules(crop)
    metadata["horizontal_rules"] = [rule.public() for rule in horizontal]
    diagnostics = dict(crop=crop,horizontal_mask=hmask)
    mx,my = xx.copy(),yy.copy()
    try:
        vy,levels_y = ordered_axis_map(horizontal,height,width)
        preview = cv2.remap(crop,xx,vy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
        vertical,vmask = find_rules(preview,vertical=True)
        metadata["vertical_rules"] = [rule.public() for rule in vertical]
        diagnostics.update(vertical_mask=vmask,vertical_preview=preview)
        hx,levels_x = ordered_axis_map(vertical,width,height)
        mx = hx.T.copy()
        my = cv2.remap(vy,mx,yy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
        quality = map_quality(mx,my)
        metadata.update(quality=quality,horizontal_levels=levels_y.tolist(),vertical_levels=levels_x.tolist(),
                        curvature_px=max([rule.curvature for rule in horizontal+vertical],default=0),
                        displacement_p95=float(np.percentile(np.hypot(mx-xx,my-yy),95)))
        if not quality["finite"] or not quality["positive"] or quality["max_jacobian"]>4:
            raise ValueError("Карта сгибается или чрезмерно растягивает пиксели")
        if metadata["displacement_p95"]<1.5:
            metadata["status"] = "no_change"
            mx,my = xx.copy(),yy.copy()
        else:
            metadata["status"] = "rectified"
    except (ValueError,np.linalg.LinAlgError) as error:
        metadata["warnings"].append(str(error))
        mx,my = xx.copy(),yy.copy()
    # Compose with the original ROI translation. Final RGB is sampled only once.
    mx,my = (mx+float(lo[0])).astype(np.float32),(my+float(lo[1])).astype(np.float32)
    if metadata["status"] in {"no_change","rejected"}:
        output = crop.copy()
    else:
        output = cv2.remap(image,mx,my,cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT,borderValue=(255,255,255))
        after_h,_ = find_rules(output)
        before_v,_ = find_rules(crop,vertical=True)
        after_v,_ = find_rules(output,vertical=True)
        checks = {}
        for axis,before,after in (("horizontal",horizontal,after_h),("vertical",before_v,after_v)):
            before_error = max([rule.curvature for rule in before],default=0)
            after_error = max([rule.curvature for rule in after],default=0)
            checks[axis] = dict(before_px=before_error,after_px=after_error,
                                before_count=len(before),after_count=len(after))
            if len(after)<3 or after_error>max(before_error+.7,before_error*1.12):
                metadata["status"] = "rejected_quality"
                metadata["warnings"].append(f"Повторная проверка линий не подтвердила улучшение: {axis}")
        metadata["straightness_check"] = checks
        if metadata["status"] == "rejected_quality":
            diagnostics["rejected_candidate"] = output
            output = crop.copy()
            mx,my = (xx+float(lo[0])).astype(np.float32),(yy+float(lo[1])).astype(np.float32)
    return output,mx,my,metadata,diagnostics
