"""Independent source-photo OCR of the printed grand total below the item grid.

Do not extend the table polygon, alter row segmentation, or derive digits from
computed totals. Only source-image pixels and observed polygon geometry may
select the footer. RapidOCR is provided by the existing application; installed
Tesseract is an optional *lazy* digit-only fallback (no model downloads).
"""
from __future__ import annotations

import re
import os
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

VERSION = "footer-total-recovery-2.0"
AMOUNT = re.compile(r"(?<![\d.,])(?:\d{1,3}(?:[ \u00a0\u202f.]\d{3})+|\d{1,8})[.,]\d{2}(?!\d)")


def amounts(text):
    """Parse *printed* decimal amounts; reject extra trailing digits."""
    matches=[]
    text=str(text or "")
    for m in AMOUNT.finditer(text):
        fragment=m.group(0).replace("\u00a0", " ").replace("\u202f", " ")
        last=max(fragment.rfind(","),fragment.rfind("."))
        if last<0:continue
        integer=re.sub(r"[ .]", "", fragment[:last])
        cents=fragment[last+1:]
        if integer.isdigit() and len(cents)==2:
            matches.append(dict(value=f"{int(integer)}.{cents}",raw=m.group(0)))
    return matches


def _validated_polygon(image, polygon):
    points=np.asarray(polygon, dtype=np.float64)
    if points.shape!=(4,2) or not np.isfinite(points).all():
        return None, 'invalid_polygon'
    h,w=image.shape[:2]
    xs,ys=points[:,0],points[:,1]
    if (np.min(xs)<-w*.1 or np.max(xs)>w*1.1 or np.min(ys)<-h*.1 or
            np.max(ys)>h*1.1 or np.ptp(xs)<80 or np.ptp(ys)<160):
        return None, 'invalid_polygon_extent'
    return points, None


def footer_search_roi(image, polygon):
    """A separate *source photo* search window; never extends the item grid.

    Derived from the detected right edge and bottom line. The width and local
    table geometry determine a bounded search margin; not a fixed 80 px crop.
    """
    points,error=_validated_polygon(image,polygon)
    if error:return None,dict(status='rejected',reason=error)
    h,w=image.shape[:2]
    tl,tr,br,bl=points
    width=float(np.mean((np.linalg.norm(tr-tl),np.linalg.norm(br-bl))))
    source_bottom_min=float(min(br[1],bl[1]))
    source_bottom_max=float(max(br[1],bl[1]))
    right=max(tr[0],br[0]);left=min(tl[0],bl[0])
    x0=int(max(0,right-.38*width))
    x1=int(min(w,right+.10*width))
    y0=int(max(0,source_bottom_min-.05*width))
    y1=int(min(h,source_bottom_max+.19*width))
    if x1-x0<70 or y1-y0<32:
        return None,dict(status='rejected',reason='footer_roi_outside_photo')
    return (x0,y0,x1,y1),dict(status='roi_found',roi=[x0,y0,x1,y1],
        bottom_limit_px=round(source_bottom_min,2),width_px=round(width,2))


def find_printed_amount_crop(image, polygon):
    """Detect digit glyphs below the observed lower grid line, excluding ₸.

    This localizes the final printed number, without OCR/model and without
    looking at expected amounts. A missing/ambiguous glyph group is rejected.
    """
    roi,report=footer_search_roi(image,polygon)
    if roi is None:return None,report
    x0,y0,x1,y1=roi
    gray=cv2.cvtColor(image[y0:y1,x0:x1],cv2.COLOR_BGR2GRAY)
    block=31 if min(gray.shape)>=31 else max(3,min(gray.shape)//2*2-1)
    binary=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV,block,11)
    hk=cv2.getStructuringElement(cv2.MORPH_RECT,(max(20,int(gray.shape[1]*.3)),1))
    vk=cv2.getStructuringElement(cv2.MORPH_RECT,(1,max(15,int(gray.shape[0]*.2))))
    ruling=cv2.bitwise_or(cv2.morphologyEx(binary,cv2.MORPH_OPEN,hk),
                          cv2.morphologyEx(binary,cv2.MORPH_OPEN,vk))
    ink=cv2.bitwise_and(binary,cv2.bitwise_not(ruling))
    n,_,stats,centers=cv2.connectedComponentsWithStats(ink,8)
    points=np.asarray(polygon,dtype=float)
    min_center_y=min(points[2,1],points[3,1])+3
    glyphs=[]
    for (x,y,ww,hh,area),(_,cy) in zip(stats[1:],centers[1:]):
        if (4<=area<=max(600,int(gray.size*.025)) and
                2<=ww<=max(42,int(gray.shape[1]*.22)) and
                8<=hh<=max(38,int(gray.shape[0]*.33)) and y0+cy>=min_center_y):
            glyphs.append((int(x),int(y),int(ww),int(hh),float(cy)))
    groups=[]
    for item in sorted(glyphs,key=lambda a:a[-1]):
        if not groups or abs(item[-1]-np.median([j[-1] for j in groups[-1]]))>15:
            groups.append([item])
        else:groups[-1].append(item)
    # Select a visually continuous, numeral-sized line rather than guessing
    # from product arithmetic. Lower than the table's final detected line.
    groups=[g for g in groups if len(g)>=5 and
            max(x+ww for x,y,ww,hh,c in g)-min(x for x,y,ww,hh,c in g)>=40]
    if not groups:
        return None,dict(report,status='no_footer_number_glyphs')
    selected=max(groups,key=lambda g:np.median([j[-1] for j in g]))
    selected.sort(key=lambda a:a[0])
    # A detached currency glyph normally lies to the right of the number.
    # Only strip it if the gap is visibly wider than the typical inter-digit gap.
    if len(selected)>=7:
        gaps=[selected[i+1][0]-(selected[i][0]+selected[i][2]) for i in range(len(selected)-1)]
        positives=sorted(max(0,g) for g in gaps[:-1])
        typical=positives[len(positives)//2] if positives else 1
        if gaps[-1] >= max(7,typical+5):selected=selected[:-1]
    xmin=min(g[0] for g in selected)
    xmax=max(g[0]+g[2] for g in selected)
    ymin=min(g[1] for g in selected)
    ymax=max(g[1]+g[3] for g in selected)
    cx0=max(0,x0+xmin-5);cx1=min(image.shape[1],x0+xmax+4)
    cy0=max(0,y0+ymin-6);cy1=min(image.shape[0],y0+ymax+6)
    if cx1-cx0<35 or cy1-cy0<15:
        return None,dict(report,status='invalid_amount_crop')
    return (cx0,cy0,cx1,cy1),dict(report,status='amount_crop_found',
                amount_crop=[cx0,cy0,cx1,cy1],glyph_count=len(selected))


def _rapid_read(crop, recognize):
    if recognize is None:return [],'not_connected'
    try:
        output=recognize(crop)
        if isinstance(output,tuple) and len(output)==2:output=output[0]
        if output is None:return [],'no_lines'
        lines=[]
        for line in output:
            if not isinstance(line,(tuple,list)) or len(line)!=3:continue
            box,text,confidence=line
            try:confidence=float(confidence)
            except (ValueError,TypeError):continue
            lines.append(dict(text=str(text),confidence=confidence))
        return lines,'ok'
    except Exception as error:
        return [],'ocr_error:'+type(error).__name__


def find_tesseract_exe(root=None):
    """Find an existing Tesseract binary; never install/download a model."""
    from dotenv import dotenv_values
    root=Path(root) if root is not None else Path(__file__).resolve().parents[2]
    cfg=dotenv_values(root/'.env',encoding='utf-8-sig')
    configured=str(cfg.get('OCR_TESSERACT_PATH') or '').strip().strip('"')
    if configured:
        candidate=Path(configured)
        if candidate.is_file():return str(candidate)
        if shutil.which(configured):return configured
        return None
    found=shutil.which('tesseract')
    if found:return found
    for home in (os.environ.get('ProgramFiles'),os.environ.get('ProgramFiles(x86)'),
                 os.environ.get('LOCALAPPDATA')):
        if home:
            candidate=Path(home)/'Tesseract-OCR'/'tesseract.exe'
            if candidate.is_file():return str(candidate)
    return None


def _tesseract_readings(crop,exe=None, timeout=12):
    exe=exe or find_tesseract_exe()
    if exe is None:return [],'not_installed'
    # No pip install, no download, no OCR model loaded unless fallback reached.
    import tempfile
    with tempfile.TemporaryDirectory(prefix='footer_total_') as dirname:
        path=Path(dirname)/'total.png'
        if not cv2.imwrite(str(path),crop):return [],'save_failed'
        results=[]
        for psm in (7,13):
            try:
                run=subprocess.run([exe,str(path),'stdout','--psm',str(psm),'-l','eng'],
                    check=False,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                    timeout=timeout)
                text=run.stdout.decode('utf-8',errors='replace').strip()
                if run.returncode!=0:return results,'ocr_failed'
                results.append(dict(psm=psm,text=text,values=amounts(text)))
            except (OSError,subprocess.TimeoutExpired):return results,'ocr_unavailable'
    return results,'ok'


def recover_footer_total(image, polygon, *, rapidocr=None, tesseract=True, save_crop=None):
    """Return recognized printed total or None, together with evidence.

    Never accepts a computed total argument and never modifies item-grid pixels.
    Two agreeing independent Tesseract layout reads are required if RapidOCR
    cannot give a sufficiently confident value. Conflicts always reject.
    """
    bounds,report=find_printed_amount_crop(image,polygon)
    if bounds is None:return None,report
    x0,y0,x1,y1=bounds
    crop=image[y0:y1,x0:x1].copy()
    if save_crop is not None:
        Path(save_crop).parent.mkdir(parents=True,exist_ok=True)
        cv2.imwrite(str(save_crop),crop)
    readings,status=_rapid_read(crop,rapidocr)
    report.update(rapidocr=dict(status=status,readings=readings))
    rapid_candidates=[dict(value=m['value'],confidence=l['confidence'],raw=m['raw'])
                     for l in readings for m in amounts(l['text']) if l['confidence']>=.50]
    unique={m['value'] for m in rapid_candidates}
    if len(unique)==1 and max(m['confidence'] for m in rapid_candidates)>=.60:
        value=next(iter(unique))
        return value,dict(report,status='confirmed',source='rapidocr_source_footer',printed_value=value)
    if not tesseract:return None,dict(report,status='unconfirmed',reason='rapidocr_uncertain_tesseract_disabled')
    ts,ts_status=_tesseract_readings(crop)
    report['tesseract']=dict(status=ts_status,readings=ts)
    if ts_status!='ok':return None,dict(report,status='unconfirmed',reason='tesseract_'+ts_status)
    tvals=[{m['value'] for m in row['values']} for row in ts]
    if len(tvals)!=2 or any(len(x)!=1 for x in tvals) or tvals[0]!=tvals[1]:
        return None,dict(report,status='unconfirmed',reason='tesseract_disagreement')
    value=next(iter(tvals[0]))
    if unique and unique!={value}:
        return None,dict(report,status='unconfirmed',reason='ocr_engine_disagreement')
    return value,dict(report,status='confirmed',source='tesseract_source_footer',printed_value=value)


def setting_mode(root=None):
    """Read only the existing root .env. The default follows 2D mode."""
    from dotenv import dotenv_values
    root=Path(root) if root is not None else Path(__file__).resolve().parents[2]
    settings=dotenv_values(root/'.env',encoding='utf-8-sig')
    value=str(settings.get('OCR_FOOTER_TOTAL_MODE') or 'auto').lower().strip()
    if value not in {'auto','shadow','select','off'}:
        raise ValueError('OCR_FOOTER_TOTAL_MODE must be auto, shadow, select or off')
    if value!='auto':return value
    geom=str(settings.get('OCR_PERSPECTIVE_2D_MODE') or 'shadow').lower().strip()
    return geom if geom in {'off', 'shadow', 'select'} else 'shadow'


def maybe_recover_footer_total(image,polygon, current_total, *, rapidocr=None,
                                output_dir=None, mode_override=None):
    """Adapter: return old/new *printed* total and JSON-safe audit.

    Shadow may run diagnostics but never changes the total. The item rows and
    shape of the rectified image are not available to, and cannot be modified
    by, this function.
    """
    mode=mode_override or setting_mode()
    if mode not in {'off','shadow','select'}:raise ValueError('invalid_footer_mode')
    audit=dict(version=VERSION,mode=mode,status='not_needed',selected=False)
    if mode=='off':return current_total,dict(audit,status='disabled')
    if amounts(current_total):
        return current_total,dict(audit,reason='printed_total_already_read')
    filename=(Path(output_dir)/'footer_total_crop.png') if output_dir is not None else None
    detected,report=recover_footer_total(image,polygon,rapidocr=rapidocr,
                                        tesseract=True,save_crop=filename)
    audit.update(status=report['status'],evidence=report,printed_candidate=detected)
    if detected is None:return current_total,audit
    if mode=='shadow':return current_total,dict(audit,status='shadow_confirmed',reason='diagnostic_only')
    return detected,dict(audit,status='selected',selected=True,
                         reason='independently_read_source_photo_total')
