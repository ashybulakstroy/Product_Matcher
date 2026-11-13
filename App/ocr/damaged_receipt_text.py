"""Evidence-only detector for narrow blank tears across the printed name column.

No generative reconstruction, no catalog substitutions, no numeric edits.
Only enables an existing selective text OCR retry if printed evidence is strong.
"""
from __future__ import annotations
import cv2
import numpy as np

VERSION='damaged-receipt-text-1.0'


def detect_damage_stripe(image, polygon, columns):
    report=dict(version=VERSION,status='not_confirmed',stripes=[],forced_ocr=False)
    if polygon is None or len(columns)!=5 or image is None:
        report['reason']='no_verified_grid';return report
    h,w=image.shape[:2]
    polygon=np.asarray(polygon, dtype=np.float32)
    cols=np.asarray(columns,dtype=float)
    if not np.all(np.isfinite(cols)) or not np.all(np.isfinite(polygon)):
        report['reason']='invalid_geometry';return report
    name_left,name_right=cols[1],cols[2]
    if name_right-name_left<80:
        report['reason']='name_column_too_small';return report
    top=float(np.mean(polygon[:2,1]));bottom=float(np.mean(polygon[2:,1]))
    span=bottom-top
    y0=int(np.clip(top+span*.20,0,h-1));y1=int(np.clip(bottom-span*.02,y0+1,h))
    x0=int(np.clip(name_left+(name_right-name_left)*.22,0,w-1))
    x1=int(np.clip(name_right-(name_right-name_left)*.14,x0+1,w))
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
    area=gray[y0:y1,x0:x1]
    if area.size<400:
        report['reason']='small_interior';return report
    ink=(area<min(168., float(np.median(area))*.75)).mean(axis=0)
    blank=ink<.012
    chunks=[]
    for x in np.flatnonzero(blank):
        if not chunks or x>chunks[-1][-1]+1:chunks.append([int(x)])
        else:chunks[-1].append(int(x))
    for chunk in chunks:
        start,end=chunk[0],chunk[-1]+1
        length=end-start
        if not (max(8,int((name_right-name_left)*.022))<=length<=max(12,int((name_right-name_left)*.085))):
            continue
        left=ink[max(0,start-35):max(0,start-8)]
        right=ink[min(len(ink),end+8):min(len(ink),end+35)]
        if not len(left) or not len(right) or np.mean(left)<.025 or np.mean(right)<.025:
            continue
        report['stripes'].append(dict(x0=x0+start,x1=x0+end,y0=y0,y1=y1,
                                     ink_fraction=round(float(np.mean(ink[start:end])),5),
                                     left_ink=round(float(np.mean(left)),4),right_ink=round(float(np.mean(right)),4)))
    if report['stripes']:
        report.update(status='confirmed_bright_gap',forced_ocr=True)
    else:
        report['reason']='no_persistent_blank_band_with_text_on_both_sides'
    return report


def affected_name_rows(rows, layout, map_y, damage_report, limit=3):
    """Select only physically overlapping product rows, never VAT or prices.

    map_y is the already-existing rectified->source coordinate mapping.
    """
    if damage_report.get('status')!='confirmed_bright_gap' or map_y is None:
        return []
    h,w=map_y.shape[:2]
    middle_x=int(np.clip(round((layout['name_left']+layout['name_right'])/2),0,w-1))
    chosen=[]
    for row in rows:
        if row.get('kind')!='product' or len(row.get('y_range',[]))!=2:continue
        top,bottom=row['y_range'];mid=int(np.clip((top+bottom)/2,0,h-1))
        original_y=float(map_y[mid,middle_x])
        if any(s['y0']<=original_y<=s['y1'] for s in damage_report['stripes']):
            chosen.append(int(row['row']))
        if len(chosen)>=limit:break
    return chosen
