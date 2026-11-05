"""Read missing receipt quantities directly from bounded image cells.

Preserve raw OCR. Accept supplemental values only when two views agree and the
printed total and VAT validate. Uses the existing recognizer from the pipeline.
"""
from __future__ import annotations

import copy
import math
import re

import cv2
import numpy as np

from App.receipt_editing import parse_number, update_receipt

VERSION = "quantity-cell-1.0"


def quantity_crop(image,row,layout):
    if row.get("kind") not in {"product","service","tax"}:
        return None
    if str(row.get("quantity",row.get("quantity_raw","")) or "").strip():
        return None
    if parse_number(row.get("price",row.get("price_raw"))) is None:
        return None
    anchors = []
    for line in row.get("raw_lines",[]):
        box = np.asarray(line["box"],float)
        if box.mean(axis=0)[0]>=layout["quantity_right"] and re.search(r"\d",line["text"]):
            anchors.append(box)
    if not anchors:
        return None
    top,bottom = row["y_range"]
    x0 = max(0,math.ceil(layout["name_right"])+3)
    x1 = min(image.shape[1],math.floor(layout["quantity_right"])-3)
    y0 = max(0,math.ceil(top)+3,math.floor(min(box[:,1].min() for box in anchors))-5)
    y1 = min(image.shape[0],math.floor(bottom)-3,math.ceil(max(box[:,1].max() for box in anchors))+5)
    if x1-x0<8 or y1-y0<8:
        return None
    crop = image[y0:y1,x0:x1].copy()
    gray = cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
    if int(gray.max())-int(gray.min())<24:
        return None
    _,mask = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    _,_,stats,_ = cv2.connectedComponentsWithStats(mask,8)
    components = [stat for stat in stats[1:] if stat[4]>=4 and stat[3]>=max(5,crop.shape[0]*.2)
                  and not (stat[3]>crop.shape[0]*.8 and stat[2]<3)
                  and stat[2]<crop.shape[1]*.9]
    if not components:
        return None
    left = max(0,min(stat[0] for stat in components)-3)
    right = min(crop.shape[1],max(stat[0]+stat[2] for stat in components)+3)
    upper = max(0,min(stat[1] for stat in components)-3)
    lower = min(crop.shape[0],max(stat[1]+stat[3] for stat in components)+3)
    return dict(row=row["row"],bounds=[int(v) for v in (x0,y0,x1,y1)],tight_bounds=[int(v) for v in (x0+left,y0+upper,x0+right,y0+lower)],
                crop=crop,tight=crop[upper:lower,left:right].copy())


def recover_quantities(receipt,image,layout,recognize):
    result = copy.deepcopy(receipt)
    proposals = []
    crops = []
    for row in result["rows"]:
        candidate = quantity_crop(image,row,layout)
        if candidate:
            crops.append(candidate)
    if not crops:
        return result,dict(version=VERSION,attempts=[],applied=False),[]
    images = [view for crop in crops for view in (crop["crop"],crop["tight"])]
    recognized = recognize(images)
    if len(recognized)!=len(images):
        raise ValueError("Numeric cell OCR returned an unexpected result count")
    for index,crop in enumerate(crops):
        views = [dict(text=str(text),confidence=float(confidence)) for text,confidence in recognized[index*2:index*2+2]]
        numbers = [re.sub(r"\s+","",view["text"]) for view in views]
        valid = all(re.fullmatch(r"\d+(?:[.,]\d+)?",text) and parse_number(text) is not None for text in numbers)
        agreed = valid and parse_number(numbers[0])==parse_number(numbers[1]) and min(view["confidence"] for view in views)>=.35
        attempt = {key:crop[key] for key in ("row","bounds","tight_bounds")}
        attempt.update(views=views,agreed=bool(agreed))
        proposals.append(attempt)
        if agreed:
            row = next(row for row in result["rows"] if row["row"]==crop["row"])
            row["quantity"] = numbers[0]
            row["quantity_source"] = "cell_ocr"
            row["quantity_ocr"] = attempt
    update_receipt(result)
    applied = any(attempt["agreed"] for attempt in proposals) and result["calculation"].get("validation_status",result["calculation"]["status"])=="match"
    record = dict(version=VERSION,attempts=proposals,applied=applied,
                  proposed_status=result["calculation"]["status"],
                  proposed_validation_status=result["calculation"].get("validation_status",result["calculation"]["status"]),
                  vat_status=result["calculation"].get("vat",{}).get("status"),
                  gate="two OCR views agree, printed total and VAT match")
    return result if applied else copy.deepcopy(receipt),record,crops
