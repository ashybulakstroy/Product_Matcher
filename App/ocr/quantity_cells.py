"""Read missing receipt quantities directly from bounded image cells.

Preserve raw OCR. The ordinary recovery requires agreement and total/VAT validation.
A clipped single-digit 1 may be retained provisionally after multiple OCR views agree;
it remains unverified until the printed total and VAT validate.
"""
from __future__ import annotations

import copy
import math
import re

import cv2
import numpy as np

from App.ocr.receipt_editing import parse_number, update_receipt

VERSION = "quantity-cell-1.1"


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
    # Narrow fallback for a digit clipped at the right edge of its quantity cell.
    # Only a supported OCR result of "1" is eligible; never infer from arithmetic.
    edge_candidates = [(crop, attempt) for crop, attempt in zip(crops, proposals)
                       if not attempt['agreed'] and crop['tight_bounds'][2] >= crop['bounds'][2]]
    edge_images = []
    for crop, _ in edge_candidates:
        x0, y0, x1, y1 = crop['bounds']
        tx0, ty0, _, ty1 = crop['tight_bounds']
        right = min(image.shape[1], math.floor(layout['quantity_right'])-1, x1+2)
        fragment = image[ty0:ty1, max(x0, tx0-3):right]
        if fragment.size == 0:
            edge_images.extend([crop['tight']]*3)
            continue
        gray = cv2.cvtColor(fragment, cv2.COLOR_BGR2GRAY)
        binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        edge_images.extend([cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR),
                            cv2.cvtColor(binary,cv2.COLOR_GRAY2BGR),
                            cv2.resize(cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR),None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)])
    if edge_images:
        edge_readings = recognize(edge_images)
        if len(edge_readings) != len(edge_images):
            raise ValueError('Numeric cell OCR returned an unexpected result count')
        for index,(crop,attempt) in enumerate(edge_candidates):
            views = [dict(text=str(text).strip(),confidence=float(confidence))
                     for text,confidence in edge_readings[index*3:index*3+3]]
            votes = sum(view['text']=='1' and view['confidence']>=.35 for view in views)
            attempt['edge_views'] = views
            attempt['edge_agreed'] = votes >= 2
            if votes >= 2:
                row = next(row for row in result['rows'] if row['row']==crop['row'])
                row['quantity'] = '1'
                row['quantity_source'] = 'cell_ocr'
                row['quantity_ocr'] = attempt
    update_receipt(result)
    applied = any(attempt["agreed"] or attempt.get("edge_agreed") for attempt in proposals) and result["calculation"].get("validation_status",result["calculation"]["status"])=="match"
    # If unrelated price errors prevent total validation, keep only the supported
    # edge readings and explicitly label them unverified for operator review.
    provisional = []
    if not applied:
        provisional = [attempt['row'] for attempt in proposals if attempt.get('edge_agreed')]
        if provisional:
            result = copy.deepcopy(receipt)
            for row in result['rows']:
                if row['row'] in provisional:
                    row['quantity'] = '1'
                    row['quantity_source'] = 'cell_ocr'
                    row['quantity_ocr'] = next(attempt for attempt in proposals if attempt['row']==row['row'])
                    row['quantity_unverified'] = True
            update_receipt(result)
    record = dict(version=VERSION,attempts=proposals,applied=applied or bool(provisional),
                  provisional_rows=provisional,
                  proposed_status=result["calculation"]["status"],
                  proposed_validation_status=result["calculation"].get("validation_status",result["calculation"]["status"]),
                  vat_status=result["calculation"].get("vat",{}).get("status"),
                  gate="two OCR views agree, printed total and VAT match")
    return result if applied or provisional else copy.deepcopy(receipt),record,crops
