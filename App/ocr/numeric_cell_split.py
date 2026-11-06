"""Conservative repair of merged quantity/price cells; immutable OCR evidence."""
import copy
import json
import re
from App.ocr.receipt_editing import parse_number, update_receipt


def recover(receipt,layout,ocr_lines):
    after=copy.deepcopy(receipt)
    diagnostics=[]
    def evidence_key(line):return (line.get('text'),json.dumps(line.get('box')))
    evidence={evidence_key(line) for line in ocr_lines}
    for row in after['rows']:
        if row.get('kind') not in {'product','service','tax'}:continue
        qty=row.get('quantity',row.get('quantity_raw'));price=row.get('price',row.get('price_raw'))
        if parse_number(qty) is not None and parse_number(price) is not None:continue
        matches=[]
        for line in row.get('raw_lines',[]):
            if evidence_key(line) not in evidence:continue
            text=str(line.get('text','')).strip()
            # Explicit separator only: a thousands space is not a cell boundary.
            parts=re.split(r'\s*[|!Il]\s*',text)
            if len(parts)!=2 or not re.fullmatch(r'\d+(?:[.,]\d+)?',parts[0]):continue
            left,right=parts
            q,p=parse_number(left),parse_number(right)
            if q is None or p is None or q<=0 or p<=0:continue
            box=line.get('box',[])
            if len(box)!=4:continue
            xs=[point[0] for point in box];cy=sum(point[1] for point in box)/4
            divider=layout['quantity_right']
            if not (min(xs)>=layout['name_right']-2 and min(xs)<divider-2 and max(xs)>divider+2 and max(xs)<=layout['right']+4):continue
            bounds=row.get('y_range')
            if not bounds or not bounds[0]<=cy<bounds[1]:continue
            if parse_number(qty) is not None and parse_number(qty)!=q:continue
            if parse_number(price) is not None and parse_number(price)!=p:continue
            matches.append(dict(quantity=left,price=right,source=copy.deepcopy(line)))
        item=dict(row=row['row'],candidates=len(matches),applied=False)
        diagnostics.append(item)
        if len(matches)!=1:
            item['reason']='ambiguous_or_no_cross_column_block';continue
        proposed=matches[0]
        row['quantity']=proposed['quantity'];row['price']=proposed['price']
        row['numeric_split_evidence']=proposed['source']
        item.update(proposed=proposed,reason='pending_full_math_check')
    checked=update_receipt(copy.deepcopy(after))
    # Numbers come from the source block, never from subtraction from the total.
    # Require all numeric rows and printed VAT/total to pass before accepting.
    accepted=checked['calculation'].get('validation_status')=='match'
    if not accepted:
        for item in diagnostics:
            if item.get('proposed'):item['reason']='math_check_not_passed'
        return copy.deepcopy(receipt),diagnostics
    for item in diagnostics:
        if item.get('proposed'):item.update(applied=True,reason='explicit_separator_and_grid_and_math')
    # Return only local field changes; avoid persisting unrelated initialisation.
    return after,diagnostics
