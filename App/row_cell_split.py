"""Locally split a merged product row only with printed and numeric evidence."""
import copy
import json
import numpy as np
from App.wave_dewarp import find_rules, line_mask
from App.ocr_catalog_eval import aligned_lines, parse_rows, split_quantity_price_line
from App.receipt_editing import parse_number, update_receipt


def numeric_pairs(members,layout):
    pairs=[];quantities=[];prices=[]
    for line in members:
        box=np.asarray(line['box'],float);cx,cy=box.mean(axis=0)
        if cx<layout['name_right']-2:continue
        split=split_quantity_price_line(dict(line,center=np.array([cx,cy])),layout)
        if split and box[:,0].min()<layout['quantity_right']<box[:,0].max():
            q,p=map(parse_number,split)
            if q is not None and p is not None and q>0 and p>0:
                pairs.append(dict(y=float(cy),quantity=split[0],price=split[1]));continue
        if parse_number(line['text']) is None:continue
        item=dict(y=float(cy),top=float(box[:,1].min()),bottom=float(box[:,1].max()),text=line['text'])
        if layout['name_right']<=cx<layout['quantity_right']:quantities.append(item)
        elif layout['quantity_right']<=cx<=layout['right']+4:prices.append(item)
    used=set()
    for price in prices:
        compatible=[(i,q) for i,q in enumerate(quantities) if i not in used and
                    min(q['bottom'],price['bottom'])-max(q['top'],price['top'])>
                    .35*min(q['bottom']-q['top'],price['bottom']-price['top'])]
        if len(compatible)==1:
            i,q=compatible[0];used.add(i)
            pairs.append(dict(y=(q['y']+price['y'])/2,quantity=q['text'],price=price['text']))
    return sorted(pairs,key=lambda pair:pair['y'])


def curve_support(rule,mask,left,right,radius=2):
    xs=np.arange(max(0,int(left)+3),min(mask.shape[1],int(right)-3))
    if len(xs)<5:return 0.
    ys=np.rint(rule.values[xs]).astype(int)
    hit=np.zeros(len(xs),bool)
    for offset in range(-radius,radius+1):hit|=mask[np.clip(ys+offset,0,mask.shape[0]-1),xs]>0
    return float(hit.mean())


def repair(receipt,layout,lines,image):
    result=copy.deepcopy(receipt);diagnostics=[];plans=[]
    # Geometry work is gated by an actual numeric anomaly, not long names.
    for index,row in enumerate(receipt['rows']):
        if row.get('kind')!='product':continue
        pairs=numeric_pairs(row.get('raw_lines',[]),layout)
        if len(pairs)>1:plans.append((index,row,pairs))
    if not plans:return result,[dict(applied=False,reason='no_merged_product_row')]
    rules,mask=find_rules(image);vertical=line_mask(image,vertical=True)
    replacements={}
    for index,row,pairs in plans:
        diagnostic=dict(row=row['row'],pairs=pairs,applied=False);diagnostics.append(diagnostic)
        if len(pairs)!=2:
            diagnostic['reason']='only_unambiguous_two_way_split_supported';continue
        bounds=row.get('y_range')
        if not bounds:
            diagnostic['reason']='missing_original_band';continue
        x=int((layout['name_left']+layout['name_right'])/2)
        candidates=[r for r in rules if pairs[0]['y']+5<r.values[x]<pairs[1]['y']-5
                    and bounds[0]+8<r.values[x]<bounds[1]-8 and r.coverage>=.8 and r.fit_error<=2]
        confirmed=[]
        for rule in candidates:
            supports=[curve_support(rule,mask,a,b) for a,b in zip(
                [layout['name_left'],layout['name_right'],layout['quantity_right']],
                [layout['name_right'],layout['quantity_right'],layout['right']])]
            joints=0
            for divider in [layout['name_right'],layout['quantity_right']]:
                xx=int(round(divider));yy=int(round(rule.values[min(xx,image.shape[1]-1)]))
                patch=vertical[max(0,yy-4):yy+5,max(0,xx-3):xx+4]
                joints+=bool(patch.size and np.count_nonzero(patch)>=5)
            if sum(s>=.7 for s in supports)>=2 and joints==2:confirmed.append(rule)
        if len(confirmed)!=1:
            diagnostic['reason']='printed_separator_not_uniquely_confirmed';continue
        rule=confirmed[0];members=row.get('raw_lines',[])
        # A block crossing the proposed separator needs cell OCR, not slicing
        # or guessing its text. This initial experiment safely rejects it.
        crossing=[]
        groups=[[],[]]
        for member in members:
            box=np.asarray(member['box'],float);columns=np.clip(np.rint(box[:,0]).astype(int),0,image.shape[1]-1)
            distance=box[:,1]-rule.values[columns]
            if distance.min() < -1 and distance.max() > 1:crossing.append(member['text'])
            cx,cy=box.mean(axis=0);side=int(cy>=rule.values[int(np.clip(cx,0,image.shape[1]-1))])
            groups[side].append(member)
        if crossing:
            diagnostic.update(reason='crossing_ocr_requires_cell_reread',crossing=crossing);continue
        cut=float(rule.values[x]);children=[]
        for group,band in zip(groups,[[bounds[0],cut],[cut,bounds[1]]]):
            prepared=aligned_lines(dict(lines=group),{})
            prepared.sort(key=lambda line:(float(line['center'][1]),float(line['center'][0])))
            for rank,line in enumerate(prepared):
                cy=band[0]+(rank+1)/(len(prepared)+1)*(band[1]-band[0])
                line['center']=np.array([line['center'][0],cy]);line['eval_box'][:,1]=cy
            parsed=parse_rows(prepared,layout,band)
            if len(parsed)!=1 or parsed[0]['kind']!='product' or not parsed[0]['name_raw'].strip():break
            child=parsed[0]
            if parse_number(child['quantity_raw']) is None or parse_number(child['price_raw']) is None:break
            child['source_row']=row['row'];children.append(child)
        if len(children)!=2:
            diagnostic['reason']='children_not_complete_independent_products';continue
        # Every original source block must occur once, including numeric blocks.
        evidence=lambda items:sorted(json.dumps({k:l.get(k) for k in ['text','box','confidence']},sort_keys=True) for l in items)
        if evidence(members)!=evidence([l for child in children for l in child['raw_lines']]):
            diagnostic['reason']='evidence_not_preserved';continue
        diagnostic.update(reason='candidate',cut=cut)
        replacements[index]=(children,diagnostic)
    if not replacements:return result,diagnostics
    revised=[]
    for index,row in enumerate(receipt['rows']):revised.extend(replacements[index][0] if index in replacements else [copy.deepcopy(row)])
    for index,row in enumerate(revised,1):row['row']=index
    proposed=dict(result,rows=revised)
    calculation=update_receipt(copy.deepcopy(proposed))['calculation']
    if calculation.get('validation_status')!='match':
        for _,diagnostic in replacements.values():diagnostic['reason']='full_math_check_not_passed'
        return result,diagnostics
    for _,diagnostic in replacements.values():diagnostic.update(applied=True,reason='two_pairs_and_printed_separator_and_complete_children_and_math')
    return proposed,diagnostics
