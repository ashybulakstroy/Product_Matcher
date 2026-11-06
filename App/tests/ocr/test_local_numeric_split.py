import copy
import pytest
from App.ocr.numeric_cell_split import recover


def example():
    block=dict(text='1| 242000 т',confidence=.84,box=[[532,330],[707,330],[707,356],[532,356]])
    receipt=dict(total_raw='247780',rows=[
        dict(row=2,kind='product',name_raw='Масло',quantity_raw='1',price_raw='5780',quantity='1',price='5780',raw_lines=[],y_range=[106,290]),
        dict(row=3,kind='product',name_raw='Генератор',quantity_raw='',price_raw=block['text'],quantity='',price=block['text'],raw_lines=[block],y_range=[290,399])])
    return receipt,dict(name_right=445,quantity_right=552,right=713),[copy.deepcopy(block)]


def test_repairs_only_two_fields_and_preserves_evidence_and_grid():
    before,layout,lines=example();snapshot=copy.deepcopy(before)
    after,diagnostics=recover(before,layout,lines)
    assert after['rows'][1]['quantity']=='1' and after['rows'][1]['price']=='242000 т'
    assert diagnostics[0]['applied']
    assert before==snapshot
    for old,new in zip(before['rows'],after['rows']):
        for field in ['name_raw','raw_lines','quantity_raw','price_raw','y_range','row','kind']:assert old[field]==new[field]


@pytest.mark.parametrize('mode',['no_crossing','no_separator','wrong_total','multiple_candidates','conflicting_quantity','foreign_block','outside_row'])
def test_guard_keeps_original_when_not_unambiguous(mode):
    before,layout,lines=example();row=before['rows'][1]
    if mode=='no_crossing':row['raw_lines'][0]['box']=[[570,330],[707,330],[707,356],[570,356]];lines=copy.deepcopy(row['raw_lines'])
    if mode=='no_separator':row['raw_lines'][0]['text']='1 242000 т';lines=copy.deepcopy(row['raw_lines'])
    if mode=='wrong_total':before['total_raw']='123'
    if mode=='multiple_candidates':row['raw_lines']*=2
    if mode=='conflicting_quantity':row['quantity']='2'
    if mode=='foreign_block':lines=[]
    if mode=='outside_row':row['y_range']=[100,250]
    snapshot=copy.deepcopy(before)
    after,diagnostics=recover(before,layout,lines)
    assert after==snapshot and not any(item['applied'] for item in diagnostics)


def test_does_not_touch_valid_numeric_pair():
    before,layout,lines=example();before['rows'][1].update(quantity='1',price='242000')
    after,diagnostics=recover(before,layout,lines)
    assert after==before and diagnostics==[]


def test_web_pipeline_keeps_repaired_price_through_matching_and_saving(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    import cv2
    import numpy as np
    import App.ocr.web_pipeline as pipeline_module
    from App.web_server import Job
    before,layout,lines=example()
    # Pipeline OCR serialises coordinates as floats; reproduce its evidence.
    for line in lines:line['box']=[[float(x),float(y)] for x,y in line['box']]
    before['rows'][1]['raw_lines']=copy.deepcopy(lines)
    for row in before['rows']:row['ocr_confidence']=.9
    layout.update(body_left=16,name_left=102,body_top=16,body_bottom=399,row_boundaries=[16,106,290,399])
    image=np.full((480,747,3),240,np.uint8)
    yy,xx=np.indices(image.shape[:2],dtype=np.float32)
    polygon=np.array([[16,16],[713,16],[713,399],[16,399]],np.float32)
    metadata=dict(method='contour_coons_patch',margin=16,body_size=[697,383],warnings=[])
    monkeypatch.setattr(pipeline_module,'detect_table_boundary',lambda im:dict(polygon=polygon,status='detected',warnings=[]))
    monkeypatch.setattr(pipeline_module,'rectify_table',lambda *a,**kw:(image,xx,yy,metadata))
    monkeypatch.setattr(pipeline_module,'table_layout',lambda *a:copy.deepcopy(layout))
    monkeypatch.setattr(pipeline_module,'choose_row_boundaries',lambda *a:(layout['row_boundaries'],'grid'))
    monkeypatch.setattr(pipeline_module,'parse_rows',lambda *a:copy.deepcopy(before['rows']))
    pipeline=pipeline_module.ReceiptPipeline()
    pipeline.matcher=SimpleNamespace(match=lambda text:dict(status='not_found',best=None,top_matches=[],gap=0))
    pipeline.catalog_count=0
    monkeypatch.setattr(pipeline,'resources',lambda:None)
    monkeypatch.setattr(pipeline_module,'orient_receipt',lambda image,**kwargs:(image,{
        'status':'UNCERTAIN','rotation_clockwise':0}))
    total_box=[[560.,420.],[710.,420.],[710.,450.],[560.,450.]]
    monkeypatch.setattr(pipeline,'recognize',lambda *a,**kw:[(line['box'],line['text'],line['confidence']) for line in lines]+[(total_box,'247780.00 T',.9)])
    (tmp_path/'upload.bin').write_bytes(cv2.imencode('.png',image)[1].tobytes())
    result=pipeline.process(Job('a'*32,'receipt.png',tmp_path))
    generator=result['rows'][1]
    assert generator['quantity']=='1' and generator['price']=='242000 т'
    assert generator['quantity_raw']=='' and generator['price_raw']=='1| 242000 т'
    assert generator['name_raw']=='Генератор' and generator['y_range']==[290,399]
    assert result['calculation']['validation_status']=='match'
    saved=json.loads((tmp_path/'receipt.json').read_text(encoding='utf-8'))
    assert saved['rows'][1]['price']=='242000 т'
    assert json.loads((tmp_path/'numeric_split.json').read_text(encoding='utf-8'))[0]['applied']
