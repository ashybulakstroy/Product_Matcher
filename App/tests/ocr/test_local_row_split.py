"""Synthetic boundary cases exercise the experimental gate, not OCR accuracy."""
import copy
import cv2
import numpy as np
from App.ocr.row_cell_split import repair


def fixture(separator=True):
    image=np.full((260,510,3),255,np.uint8)
    for x in [10,50,300,380,490]:cv2.line(image,(x,20),(x,240),(0,0,0),1)
    for y in [20,240]+([130] if separator else []):cv2.line(image,(10,y),(490,y),(0,0,0),1)
    def line(text,x,y,w=50):
        box=[[float(x),float(y-9)],[float(x+w),float(y-9)],[float(x+w),float(y+9)],[float(x),float(y+9)]]
        return dict(text=text,confidence=.95,box=box,box_original=copy.deepcopy(box))
    lines=[line('Товар А',70,65,150),line('1',330,70,20),line('10',410,70,40),
           line('Товар Б',70,175,150),line('2',330,180,20),line('20',410,180,40)]
    receipt=dict(total_raw='50',rows=[dict(row=1,kind='product',name_raw='Товар А Товар Б',
        quantity_raw='1 2',price_raw='10 20',quantity='1 2',price='10 20',raw_lines=copy.deepcopy(lines),y_range=[20,240])])
    layout=dict(body_left=10,name_left=50,name_right=300,quantity_right=380,right=490,body_top=20,body_bottom=240)
    return receipt,layout,lines,image


def test_real_separator_two_pairs_and_math_restore_two_products():
    receipt,layout,lines,image=fixture();snapshot=copy.deepcopy(receipt)
    after,diagnostics=repair(receipt,layout,lines,image)
    assert len(after['rows'])==2,diagnostics
    assert [r['name_raw'] for r in after['rows']]==['Товар А','Товар Б']
    assert [(r['quantity_raw'],r['price_raw']) for r in after['rows']]==[('1','10'),('2','20')]
    assert any(d['applied'] for d in diagnostics)
    assert receipt==snapshot


def test_price_pairs_without_printed_separator_are_not_enough():
    receipt,layout,lines,image=fixture(separator=False)
    after,diagnostics=repair(receipt,layout,lines,image)
    assert after==receipt and not any(d['applied'] for d in diagnostics)


def test_wrapped_name_with_one_price_pair_does_not_trigger():
    receipt,layout,lines,image=fixture()
    receipt['rows'][0]['raw_lines']=[lines[0],lines[1],lines[2],lines[3]]
    after,diagnostics=repair(receipt,layout,lines,image)
    assert after==receipt and diagnostics[0]['reason']=='no_merged_product_row'


def test_crossing_name_requires_cell_reread_not_guessing():
    receipt,layout,lines,image=fixture()
    block=receipt['rows'][0]['raw_lines'][0]
    block['box']=[[70.,118.],[220.,118.],[220.,140.],[70.,140.]]
    after,diagnostics=repair(receipt,layout,lines,image)
    assert after==receipt
    assert diagnostics[0]['reason']=='crossing_ocr_requires_cell_reread'


def test_wrong_total_rejects_otherwise_plausible_split():
    receipt,layout,lines,image=fixture();receipt['total_raw']='999'
    after,diagnostics=repair(receipt,layout,lines,image)
    assert after==receipt and diagnostics[0]['reason']=='full_math_check_not_passed'


def test_no_transfer_between_existing_neighbour_rows():
    receipt,layout,lines,image=fixture()
    receipt['rows']=[dict(receipt['rows'][0],raw_lines=lines[:3],y_range=[20,130]),
                     dict(receipt['rows'][0],row=2,raw_lines=lines[3:],y_range=[130,240])]
    after,diagnostics=repair(receipt,layout,lines,image)
    assert after==receipt and diagnostics[0]['reason']=='no_merged_product_row'


def test_user_reviewed_real_receipt_restores_exact_approved_products():
    from pathlib import Path
    import json,hashlib
    fixtures=Path(__file__).resolve().parent/'fixtures'
    reference=json.loads((fixtures/'reviewed_row_split.json').read_text(encoding='utf-8'))
    image_path=fixtures/'reviewed_row_split.png'
    assert hashlib.sha256(image_path.read_bytes()).hexdigest()==reference['input_sha256']['rectified.png']
    layout=reference['layout'];lines=reference['ocr_lines']
    image=cv2.imdecode(np.frombuffer(image_path.read_bytes(),np.uint8),1)
    result,diagnostics=repair(reference['before'],layout,lines,image)
    columns=['name_raw','quantity_raw','price_raw']
    assert [[r[k] for k in columns] for r in result['rows']]==[[r[k] for k in columns] for r in reference['after']['rows']]
    assert any(d['applied'] for d in diagnostics)


def test_web_pipeline_applies_row_split_and_baseline_can_disable_it(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    import App.ocr.web_pipeline as pipeline_module
    from App.web_server import Job
    before,layout,lines,image=fixture()
    before['rows'][0]['ocr_confidence']=.95
    layout['row_boundaries']=[20,240]
    yy,xx=np.indices(image.shape[:2],dtype=np.float32)
    polygon=np.array([[10,20],[490,20],[490,240],[10,240]],np.float32)
    metadata=dict(method='contour_coons_patch',margin=10,body_size=[480,220],warnings=[])
    monkeypatch.setattr(pipeline_module,'detect_table_boundary',lambda im:dict(polygon=polygon,status='detected',warnings=[]))
    monkeypatch.setattr(pipeline_module,'rectify_table',lambda *a,**kw:(image,xx,yy,metadata))
    monkeypatch.setattr(pipeline_module,'table_layout',lambda *a:copy.deepcopy(layout))
    monkeypatch.setattr(pipeline_module,'choose_row_boundaries',lambda *a:([20,240],'grid'))
    monkeypatch.setattr(pipeline_module,'parse_rows',lambda *a:copy.deepcopy(before['rows']))
    for enabled in [False,True]:
        directory=tmp_path/str(enabled);directory.mkdir()
        pipeline=pipeline_module.ReceiptPipeline(split_product_rows=enabled)
        pipeline.matcher=SimpleNamespace(match=lambda text:dict(status='not_found',best=None,top_matches=[],gap=0))
        pipeline.catalog_count=0
        monkeypatch.setattr(pipeline,'resources',lambda:None)
        monkeypatch.setattr(pipeline_module,'orient_receipt',lambda image,**kwargs:(image,{
            'status':'UNCERTAIN','rotation_clockwise':0}))
        total_box=[[400.,243.],[488.,243.],[488.,259.],[400.,259.]]
        monkeypatch.setattr(pipeline,'recognize',lambda *a,**kw:[(l['box'],l['text'],l['confidence']) for l in lines]+[(total_box,'50.00 T',.95)])
        (directory/'upload.bin').write_bytes(cv2.imencode('.png',image)[1].tobytes())
        result=pipeline.process(Job('b'*32,'receipt.png',directory))
        assert len(result['rows'])==(2 if enabled else 1)
        assert (result['calculation']['validation_status']=='match')==enabled
        diagnostic=json.loads((directory/'row_split.json').read_text(encoding='utf-8'))
        assert any(d['applied'] for d in diagnostic)==enabled
        if enabled:assert [r['name_raw'] for r in result['rows']]==['Товар А','Товар Б']
