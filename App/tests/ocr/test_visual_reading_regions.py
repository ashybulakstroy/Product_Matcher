import numpy as np
import pytest
from App.ocr.web_pipeline import ReceiptPipeline
from App.ocr import footer_total_recovery as footer


def test_reading_highlight_brackets_each_actual_recognition_batch():
    events=[]
    boxes=np.array([[[0,0],[10,0],[10,10],[0,10]],[[20,20],[30,20],[30,30],[20,30]]],dtype=float)
    class Recognizer:
        rec_batch_num=1
        def __call__(self,images):
            events.append(('actual_read',len(images)))
            return [('1',.99)]*len(images),.01
    class Engine:
        text_detector=staticmethod(lambda _: (boxes,0.))
        text_recognizer=Recognizer()
        sorted_boxes=staticmethod(lambda x:x)
        def __call__(self,path):
            self.text_detector(None)
            self.text_recognizer([np.zeros((10,10,3),np.uint8)]*2)
            return [],None
    class Job:
        def publish(self,kind,**kw):
            if kind=='visual':events.append((kw['action'],kw.get('boxes')))
    pipeline=ReceiptPipeline(knowledge={});pipeline.engine=Engine()
    original=pipeline.engine.text_recognizer
    pipeline.recognize('unused.png',Job(),'rectified',64)
    assert [e[0] for e in events]==['scan_region','actual_read','scan_region_end']*2
    assert events[0][1]==[boxes[0].tolist()]
    assert events[3][1]==[boxes[1].tolist()]
    assert pipeline.engine.text_recognizer is original


@pytest.mark.parametrize('fail',[False,True])
def test_footer_highlight_is_removed_even_when_ocr_fails(monkeypatch,fail):
    events=[]
    monkeypatch.setattr(footer,'find_printed_amount_crop',lambda *a:((1,2,20,15),dict(status='amount_crop_found')))
    def read(*args):
        assert events==[((1,2,20,15),True)]
        if fail:raise RuntimeError('test failure')
        return '10.00',dict(status='confirmed')
    monkeypatch.setattr(footer,'_read_footer_crop',read)
    if fail:
        with pytest.raises(RuntimeError):footer.recover_footer_total(np.zeros((30,30,3),np.uint8),None,on_region=lambda *x:events.append(x))
    else:
        value,_=footer.recover_footer_total(np.zeros((30,30,3),np.uint8),None,on_region=lambda *x:events.append(x))
        assert value=='10.00'
    assert events==[((1,2,20,15),True),((1,2,20,15),False)]
