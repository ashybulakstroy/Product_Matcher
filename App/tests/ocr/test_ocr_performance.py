import json
import threading
import pytest

from App.web_server import Job, JobCancelled
from App.ocr.web_pipeline import ReceiptPipeline
from App.ocr import ocr_performance


def test_cancelled_ocr_profile_records_cancellation(tmp_path, monkeypatch):
    job = Job('a'*32, 'test.png', tmp_path); job.status = 'processing'
    started = threading.Event(); pipe = ReceiptPipeline(knowledge={})
    def run(job, **kwargs):
        started.set(); assert job.cancel_event.wait(3); job.check_cancelled()
    monkeypatch.setattr(pipe, '_process_attempt', run)
    errors = []
    def invoke():
        try: pipe.process(job)
        except JobCancelled: errors.append('cancelled')
    thread = threading.Thread(target=invoke);thread.start();assert started.wait(2)
    job.cancel();thread.join(3)
    assert not thread.is_alive() and errors == ['cancelled']
    report = json.loads((tmp_path/'performance_profile.json').read_text(encoding='utf-8'))
    assert report['outcome'] == 'cancelled'


def test_failed_profile_persistence_does_not_fail_ocr(tmp_path, monkeypatch):
    job = Job('b'*32, 'test.png', tmp_path);pipe = ReceiptPipeline(knowledge={})
    monkeypatch.setattr(pipe, '_process_attempt', lambda job, **kwargs: {'ok': True})
    monkeypatch.setattr(ocr_performance.OCRPerformance, 'save', lambda self, directory: (_ for _ in ()).throw(ValueError('invalid profile metadata')))
    assert pipe.process(job) == {'ok': True}


def test_fast_path_math_failure_retries_full_geometry(tmp_path, monkeypatch):
    import App.ocr.web_pipeline as web
    job=Job('c'*32,'test.png',tmp_path);pipe=ReceiptPipeline(knowledge={});calls=[];saved=[]
    def attempt(job, *, orientation_mode, enable_fast_geometry=True):
        calls.append((orientation_mode,enable_fast_geometry));job._ocr_fast_selected=enable_fast_geometry
        job._ocr_fast_report={'selected':True,'status':'selected'}
        return {'calculation':{'status':'mismatch'},'rows':[],'pass':len(calls)}
    monkeypatch.setattr(pipe,'_process_attempt',attempt)
    monkeypatch.setattr(web,'save_receipt',lambda directory,result:saved.append(result.copy()))
    result=pipe.process(job)
    assert calls == [('normal',True),('normal',False)]
    assert result['pass']==2 and result['ocr_fast_path']['status']=='fallback_full_geometry'
    assert (tmp_path/'fast_path_first_pass.json').exists() and saved[-1]['pass']==2
