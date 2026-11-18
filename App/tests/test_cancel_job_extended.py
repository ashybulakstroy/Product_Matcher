import json
import threading
import urllib.request
import urllib.error

import numpy as np
import pytest

from App.web_server import create_server, Job, JobStore, JobCancelled
from App.ocr.web_pipeline import ReceiptPipeline


@pytest.fixture
def service(tmp_path):
    servers = []
    def start(processor):
        server = create_server('127.0.0.1', 0, tmp_path, processor=processor)
        thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': 0.05}, daemon=True)
        thread.start()
        servers.append((server, thread))
        return server, f'http://127.0.0.1:{server.server_port}'
    yield start
    for server, thread in servers:
        server.shutdown(); server.server_close()
        server.store.pool.shutdown(wait=True)
        thread.join(3)


def request(base, path, data=None, method=None):
    req = urllib.request.Request(base+path, data=data, method=method)
    with urllib.request.urlopen(req, timeout=4) as response:
        return json.load(response)


def test_cancellation_closes_sse_and_survives_reload(service, tmp_path):
    started = threading.Event()
    def processor(job):
        started.set()
        job.cancel_event.wait(3)
        job.check_cancelled()
    server, base = service(processor)
    accepted = request(base, '/api/jobs', b'image')
    assert started.wait(2)
    request(base, f"/api/jobs/{accepted['id']}/cancel", method='POST')
    with urllib.request.urlopen(base+f"/api/jobs/{accepted['id']}/events", timeout=4) as stream:
        events = stream.read().decode()
    assert '"type": "cancelled"' in events and '"type": "complete"' not in events
    server.store.pool.shutdown(wait=True)
    reloaded = JobStore(tmp_path, processor=lambda job: pytest.fail('cancelled job restarted'))
    try:
        job = reloaded.get(accepted['id'])
        assert job.status == 'cancelled' and job.cancel_event.is_set()
        with pytest.raises(JobCancelled): job.check_cancelled()
        assert not (job.directory/'error.log').exists()
    finally:
        reloaded.pool.shutdown(wait=True)


@pytest.mark.parametrize('raise_error', [False, True])
def test_late_processor_result_or_exception_cannot_overwrite_cancel(service, raise_error):
    started, release = threading.Event(), threading.Event()
    def processor(job):
        started.set()
        assert release.wait(3)
        if raise_error: raise ValueError('native OCR failed after cancellation')
        return {'ok': True}
    server, base = service(processor)
    accepted = request(base, '/api/jobs', b'image')
    assert started.wait(2)
    try:
        response = request(base, f"/api/jobs/{accepted['id']}/cancel", method='POST')
        assert response['status'] == 'cancelled'
    finally:
        release.set()
    server.store.pool.shutdown(wait=True)
    result = request(base, f"/api/jobs/{accepted['id']}")
    assert result['status'] == 'cancelled' and result['result'] is None
    assert not {'complete', 'error'} & {e['type'] for e in result['events']}
    assert not (server.store.get(accepted['id']).directory/'error.log').exists()


def test_cancel_completed_job_does_not_change_result(service):
    server, base = service(lambda job: {'ok': True})
    accepted = request(base, '/api/jobs', b'image')
    server.store.pool.shutdown(wait=True)
    response = request(base, f"/api/jobs/{accepted['id']}/cancel", method='POST')
    assert response == {'id': accepted['id'], 'status': 'done', 'cancelled': False}
    result = request(base, f"/api/jobs/{accepted['id']}")
    assert result['result'] == {'ok': True}
    assert not any(e['type']=='cancelled' for e in result['events'])


def test_cancel_rejects_request_body(service):
    _, base = service(lambda job: {'ok': True})
    with pytest.raises(urllib.error.HTTPError) as error:
        request(base, '/api/jobs/'+'f'*32+'/cancel', b'body')
    assert error.value.code == 400


def test_cancel_during_batch_restores_engine_and_never_starts_next_batch(tmp_path):
    job = Job('a'*32, 'test.png', tmp_path)
    job.status = 'processing'
    calls = []
    boxes = np.array([[[0,0],[10,0],[10,10],[0,10]], [[20,20],[30,20],[30,30],[20,30]]], dtype=float)
    class Recognizer:
        rec_batch_num = 1
        def __call__(self, images):
            calls.append(len(images))
            job.cancel()
            return [('1', .99)]*len(images), .01
    class Engine:
        text_detector = staticmethod(lambda _: (boxes, 0.))
        text_recognizer = Recognizer()
        sorted_boxes = staticmethod(lambda x: x)
        def __call__(self, path):
            self.text_detector(None)
            self.text_recognizer([np.zeros((10,10,3), np.uint8)]*2)
            return [], None
    pipeline = ReceiptPipeline(knowledge={}); pipeline.engine = Engine()
    recognizer, detector = pipeline.engine.text_recognizer, pipeline.engine.text_detector
    with pytest.raises(JobCancelled): pipeline.recognize('unused', job, 'rectified', 64)
    assert calls == [1]
    assert pipeline.engine.text_recognizer is recognizer
    assert pipeline.engine.text_detector is detector
    assert job.status == 'cancelled'
