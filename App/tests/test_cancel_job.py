"""HTTP cancellation integration tests: no model download, no real OCR required.
Run from Product_Matcher root: python -m unittest discover -s <experiment>/tests -v
"""
import http.client
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from App.web_server import create_server


class CancelJobHttpTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.started = threading.Event()
        self.stopped = threading.Event()
        self.server = None

    def tearDown(self):
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.server.store.pool.shutdown(wait=True)
            self.server_thread.join(timeout=2)
        self.tmp.cleanup()

    def _start(self, processor):
        self.server = create_server('127.0.0.1', 0, output_dir=Path(self.tmp.name), processor=processor)
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()

    def _request(self, path, method='GET', body=None):
        conn = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=5)
        try:
            conn.request(method, path, body=body, headers=({'Content-Type': 'image/png'} if body else {}))
            reply = conn.getresponse()
            payload = json.loads(reply.read())
            return reply.status, payload
        finally:
            conn.close()

    def test_cancel_in_flight_job_is_terminal_not_done(self):
        def processor(job):
            self.started.set()
            try:
                while True:
                    job.phase('ocr', 58, 'Читаем OCR')
                    time.sleep(.01)
            finally:
                self.stopped.set()

        self._start(processor)
        code, accepted = self._request('/api/jobs?filename=test.png', 'POST', body=b'fake image')
        self.assertEqual(code, 202)
        self.assertTrue(self.started.wait(2))
        job_id = accepted['id']
        code, cancelled = self._request(f'/api/jobs/{job_id}/cancel', 'POST')
        self.assertEqual(code, 200)
        self.assertTrue(cancelled['cancelled'])
        self.assertEqual(cancelled['status'], 'cancelled')
        self.assertTrue(self.stopped.wait(2))
        code, result = self._request(f'/api/jobs/{job_id}')
        self.assertEqual(code, 200)
        self.assertEqual(result['status'], 'cancelled')
        self.assertIsNone(result['result'])
        self.assertNotIn('complete', [event['type'] for event in result['events']])
        self.assertIn('cancelled', [event['type'] for event in result['events']])
        self.assertEqual(json.loads((Path(self.tmp.name)/job_id/'state.json').read_text(encoding='utf-8'))['status'], 'cancelled')
        code, repeated = self._request(f'/api/jobs/{job_id}/cancel', 'POST')
        self.assertEqual(code, 200)
        self.assertFalse(repeated['cancelled'])

    def test_cancel_queued_job_never_enters_processor(self):
        release_first = threading.Event()
        names = []
        def processor(job):
            names.append(job.filename)
            if job.filename == 'first.png':
                self.started.set()
                release_first.wait(3)
            job.check_cancelled()
            return {'ok': True}
        self._start(processor)
        _, first = self._request('/api/jobs?filename=first.png', 'POST', b'first')
        self.assertTrue(self.started.wait(2))
        _, second = self._request('/api/jobs?filename=second.png', 'POST', b'second')
        second_id = second['id']
        try:
            code, result = self._request(f'/api/jobs/{second_id}/cancel','POST')
            self.assertEqual(code, 200)
            self.assertEqual(result['status'], 'cancelled')
        finally:
            release_first.set()
        self.server.store.pool.shutdown(wait=True)
        self.assertEqual(names, ['first.png'])
        _, result = self._request(f'/api/jobs/{second_id}')
        self.assertEqual(result['status'], 'cancelled')

    def test_bad_cancel_target_has_404(self):
        self._start(lambda job: {'ok': True})
        code, result = self._request('/api/jobs/' + 'f'*32 + '/cancel','POST')
        self.assertEqual(code, 404)
        self.assertIn('error', result)


if __name__ == '__main__':
    unittest.main()
