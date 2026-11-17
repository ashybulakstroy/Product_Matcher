"""MANDATORY acceptance: full OCR against real user photo (run explicitly)."""
import hashlib
import json
import os
from pathlib import Path

import pytest

from App.ocr.receipt_editing import parse_number

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'data/ocr_regression/row3_069_690'

pytestmark = pytest.mark.skipif(
    os.environ.get('OCR_REAL_PHOTO_TESTS') != '1',
    reason='Explicitly set OCR_REAL_PHOTO_TESTS=1: a skipped test is NOT acceptance',
)


class _NoCatalogMatch:
    def match(self, name):
        return dict(status='not_matched', best=None, top_matches=[])


def test_real_photo_full_pipeline_recovers_price_and_exact_total(monkeypatch, tmp_path):
    from rapidocr_onnxruntime import RapidOCR
    from App.ocr.ocr_run import MODEL
    from App.ocr.web_pipeline import ReceiptPipeline
    from App.web_server import Job

    photo = DATA / 'source/original.png'
    expected = json.loads((DATA / 'expected/acceptance.json').read_text(encoding='utf-8'))
    assert photo.is_file() and MODEL.is_file()
    assert hashlib.sha256(photo.read_bytes()).hexdigest() == expected['photo_sha256']

    pipeline = ReceiptPipeline()
    matcher = _NoCatalogMatch()

    def resources():
        if pipeline.engine is None:
            pipeline.engine = RapidOCR(rec_model_path=str(MODEL), rec_img_shape=[3, 48, 320], text_score=.3)
        pipeline.matcher = matcher

    monkeypatch.setattr(pipeline, 'resources', resources)
    (tmp_path / 'upload.bin').write_bytes(photo.read_bytes())
    output = pipeline.process(Job('a' * 32, photo.name, tmp_path))
    row = next(r for r in output['rows'] if r['row'] == 3)
    assert parse_number(row['price']) == parse_number('690'), row
    calc = output['calculation']
    assert calc['validation_status'] == 'match', calc
    assert calc['computed_total'] == '155626.00'
    assert calc['products_subtotal'] == '134160.00'
    assert calc['vat']['actual_tax'] == '21466.00'
    assert output['numeric_overlay']['cells'] == []
