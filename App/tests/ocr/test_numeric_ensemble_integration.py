import numpy as np
import pytest

from App.ocr import numeric_fallback_ensemble as ensemble
from App.ocr import numeric_recovery as recovery
from App.ocr import selective_price_ocr


@pytest.mark.parametrize('mode', ['disabled', 'error', 'corrected', 'matching'])
def test_ensemble_preserves_legacy_fallback(monkeypatch, mode):
    receipt = {'rows': [{'row': 1, 'kind': 'product', 'quantity_raw': '1', 'price_raw': '907'}], 'total_raw': '90'}
    calls = []
    monkeypatch.setattr(selective_price_ocr, 'recover_missing_prices', lambda current, *a, **k: (current, {'applied': []}))
    monkeypatch.setattr(recovery, '_clean_state', lambda current: (
        {'status': 'match' if mode == 'matching' or current['rows'][0].get('price') == '90' else 'mismatch', 'vat': {'status': 'not_applicable'}}, 'not_applicable'))
    monkeypatch.setattr(recovery, '_crop_cell', lambda *a: None)
    monkeypatch.setattr(recovery, '_printed_total_crop', lambda *a: None)
    monkeypatch.setattr(recovery, '_recognition_pairs', lambda *a: calls.append('legacy') or [])
    monkeypatch.setattr(ensemble, 'load_ensemble_options', lambda: {'enabled': mode != 'disabled', 'tesseract_exe': 'tesseract', 'enable_easyocr': False, 'max_cells': 4, 'max_views': 2, 'use_tesseract': False, 'require_two_engines': True})
    monkeypatch.setattr(ensemble, 'optional_easyocr_reader', lambda *a, **k: None)

    def run(current, *args, **kwargs):
        calls.append('ensemble')
        if mode == 'error':
            raise RuntimeError('unavailable')
        current['rows'][0]['price'] = '90'
        return current, {'triggered': True, 'applied': [{'row': 1, 'field': 'price'}], 'warnings': []}

    monkeypatch.setattr(ensemble, 'recover_mismatched_numbers', run)
    cfg = {'vat_rate_percent': '16', 'vat_rate_source': 'test', 'max_images': 10}
    result, audit = recovery.recover_numeric_receipt(receipt, np.zeros((10, 10, 3), dtype=np.uint8), {}, [], lambda _: [], settings=cfg)
    assert ('legacy' in calls) == (mode in {'disabled', 'error'})
    assert ('ensemble' in calls) == (mode in {'error', 'corrected'})
    assert result['rows'][0]['price_raw'] == '907'
    if mode == 'corrected':
        assert result['rows'][0]['price'] == '90'
        assert audit['applied_count'] == 1
    if mode == 'error':
        assert audit['ensemble']['status'] == 'error'
