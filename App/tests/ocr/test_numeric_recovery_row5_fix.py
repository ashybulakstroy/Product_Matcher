import numpy as np

from App.ocr import numeric_recovery as recovery
from App.ocr import selective_price_ocr


def test_non_total_fix_is_accepted_without_replacing_existing_printed_total(monkeypatch):
    receipt = {
        'rows': [
            {'row': 1, 'kind': 'product', 'name_raw': 'Товар', 'quantity_raw': '', 'price_raw': '1100', 'y_range': [0, 40]},
            {'row': 2, 'kind': 'tax', 'name_raw': 'НДС', 'quantity_raw': '160', 'price_raw': '1', 'y_range': [40, 80]},
        ],
        'total_raw': '1260',
        'expected_total': '1260',
        'expected_total_source': 'ocr',
    }
    monkeypatch.setattr(selective_price_ocr, 'recover_missing_prices', lambda current, *a, **k: (current, {'applied': []}))
    monkeypatch.setattr(recovery, '_crop_cell', lambda *a, **k: (0, 0, 10, 10))
    monkeypatch.setattr(recovery, '_printed_total_crop', lambda *a, **k: (0, 0, 10, 10))
    monkeypatch.setattr(recovery, '_recognition_pairs', lambda *a, **k: [
        dict(row=1, kind='product', field='quantity', bounds=[0, 0, 10, 10], agreed=True, candidate='1', views=[]),
        dict(row=None, kind='total', field='total', bounds=[0, 0, 10, 10], agreed=True, candidate='1260.001', views=[]),
    ])

    def fake_clean_state(current):
        qty = current['rows'][0].get('quantity', current['rows'][0].get('quantity_raw', ''))
        total = current.get('expected_total', current.get('total_raw'))
        if str(qty) == '1' and str(total) == '1260':
            return ({'status': 'match', 'vat': {'status': 'match'}}, 'match')
        if str(qty) == '1' and str(total) == '1260.001':
            return ({'status': 'mismatch', 'vat': {'status': 'match'}}, 'match')
        return ({'status': 'incomplete', 'vat': {'status': 'incomplete'}}, 'incomplete')

    monkeypatch.setattr(recovery, '_clean_state', fake_clean_state)
    cfg = {'vat_rate_percent': '16', 'vat_rate_source': 'test', 'max_images': 10}
    result, audit = recovery.recover_numeric_receipt(receipt, np.zeros((10, 10, 3), dtype=np.uint8), {'name_right': 1, 'quantity_right': 2, 'right': 3, 'body_bottom': 10}, [], lambda _: [], settings=cfg)
    assert result['rows'][0]['quantity'] == '1'
    assert result['expected_total'] == '1260'
    assert audit['accepted_without_total_retry'] is True
    assert any(item['field'] == 'quantity' for item in audit['applied'])
    assert all(item['field'] != 'total' for item in audit['applied'])
