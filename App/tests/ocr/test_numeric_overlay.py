import numpy as np

from App.ocr.numeric_overlay import overlay_invalid_numeric_cells


def test_overlay_marks_invalid_quantity_cell_red():
    image = np.full((100, 160, 3), 255, dtype=np.uint8)
    receipt = {'rows': [{'row': 5, 'kind': 'product', 'y_range': [20, 60], 'invalid_fields': ['quantity'], 'review_reason': 'Проверьте количество'}]}
    layout = {'name_right': 70, 'quantity_right': 100, 'right': 150}
    out, report = overlay_invalid_numeric_cells(image, receipt, layout, alpha=0.5)
    assert report['cells']
    cell = report['cells'][0]
    assert cell['row'] == 5 and cell['field'] == 'quantity'
    sample = out[40, 85]
    assert int(sample[2]) > int(sample[0])
    assert int(sample[2]) > int(sample[1])
