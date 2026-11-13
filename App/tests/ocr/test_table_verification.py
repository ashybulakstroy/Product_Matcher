import copy

import cv2
import numpy as np
import pytest

from App.ocr.table_verification import _header_evidence, header_role, verify_table_candidate
from App.ocr.receipt_roi_recovery import recover_compact_table
from App.ocr.table_boundary import detect_table_boundary


def table_image(scale=1):
    image = np.full((440, 640, 3), 220, np.uint8)
    xs = (40, 96, 376, 471, 600)
    for x in xs:
        cv2.line(image, (x, 50), (x, 350), (20, 20, 20), 2)
    for y in (50, 106, 220, 300, 350):
        cv2.line(image, (40, y), (600, y), (20, 20, 20), 2)
    polygon = np.array([[40, 50], [600, 50], [600, 350], [40, 350]], np.float32)
    image = cv2.resize(image, None, fx=scale, fy=scale)
    boundary = dict(polygon=polygon*scale, selected={'support': [5, 5]}, warnings=[])
    return image, boundary


@pytest.mark.parametrize('scale', [.6, 1., 2.])
def test_unreadable_header_can_be_confirmed_by_relative_grid(scale):
    image, boundary = table_image(scale)
    report = verify_table_candidate(image, boundary, lambda _: [])
    assert report['status'] == 'confirmed'
    assert report['reason'] == 'four_columns_and_header_geometry'
    assert report['header']['state'] == 'unreadable_or_incomplete'


def test_header_reader_failure_preserves_geometrically_valid_table():
    image, boundary = table_image()
    def unavailable(_):
        raise RuntimeError('OCR unavailable')
    report = verify_table_candidate(image, boundary, unavailable)
    assert report['status'] == 'confirmed'
    assert report['header']['state'] == 'unavailable'


@pytest.mark.parametrize(('text', 'role'), [('Тауардың атауы', 'name'), ('Саны', 'quantity'),
    ('Бағасы', 'price'), ('Наименование', 'name'), ('Количество', 'quantity'), ('Цена', 'price'),
    ('Канистра железная 20 л', None)])
def test_kazakh_and_russian_header_labels(text, role):
    assert header_role(text) == role


def reading(text, cx, cy, confidence=.9):
    return ([[cx-25, cy-8], [cx+25, cy-8], [cx+25, cy+8], [cx-25, cy+8]], text, confidence)


def test_header_words_need_different_columns_on_same_line():
    image = np.zeros((160, 600, 3), np.uint8)
    good = [reading('Тауардың атауы', 180, 35), reading('Саны', 440, 35)]
    assert _header_evidence(image, lambda _: good)['state'] == 'confirmed'
    wrong_column = [good[0], reading('Саны', 200, 35)]
    assert _header_evidence(image, lambda _: wrong_column)['state'] != 'confirmed'
    wrong_row = [good[0], reading('Саны', 440, 120)]
    assert _header_evidence(image, lambda _: wrong_row)['state'] != 'confirmed'
    weak = [good[0], reading('Саны', 440, 35, .1)]
    assert _header_evidence(image, lambda _: weak)['state'] != 'confirmed'


def test_uncertain_verification_does_not_mutate_existing_boundary():
    image = np.full((300, 600, 3), 200, np.uint8)
    boundary = dict(polygon=np.array([[40, 40], [550, 40], [550, 250], [40, 250]]),
                    selected={'support': [5, 5]}, warnings=['existing_warning'])
    original = copy.deepcopy(boundary)
    report = verify_table_candidate(image, boundary, lambda _: [])
    assert report['status'] == 'uncertain'
    assert np.array_equal(boundary['polygon'], original['polygon'])
    assert boundary['warnings'] == original['warnings']


def test_barcode_is_rejected_before_header_ocr():
    image = np.full((140, 70, 3), 240, np.uint8)
    for x in range(2, 65, 10):
        image[:, x:x+6] = 10
    boundary = dict(polygon=np.array([[0, 0], [69, 0], [69, 139], [0, 139]]),
                    selected={'support': [6, 6]})
    def unexpected(_):
        raise AssertionError('Do not OCR a known barcode')
    report = verify_table_candidate(image, boundary, unexpected)
    assert report['status'] == 'rejected'
    assert report['reason'] == 'barcode_not_product_grid'


def test_compact_grid_is_allowed_to_occupy_small_part_of_paper():
    image = np.full((1200, 720, 3), 45, np.uint8)
    cv2.rectangle(image, (120, 160), (540, 1060), (230, 230, 230), -1)
    for x in (145, 180, 355, 420, 510):
        cv2.line(image, (x, 410), (x, 590), (15, 15, 15), 2)
    for y in (410, 450, 515, 555, 590):
        cv2.line(image, (145, y), (510, y), (15, 15, 15), 2)
    previous = dict(polygon=None, warnings=['table_boundary_not_found'])
    boundary, report = recover_compact_table(image, detect_table_boundary, previous, lambda _: [])
    assert report['status'] == 'recovered'
    assert report['verification']['status'] == 'confirmed'
    assert boundary['polygon'][:, 1].min() > 390
    assert boundary['polygon'][:, 1].max() < 610


def test_paper_outline_alone_cannot_become_compact_table():
    image = np.full((1200, 720, 3), 45, np.uint8)
    cv2.rectangle(image, (120, 160), (540, 1060), (230, 230, 230), -1)
    previous = dict(polygon=None, warnings=['table_boundary_not_found'])
    boundary, report = recover_compact_table(image, detect_table_boundary, previous, lambda _: [])
    assert boundary is previous
    assert report['status'] == 'not_recovered'
