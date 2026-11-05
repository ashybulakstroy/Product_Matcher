import pytest

from App.services.opencode_matching import match_products_opencode_4
from App.web_debug import match_debug_payload


@pytest.mark.parametrize('query,correct,wrong', [
    ('TYTAN клей монтажный HYDRO FIX 310 мл прозрачный',
     'TYTAN клей монтажный HYDRO FIX 310 мл',
     'TYTAN клей монтажный CLASSIC FIX 310 мл прозрачный'),
    ('ЗУБР МК-400 черный 2-4 мм маркер-краска (06325-2)',
     'ЗУБР МК-400 черный 2-4 мм маркер-краска (06325-2) кисть щетка',
     'ЗУБР МК-400 красный 2-4 мм маркер-краска (06325-3)'),
    ('Краска белая 1 л', 'Краска белый 1000 мл', 'Краска черная 1 л'),
    ('Труба 20х30х2 мм', 'Труба 2x3x0.2 см', 'Труба 20х40х2 мм'),
    ('ЗУБР сверло 5 мм', 'ZUBR сверло 5 мм', 'BOSCH сверло 5 мм'),
    ('Болт М8 20 мм', 'Болт М8 2 см', 'Гайка М8 20 мм'),
])
def test_attribute_ranking(query, correct, wrong):
    candidates = match_products_opencode_4(
        [{'name': query}], [{'id': 'a', 'name': wrong}, {'id': 'b', 'name': correct}],
    )[0]['candidates']
    assert candidates[0]['stock_id'] == 'b'
    assert candidates[0]['match_reasons']
    assert len(candidates) == 1 or candidates[0]['similarity_pct'] > candidates[1]['similarity_pct']


def test_empty_and_unrelated_queries_return_no_candidates():
    for name in ('', 'Краскопульт'):
        assert not match_products_opencode_4([{'name': name}], [{'name': 'Краска'}])[0]['candidates']


def test_article_only_can_find_matching_card():
    results = match_products_opencode_4([{'name': '06325-2'}],
        [{'id': 'a', 'name': 'Маркер 06325-3'}, {'id': 'b', 'name': 'Маркер 06325-2'}])
    assert results[0]['candidates'][0]['stock_id'] == 'b'


def test_volume_after_model_is_not_part_of_model_identifier():
    candidate = match_products_opencode_4(
        [{'name': 'TYTAN клей HYDRO FIX 310 мл'}],
        [{'id': 's', 'name': 'TYTAN клей HYDRO FIX (310 мл)'}],
    )[0]['candidates'][0]
    assert 'Модель/серия: совпадает' in candidate['match_reasons']


def test_shared_price_ranking_and_all_algorithms_include_opencode():
    result = match_debug_payload([{'name': 'Краска белая', 'price': 100}],
        [{'id': 'a', 'name': 'Краска белая', 'price': 200, 'quantity': 20},
         {'id': 'b', 'name': 'Краска белая', 'price': 101, 'quantity': 20}],
        algorithm_id='all')
    algorithm = next(r for r in result['results_by_algorithm'] if r['algorithm_id'] == 'opencode_4')
    assert algorithm['best_candidate']['stock_id'] == 'b'
    assert algorithm['best_candidate']['price_difference_display'] == '1%'
