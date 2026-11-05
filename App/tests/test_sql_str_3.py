import pytest

from App.services.matching import match_products_sql_str_3
from App.web_debug import match_debug_payload


def rank(query, names):
    return match_products_sql_str_3(
        [{'id': 'q', 'name': query}],
        [{'id': str(i), 'name': name} for i, name in enumerate(names)],
    )[0]['candidates']


@pytest.mark.parametrize('query,correct,wrong', [
    ('Электроды МР-3 3 мм 2,5 кг', 'Электроды МР3 3мм 2500 г', 'Электроды МР-3 4 мм 5 кг'),
    ('Краска белая 1 л', 'Краска белая 1000 мл', 'Краска белая 5 л'),
    ('Болт М8 20 мм', 'Болт М8 2 см', 'Болт М10 20 мм'),
])
def test_characteristics_rank_compatible_product_first(query, correct, wrong):
    results = rank(query, [wrong, correct])
    assert results[0]['stock_name'] == correct
    assert results[0]['similarity_pct'] > results[1]['similarity_pct']
    assert results[0]['match_reasons']


def test_substring_does_not_confuse_product_types():
    assert rank('краска', ['Краскопульт электрический']) == []


def test_complete_short_name_keeps_high_score():
    assert rank('Тачка садовая', ['Тачка садовая усиленная ЭКСПЕРТ'])[0]['similarity_pct'] >= 90


def test_reordering_decimal_units_and_model_spacing_are_equivalent():
    result = rank('Электроды МР-3 3 мм 2,5 кг', ['2500 г электроды МР 3 3мм'])
    assert result[0]['similarity_pct'] == 100


def test_new_algorithm_available_through_debug_api():
    result = match_debug_payload(
        [{'id': 'q', 'name': 'Болт М8 20 мм'}],
        [{'id': 's', 'name': 'Болт М8 2 см', 'quantity': 20}],
        algorithm_id='sql_str_3',
    )
    assert result['algorithm_id'] == 'sql_str_3'
    assert result['best_candidate']['stock_id'] == 's'
