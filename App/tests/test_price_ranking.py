import pytest

from App.web_debug import match_debug_payload, _price


@pytest.mark.parametrize('algorithm', ['copilot_1', 'sql_str_2', 'sql_str_3'])
def test_equal_scores_use_closest_price_before_top_n(algorithm):
    result = match_debug_payload(
        [{'id': 'q', 'name': 'Молоко', 'price': '100,00'}],
        [{'id': 'a', 'name': 'Молоко', 'price': 200},
         {'id': 'b', 'name': 'Молоко'},
         {'id': 'c', 'name': 'Молоко', 'accounting_price': '101.5'}],
        top_n=1, algorithm_id=algorithm,
    )
    assert result['best_candidate']['stock_id'] == 'c'
    assert result['best_candidate']['price_difference'] == 1.5
    assert result['best_candidate']['stock_price'] == 101.5


@pytest.mark.parametrize('algorithm', ['copilot_1', 'sql_str_2', 'sql_str_3'])
def test_price_never_overrides_better_name_score(algorithm):
    result = match_debug_payload(
        [{'name': 'Молоко пастеризованное', 'price': 100}],
        [{'id': 'a', 'name': 'Молоко пастеризованное', 'price': 200},
         {'id': 'b', 'name': 'Молоко', 'price': 100}],
        algorithm_id=algorithm,
    )
    assert result['best_candidate']['stock_id'] == 'a'


def test_each_receipt_row_has_own_price_and_low_stock_ranking():
    result = match_debug_payload(
        [{'name': 'Молоко', 'price': 0}, {'name': 'Молоко', 'price': 100}],
        [{'id': 'a', 'name': 'Молоко', 'price': 100, 'quantity': 1},
         {'id': 'b', 'name': 'Молоко', 'price': 0, 'quantity': 1}],
        min_stock=10, algorithm_id='sql_str_3',
    )
    assert [row['low_stock_candidate']['stock_id'] for row in result['items']] == ['b', 'a']


def test_missing_query_price_preserves_original_order():
    result = match_debug_payload(
        [{'name': 'Молоко'}],
        [{'id': 'a', 'name': 'Молоко'}, {'id': 'b', 'name': 'Молоко', 'price': 100}],
    )
    assert result['best_candidate']['stock_id'] == 'a'


@pytest.mark.parametrize('value', ['', None, 'NaN', 'Infinity', '-1', True, 'abc'])
def test_invalid_prices_are_unknown(value):
    assert _price(value) is None
