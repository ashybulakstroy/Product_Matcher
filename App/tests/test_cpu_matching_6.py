import importlib

import pytest

from App.matching.algorithms.fuzzy.copilot_1 import match_products
from App.matching.algorithms.fuzzy.cpu6_normalization import normalize_name
from App.matching.engine import match
from App.matching.registry import get_algorithm

NEW_IDS = ['exact_normalized', 'token_sort', 'token_set', 'levenshtein', 'jaro_winkler', 'jaccard']


def search(identifier, query, stock):
    spec = get_algorithm(identifier)
    return getattr(importlib.import_module(spec.module), spec.function)([{'name': query}], stock)[0]['candidates']


@pytest.mark.parametrize('raw,expected', [
    (' Ёмкость 1,5 л ', 'емкость 1.5 л'),
    ('4,8Х35', '4.8x35'), ('4.8×35', '4.8x35'), ('900Вт', '900 вт'),
])
def test_donor_normalization_is_preserved(raw, expected):
    assert normalize_name(raw) == expected


@pytest.mark.parametrize('identifier', NEW_IDS)
def test_six_algorithms_find_normalized_match_and_keep_string_codes(identifier):
    candidates = search(identifier, 'Саморез кровельный 4.8×35',
        [{'id': '00000001', 'name': 'Саморез кровельный 4,8х35'},
         {'id': '00000002', 'name': 'Муфта ППР 20 мм'}])
    assert candidates[0]['stock_id'] == '00000001'
    assert candidates[0]['raw_score'] == 1


@pytest.mark.parametrize('identifier', ['exact_normalized', 'jaccard'])
def test_no_evidence_produces_no_candidate(identifier):
    assert not search(identifier, 'Болгарка', [{'id': 's', 'name': 'Шуруп'}])


def test_exact_rejects_different_size():
    assert not search('exact_normalized', 'Саморез 4,8х35', [{'id': 's', 'name': 'Саморез 4,8х50'}])


def test_token_sort_is_not_a_duplicate_of_copilot_maximum():
    query = 'болт гайка'
    stock = [{'id': 'a', 'name': 'долт гайка'}, {'id': 'b', 'name': 'болт гайки'}]
    assert match_products([{'name': query}], stock)[0]['candidates'][0]['stock_id'] == 'a'
    assert search('token_sort', query, stock)[0]['stock_id'] == 'b'


def test_token_set_tolerates_extra_words_unlike_sort():
    stock = [{'id': 's', 'name': 'болт гайка шайба'}]
    assert search('token_set', 'болт гайка', stock)[0]['raw_score'] == 1
    assert search('token_sort', 'болт гайка', stock)[0]['raw_score'] < 1


def test_levenshtein_is_not_the_indel_ratio_used_by_copilot():
    stock = [{'id': 's', 'name': 'ba'}]
    assert match_products([{'name': 'ab'}], stock)[0]['candidates'][0]['similarity_pct'] == 50
    assert not search('levenshtein', 'ab', stock)


def test_jaro_winkler_transposition_and_jaccard_overlap():
    assert search('jaro_winkler', 'MARTHA', [{'id': 's', 'name': 'MARHTA'}])[0]['raw_score'] == pytest.approx(0.9611111111)
    assert search('jaccard', 'болт гайка', [{'id': 's', 'name': 'болт гайка шайба'}])[0]['raw_score'] == pytest.approx(2 / 3)


def test_new_algorithms_use_shared_price_stock_and_parallel_execution():
    checked = [{'name': 'Болт М8', 'price': 100}]
    stock = [{'id': 'a', 'name': 'Болт М8', 'price': 200, 'quantity': 10},
             {'id': 'b', 'name': 'Болт М8', 'price': 101, 'quantity': 10},
             {'id': 'c', 'name': 'Болт М8', 'price': 100, 'quantity': 1}]
    parallel = match(checked, stock, algorithms=NEW_IDS, top_n=1, min_stock=3)
    sequential = match(checked, stock, algorithms=NEW_IDS, top_n=1, min_stock=3, parallel=False)
    for result, original in zip(parallel['results_by_algorithm'], sequential['results_by_algorithm']):
        assert result['executor'] == 'cpu_process'
        assert result['best_candidate']['stock_id'] == 'b'
        assert result['low_stock_candidate']['stock_id'] == 'c'
        assert result['items'] == original['items']
