"""Synthetic tests for lexical algorithms 14/15; safe for OCR regression."""
import importlib
import math

import pytest

from App.matching.engine import match
from App.matching.registry import get_algorithm

METHODS = ('bm25_14', 'tfidf_15')


def candidates(identifier, name, stock, top_n=5):
    spec = get_algorithm(identifier)
    method = getattr(importlib.import_module(spec.module), spec.function)
    return method([{'id': 'Q000', 'name': name}], stock, top_n=top_n)[0]['candidates']


@pytest.mark.parametrize('identifier', METHODS)
def test_registered_and_call_contract(identifier):
    spec = get_algorithm(identifier)
    assert spec.execution == 'cpu'
    assert spec.module.startswith('App.matching.algorithms.lexical.')
    stock = [{'id': '00000125', 'name': 'Болт М8 20 мм'},
             {'id': '00000126', 'name': 'Гайка М8'}]
    result = candidates(identifier, 'Болт М8 20 мм', stock)
    assert result[0]['stock_id'] == '00000125'
    assert result[0]['similarity_pct'] == pytest.approx(100.0)
    assert result[0]['raw_score'] == pytest.approx(1.0)
    assert result[0]['stock_name'] == stock[0]['name']
    assert result[0]['match_reasons']


@pytest.mark.parametrize('identifier', METHODS)
def test_two_receipt_rows_and_duplicate_names_remain_independent(identifier):
    spec = get_algorithm(identifier)
    method = getattr(importlib.import_module(spec.module), spec.function)
    checked = [{'id': '000000Q1', 'name': 'Гайка М8'},
               {'id': '000000Q2', 'name': 'Болт М8'}]
    stock = [{'id': '00000001', 'name': 'Гайка М8'},
             {'id': '00000002', 'name': 'Гайка М8'},
             {'id': '00000003', 'name': 'Болт М8'}]
    result = method(checked, stock, top_n=len(stock))
    assert len(result) == 2
    assert [row['checked_id'] for row in result] == ['000000Q1', '000000Q2']
    assert [c['stock_id'] for c in result[0]['candidates'][:2]] == ['00000001', '00000002']
    assert result[1]['candidates'][0]['stock_id'] == '00000003'
    assert checked[0]['name'] == 'Гайка М8'  # no mutation


@pytest.mark.parametrize('identifier', METHODS)
def test_missing_name_empty_stock_and_invalid_top_n(identifier):
    spec = get_algorithm(identifier)
    method = getattr(importlib.import_module(spec.module), spec.function)
    assert method([{'name': 'Болт'}], [], top_n=5)[0]['candidates'] == []
    assert method([{'name': ''}], [{'id': '1', 'name': 'Болт'}], top_n=5)[0]['candidates'] == []
    assert method([{'name': 'Болт'}], [{'id': '1', 'name': 'Болт'}], top_n=0)[0]['candidates'] == []
    with pytest.raises(ValueError):
        method([{'name': 'Болт'}], [], top_n=True)


def test_bm25_rare_terms_and_no_accidental_semantics():
    stock = ([{'id': f'A{i}', 'name': 'Краска белая водная'} for i in range(25)] +
             [{'id': '00000077', 'name': 'Сверло по металлу 8 мм'}])
    result = candidates('bm25_14', 'Сверло металлу 8 мм', stock)
    assert result[0]['stock_id'] == '00000077'
    assert candidates('bm25_14', 'жираф', [{'id': 's', 'name': 'болт'}]) == []


def test_tfidf_ocr_typo_character_ngrams():
    stock = [{'id': '000001', 'name': 'Саморез кровельный 4,8х35'},
             {'id': '000002', 'name': 'Клей плиточный 25 кг'}]
    result = candidates('tfidf_15', 'Саморез кровелный 4,8х35', stock)
    assert result and result[0]['stock_id'] == '000001'
    assert result[0]['raw_score'] > 0.1


@pytest.mark.parametrize('identifier', METHODS)
def test_name_change_rebuilds_memory_cache(identifier):
    stock_first = [{'id': '0001', 'name': 'Болт'}, {'id': '0002', 'name': 'Гайка'}]
    stock_second = [{'id': '0001', 'name': 'Гайка'}, {'id': '0002', 'name': 'Болт'}]
    assert candidates(identifier, 'Болт', stock_first)[0]['stock_id'] == '0001'
    assert candidates(identifier, 'Болт', stock_second)[0]['stock_id'] == '0002'


@pytest.mark.parametrize('identifier', METHODS)
def test_shared_stock_price_ranking_and_cpu_parallelism(identifier):
    checked = [{'id': 'T1', 'name': 'Болт М8', 'price': 100}]
    stock = [{'id': 'a', 'name': 'Болт М8', 'price': 200, 'quantity': 10},
             {'id': 'b', 'name': 'Болт М8', 'price': 101, 'quantity': 10},
             {'id': 'c', 'name': 'Болт М8', 'price': 100, 'quantity': 1}]
    sequential = match(checked, stock, top_n=1, min_stock=3,
                       algorithms=[identifier], parallel=False)['results_by_algorithm'][0]
    parallel = match(checked, stock, top_n=1, min_stock=3,
                     algorithms=[identifier], parallel=True)['results_by_algorithm'][0]
    assert sequential['status'] == 'ok'
    assert parallel['status'] == 'ok'
    assert parallel['executor'] == 'cpu_process'
    assert sequential['items'] == parallel['items']
    assert sequential['best_candidate']['stock_id'] == 'b'
    assert sequential['low_stock_candidate']['stock_id'] == 'c'


@pytest.mark.parametrize('identifier', METHODS)
def test_scores_are_finite_bounded_and_ordered(identifier):
    stock = [{'id': f'{n:08d}', 'name': name} for n, name in enumerate([
        'Болт М8 20 мм', 'Болт М8 30 мм', 'Гайка М8', 'Круг отрезной 125 мм',
        'Кабель медный 3х2,5', 'Пена монтажная', 'Болгарка 125 мм',
    ])]
    result = candidates(identifier, 'Болт М8 20 мм', stock, len(stock))
    assert result
    assert all(math.isfinite(c['raw_score']) and 0 < c['raw_score'] <= 1 for c in result)
    assert all(math.isfinite(c['similarity_pct']) and 0 < c['similarity_pct'] <= 100 for c in result)
    assert result == sorted(result, key=lambda c: (-c['similarity_pct'], c['stock_id']))
    assert len({c['stock_id'] for c in result}) == len(result)


@pytest.mark.parametrize('identifier', METHODS)
def test_cached_names_do_not_cache_identifiers_prices_or_stock(identifier):
    checked = [{'id': 'q', 'name': 'Болт М8', 'price': 100}]
    first = [{'id': '00001', 'name': 'Болт М8', 'quantity': 10, 'price': 100}]
    second = [{'id': '00002', 'name': 'Болт М8', 'quantity': 1, 'price': 180}]
    a = match(checked, first, algorithms=[identifier], min_stock=3, parallel=False)
    b = match(checked, second, algorithms=[identifier], min_stock=3, parallel=False)
    assert a['best_candidate']['stock_id'] == '00001'
    assert b['best_candidate'] is None
    assert b['low_stock_candidate']['stock_id'] == '00002'
    assert b['low_stock_candidate']['stock_price'] == 180


def test_both_lexical_algorithms_participate_in_voting():
    result = match([{'id': 'q', 'name': 'Болт М8'}],
                   [{'id': '00001', 'name': 'Болт М8', 'quantity': 10}],
                   algorithms=list(METHODS), min_stock=3, parallel=True)
    vote = result['voting']['positions'][0]
    assert vote['recommended_stock_id'] == '00001'
    assert vote['voting_algorithms'] == 2
    assert set(vote['candidates'][0]['scores_by_algorithm']) == set(METHODS)
