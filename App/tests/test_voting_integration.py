from App.matching.engine import match


def _search(algorithms, parallel=True):
    return match([{'id': 't', 'name': 'болт гайка', 'price': 100}],
        [{'id': '0001', 'name': 'болт гайка', 'price': 200, 'quantity': 10},
         {'id': '0002', 'name': 'болт гайка шайба', 'price': 100, 'quantity': 10}],
        algorithms=algorithms, min_stock=3, parallel=parallel)


def test_voting_unique_leader_and_parallel_sequential_equivalence():
    algorithms = ['exact_normalized', 'token_set', 'jaccard']
    parallel = _search(algorithms)
    sequential = _search(algorithms, parallel=False)
    position = parallel['voting']['positions'][0]
    assert position == sequential['voting']['positions'][0]
    assert position['recommended_stock_id'] == '0001'
    assert [candidate['votes'] for candidate in position['candidates']] == [2, 1]
    assert position['candidates'][0]['vote_pct'] == 66.67
    assert position['selected_stock_id'] is None
    assert position['candidates'][0]['stock_price_display'] == '200'
    assert position['candidates'][0]['price_difference_display'] == '100%'
    assert parallel['best_candidate'] == parallel['results_by_algorithm'][0]['best_candidate']


def test_voting_tie_does_not_change_legacy_primary_candidate():
    result = _search(['exact_normalized', 'token_set'])
    position = result['voting']['positions'][0]
    assert position['is_tie']
    assert position['recommended_stock_id'] is None
    assert position['selected_stock_id'] is None
    assert position['leader_stock_ids'] == ['0001', '0002']
    assert result['best_candidate']['stock_id'] == '0001'


def test_stock_filter_is_applied_before_voting():
    result = match([{'name': 'болт гайка'}],
        [{'id': 'low', 'name': 'болт гайка', 'quantity': 1},
         {'id': 'ready', 'name': 'болт гайка шайба', 'quantity': 10}],
        algorithms=['exact_normalized', 'token_set'], min_stock=3)
    position = result['voting']['positions'][0]
    assert position['voting_algorithms'] == 1
    assert position['recommended_stock_id'] == 'ready'
    assert result['results_by_algorithm'][0]['low_stock_candidate']['stock_id'] == 'low'
