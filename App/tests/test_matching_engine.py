import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor

import pytest

from App.matching import engine
from App.matching.cache import _load_full_result
from App.matching.registry import ALGORITHMS, get_algorithm
from App.matching.worker import run_algorithm

CHECKED = [{'id': 'q', 'name': 'Болт М8 20 мм', 'price': 100}]
STOCK = [
    {'id': 'a', 'name': 'Болт М8 20 мм', 'price': 200, 'quantity': 10},
    {'id': 'b', 'name': 'Болт М8 20 мм', 'price': 101, 'quantity': 10},
    {'id': 'c', 'name': 'Болт М8 20 мм', 'price': 100, 'quantity': 1},
]


def _without_runtime(result):
    return [{key: value for key, value in row.items()
             if key not in {'elapsed_ms', 'worker_pid', 'executor', 'full_results_url'}}
            for row in result['results_by_algorithm']]


def test_parallel_cpu_processes_preserve_results_price_stock_and_selection_order():
    selected = ['sql_str_3', 'copilot_1', 'opencode_4', 'sql_str_2']
    parallel = engine.match(CHECKED, STOCK, algorithms=selected, min_stock=3)
    sequential = engine.match(CHECKED, STOCK, algorithms=selected, min_stock=3, parallel=False)
    assert _without_runtime(parallel) == _without_runtime(sequential)
    assert parallel['execution_mode'] == 'parallel'
    assert sequential['execution_mode'] == 'sequential'
    assert [row['algorithm_id'] for row in parallel['results_by_algorithm']] == selected
    assert all(row['worker_pid'] != os.getpid() for row in parallel['results_by_algorithm'])
    assert all(row['executor'] == 'cpu_process' for row in parallel['results_by_algorithm'])
    assert all(row['best_candidate']['stock_id'] == 'b' for row in parallel['results_by_algorithm'])
    assert all(row['low_stock_candidate']['stock_id'] == 'c' for row in parallel['results_by_algorithm'])


def test_all_tasks_are_submitted_before_waiting(monkeypatch):
    barrier = threading.Barrier(2)
    with ThreadPoolExecutor(max_workers=2) as pool:
        def submit(spec, checked, stock, top_n):
            def work():
                # Sequential submit-and-wait would break the barrier.
                barrier.wait(timeout=3)
                return run_algorithm(spec, checked, stock, top_n)
            return pool.submit(work)
        monkeypatch.setattr(engine, 'submit_algorithm', submit)
        result = engine.match(CHECKED, STOCK, algorithms=['copilot_1', 'sql_str_2'])
        assert all(row['status'] == 'ok' for row in result['results_by_algorithm'])


def test_worker_failure_does_not_discard_other_results(monkeypatch):
    def submit(spec, checked, stock, top_n):
        future = Future()
        if spec.id == 'sql_str_2':
            future.set_exception(RuntimeError('worker stopped'))
        else:
            future.set_result(run_algorithm(spec, checked, stock, top_n))
        return future
    monkeypatch.setattr(engine, 'submit_algorithm', submit)
    result = engine.match(CHECKED, STOCK, algorithms=['copilot_1', 'sql_str_2'], min_stock=3)
    assert result['best_candidate']['stock_id'] == 'b'
    assert result['results_by_algorithm'][1]['status'] == 'error'


def test_full_result_keeps_per_algorithm_timing():
    from urllib.parse import parse_qs, urlparse
    result = engine.match(CHECKED, STOCK)
    algorithm = result['results_by_algorithm'][0]
    result_id = parse_qs(urlparse(algorithm['full_results_url']).query)['result_id'][0]
    saved = _load_full_result(result_id)
    assert saved['elapsed_ms'] == algorithm['elapsed_ms']
    assert len(saved['items'][0]['candidates']) == 3


@pytest.mark.parametrize('kwargs', [
    {'algorithm_id': 'does_not_exist'}, {'parallel': 'false'}, {'top_n': 0}, {'min_stock': -1},
])
def test_invalid_request_is_rejected_before_execution(kwargs):
    with pytest.raises(ValueError):
        engine.match(CHECKED, STOCK, **kwargs)


def test_registry_uses_only_matching_package_implementations():
    assert {'copilot_1', 'sql_str_2', 'sql_str_3', 'opencode_4', 'semantic_5'} <= ALGORITHMS.keys()
    for spec in ALGORITHMS.values():
        assert spec.module.startswith('App.matching.algorithms.')
        assert get_algorithm(spec.id) == spec
