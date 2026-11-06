"""Run registered algorithms, preserving their scores and candidate ordering rules."""
import logging
from time import perf_counter
from urllib.parse import quote

from App.matching.cache import _store_full_result
from App.matching.contracts import AlgorithmRun, extract_query
from App.matching.cpu_executor import submit_algorithm
from App.matching.ranking import rank_result
from App.matching.registry import get_algorithm, select_algorithms
from App.matching.worker import run_algorithm


def _complete(spec, run, checked, stock, top_n, min_stock):
    timing = dict(elapsed_ms=run.elapsed_ms, worker_pid=run.worker_pid, executor=run.executor)
    if run.status != 'ok':
        return dict(status=run.status, algorithm_id=spec.id, algorithm_name=spec.name,
                    error=run.error, items=[], best_candidate=None, low_stock_candidate=None, **timing)
    result, full_result = rank_result(run.items, checked, stock, spec, top_n, min_stock)
    result.update(timing)
    full_result.update(timing)
    result_id = _store_full_result(full_result)
    result['full_results_url'] = f'/analog/list?algorithm_id={quote(spec.id)}&result_id={quote(result_id)}'
    return result


def build_algorithm_result(checked_products, stock_products, top_n=5, min_stock=0, algorithm_id='copilot_1'):
    """Single-algorithm entry point retained for existing consumers/tests."""
    checked, stock = list(checked_products), list(stock_products)
    spec = get_algorithm(algorithm_id)
    run = run_algorithm(spec, checked, stock, max(top_n, len(stock)))
    return _complete(spec, run, checked, stock, top_n, min_stock)


def match(checked_products, stock_products, top_n=5, min_stock=0,
          algorithm_id='copilot_1', algorithms=None, parallel=True):
    started = perf_counter()
    if not isinstance(parallel, bool):
        raise ValueError('parallel должен быть true или false')
    if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 1:
        raise ValueError('Top X должен быть положительным целым числом')
    if min_stock < 0:
        raise ValueError('Минимальный остаток не может быть отрицательным')
    specs = select_algorithms(algorithm_id, algorithms)
    checked, stock = list(checked_products), list(stock_products)
    limit = max(top_n, len(stock))
    results = []
    if parallel:
        # Submit every task before waiting. Gather in registry/selection order,
        # so worker completion order never changes the primary algorithm.
        futures = [submit_algorithm(spec, checked, stock, limit) for spec in specs]
        runs = []
        for spec, future in zip(specs, futures):
            try:
                runs.append(future.result())
            except Exception:
                logging.exception('Matching worker failed: %s', spec.id)
                runs.append(AlgorithmRun(status='error', error='Ошибка процесса алгоритма. Проверьте журнал сервера.'))
    else:
        runs = [run_algorithm(spec, checked, stock, limit) for spec in specs]
    for spec, run in zip(specs, runs):
        results.append(_complete(spec, run, checked, stock, top_n, min_stock))
    primary = results[0]
    return dict(status='ok', top_n=top_n, algorithm_id=primary['algorithm_id'],
                algorithm_name=primary['algorithm_name'], query=extract_query(checked),
                best_candidate=primary.get('best_candidate'), items=primary.get('items', []),
                low_stock_candidate=primary.get('low_stock_candidate'), results_by_algorithm=results,
                execution_mode='parallel' if parallel else 'sequential',
                elapsed_ms=round((perf_counter() - started) * 1000, 2))
