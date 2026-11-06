"""Picklable entry point for Windows spawn workers and the model thread."""
import importlib
import logging
import os
from time import perf_counter

from App.matching.contracts import AlgorithmRun


def run_algorithm(spec, checked, stock, top_n, executor='inline'):
    started = perf_counter()
    try:
        algorithm = getattr(importlib.import_module(spec.module), spec.function)
        items = algorithm(checked, stock, top_n=top_n)
        result = AlgorithmRun(items=items)
    except Exception as error:
        # Failures are isolated per algorithm, with the reason visible in the UI.
        logging.exception('Matching algorithm %s failed', spec.id)
        from App.matching.algorithms.semantic.model import SemanticUnavailable
        if isinstance(error, SemanticUnavailable):
            result = AlgorithmRun(status='unavailable', error=str(error))
        else:
            result = AlgorithmRun(status='error', error='Алгоритм завершился с ошибкой. Подробности в журнале сервера.')
    result.elapsed_ms = round((perf_counter() - started) * 1000, 2)
    result.worker_pid = os.getpid()
    result.executor = executor
    return result
