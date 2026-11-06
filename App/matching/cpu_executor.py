"""Persistent CPU processes plus a shared-model thread executor."""
import atexit
import multiprocessing
import os
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from App.matching.worker import run_algorithm

_lock = threading.Lock()
_cpu_pool = None
_model_pool = None
MAX_CPU_WORKERS = min(4, os.cpu_count() or 1)


def submit_algorithm(spec, checked, stock, top_n):
    global _cpu_pool, _model_pool
    with _lock:
        if spec.execution == 'shared_model':
            # E5 uses ONNX's own CPU threads and a single shared in-memory model.
            if _model_pool is None:
                _model_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='matching-model')
            pool, executor = _model_pool, 'shared_model_thread'
        else:
            if _cpu_pool is None:
                _cpu_pool = ProcessPoolExecutor(max_workers=MAX_CPU_WORKERS,
                    mp_context=multiprocessing.get_context('spawn'))
            pool, executor = _cpu_pool, 'cpu_process'
        return pool.submit(run_algorithm, spec, checked, stock, top_n, executor)


def shutdown_executors():
    global _cpu_pool, _model_pool
    with _lock:
        pools = (_cpu_pool, _model_pool)
        _cpu_pool = _model_pool = None
    for pool in pools:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)


atexit.register(shutdown_executors)
