"""Bounded thread-safe result cache for complete candidate-list links."""
import threading
import time
import uuid
from collections import OrderedDict

_RESULT_CACHE = OrderedDict()
_RESULT_CACHE_LOCK = threading.Lock()
_RESULT_CACHE_TTL_SECONDS = 3600
_RESULT_CACHE_MAX_ENTRIES = 32


def _store_full_result(result: dict) -> str:
    result_id = uuid.uuid4().hex
    stored_at = time.time()
    with _RESULT_CACHE_LOCK:
        _RESULT_CACHE[result_id] = (stored_at, result)
        while len(_RESULT_CACHE) > _RESULT_CACHE_MAX_ENTRIES:
            _RESULT_CACHE.popitem(last=False)
    return result_id


def _load_full_result(result_id: str):
    if not result_id:
        return None
    now = time.time()
    with _RESULT_CACHE_LOCK:
        item = _RESULT_CACHE.get(result_id)
        if item is None:
            return None
        stored_at, result = item
        if now - stored_at > _RESULT_CACHE_TTL_SECONDS:
            _RESULT_CACHE.pop(result_id, None)
            return None
        _RESULT_CACHE.move_to_end(result_id)
        return result


def clear_full_result_cache() -> None:
    with _RESULT_CACHE_LOCK:
        _RESULT_CACHE.clear()
