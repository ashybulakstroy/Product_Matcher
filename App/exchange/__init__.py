"""HTTP exchange layer between the №16 web server and 1C.

The package does not duplicate OCR, invoice generation or mapping rules: it
binds an open 1C form (``request_id``) to a normal №16 job, freezes confirmed
invoice snapshots and serves the 1.1 contract to the 1C extension.
"""
from App.exchange.store import (
    ExchangeConflict,
    ExchangeError,
    ExchangeNotFound,
    ExchangeStore,
    ExchangeTask,
    task_document,
)

__all__ = [
    "ExchangeConflict",
    "ExchangeError",
    "ExchangeNotFound",
    "ExchangeStore",
    "ExchangeTask",
    "task_document",
]
