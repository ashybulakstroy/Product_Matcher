"""Persistent registry of 1C exchange tasks.

A task is the binding between an open 1C invoice form (``request_id``) and a
normal №16 job. It survives server restarts: every change is written to
``<root>/<task_id>/task.json`` atomically under a process-wide lock. The
registry stays process-local by design; the server must run as one process.
"""
from __future__ import annotations

import json
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

TASK_ID = re.compile(r"^[a-f0-9]{32}$")
JOB_ID = re.compile(r"^[a-f0-9]{32}$")
TASK_SCHEMA = 1
STATE_WAITING_PHOTO = "waiting_photo"
STATE_REVIEW = "review"
STATE_CONFIRMED = "confirmed"
STATE_APPLIED = "applied"


class ExchangeError(ValueError):
    """Bad request for the exchange API (HTTP 400)."""

    status = 400


class ExchangeConflict(ExchangeError):
    """Idempotency or binding conflict (HTTP 409)."""

    status = 409


class ExchangeNotFound(ExchangeError):
    """Unknown task (HTTP 404)."""

    status = 404


@dataclass
class ExchangeTask:
    task_id: str
    database_id: str
    request_id: str
    created_at: float
    document_uuid: str | None = None
    job_id: str | None = None
    result_version: int = 0
    confirmed_at: float | None = None
    confirmed_revision: str | None = None
    input_revision: str | None = None
    stale_reason: str | None = None
    result: dict = field(default_factory=dict)
    receipts: list[dict] = field(default_factory=list)

    @property
    def applied(self) -> dict | None:
        return next((receipt for receipt in reversed(self.receipts)
                     if receipt.get("status") == "applied"), None)

    @property
    def state(self) -> str:
        if self.applied is not None:
            return STATE_APPLIED
        if self.stale_reason:
            return STATE_REVIEW
        if self.confirmed_at is not None and self.result.get("checksum"):
            return STATE_CONFIRMED
        if self.job_id:
            return STATE_REVIEW
        return STATE_WAITING_PHOTO

    def as_dict(self) -> dict:
        return {
            "schema_version": TASK_SCHEMA,
            "task_id": self.task_id,
            "database_id": self.database_id,
            "request_id": self.request_id,
            "document_uuid": self.document_uuid,
            "job_id": self.job_id,
            "created_at": self.created_at,
            "result_version": self.result_version,
            "confirmed_at": self.confirmed_at,
            "confirmed_revision": self.confirmed_revision,
            "input_revision": self.input_revision,
            "stale_reason": self.stale_reason,
            "result": dict(self.result),
            "receipts": [dict(receipt) for receipt in self.receipts],
        }


def _text(value, name: str, limit: int = 128) -> str:
    if not isinstance(value, str):
        raise ExchangeError(f"{name}: требуется строка")
    text = value.strip()
    if not text:
        raise ExchangeError(f"{name}: пустое значение")
    if len(text) > limit:
        raise ExchangeError(f"{name}: слишком длинное значение")
    if any(ord(character) < 32 for character in text):
        raise ExchangeError(f"{name}: недопустимые символы")
    return text


def task_document(task: ExchangeTask, base_url: str = "") -> dict:
    """Public task state; contains no secrets and no photo contents."""
    ui_url = f"/tasks/{task.task_id}/ui"
    if base_url:
        ui_url = base_url.rstrip("/") + ui_url
    applied = task.applied
    return {
        "task_id": task.task_id,
        "database_id": task.database_id,
        "request_id": task.request_id,
        "document_uuid": task.document_uuid,
        "job_id": task.job_id,
        "state": task.state,
        "created_at": task.created_at,
        "result_version": task.result_version,
        "revision": task.result.get("revision"),
        "checksum": task.result.get("checksum"),
        "stale_reason": task.stale_reason,
        "applied": applied is not None,
        "confirmed_by_operator": task.result.get("confirmation_origin") == "operator" and task.confirmed_at is not None,
        "receipts": len(task.receipts),
        "ui_url": ui_url,
    }


class ExchangeStore:
    """Thread-safe task registry with atomic JSON persistence."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.tasks: dict[str, ExchangeTask] = {}
        self.lock = threading.RLock()
        self._load()

    def _load(self) -> None:
        for path in sorted(self.root.glob("*/task.json")):
            if not TASK_ID.fullmatch(path.parent.name):
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("schema_version") != TASK_SCHEMA:
                    continue
                task = ExchangeTask(
                    task_id=path.parent.name,
                    database_id=_text(data["database_id"], "database_id"),
                    request_id=_text(data["request_id"], "request_id"),
                    created_at=float(data["created_at"]),
                    document_uuid=(_text(data["document_uuid"], "document_uuid")
                                   if data.get("document_uuid") else None),
                    job_id=(_text(data["job_id"], "job_id") if data.get("job_id") else None),
                    result_version=int(data.get("result_version") or 0),
                    confirmed_at=(float(data["confirmed_at"])
                                  if data.get("confirmed_at") is not None else None),
                    confirmed_revision=(_text(data["confirmed_revision"], "confirmed_revision")
                                        if data.get("confirmed_revision") else None),
                    input_revision=(_text(data["input_revision"], "input_revision")
                                    if data.get("input_revision") else None),
                    stale_reason=(_text(data["stale_reason"], "stale_reason", 500)
                                  if data.get("stale_reason") else None),
                    result=dict(data.get("result") or {}),
                    receipts=[dict(receipt) for receipt in data.get("receipts") or []],
                )
            except (OSError, ValueError, KeyError, TypeError):
                continue
            self.tasks[task.task_id] = task

    def _save(self, task: ExchangeTask) -> None:
        path = self.root / task.task_id / "task.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(task.as_dict(), ensure_ascii=False, indent=2),
                             encoding="utf-8")
        temporary.replace(path)

    def get(self, task_id: str) -> ExchangeTask:
        if not isinstance(task_id, str) or not TASK_ID.fullmatch(task_id):
            raise ExchangeNotFound("Задание не найдено")
        with self.lock:
            task = self.tasks.get(task_id)
        if task is None:
            raise ExchangeNotFound("Задание не найдено")
        return task

    def create(self, database_id, request_id, document_uuid=None):
        """Idempotently create a task for ``database_id + request_id``.

        Returns ``(task, created)``. Repeating the same pair returns the
        existing task (HTTP 200); a different ``document_uuid`` for the same
        pair is a conflict.
        """
        database_id = _text(database_id, "database_id")
        request_id = _text(request_id, "request_id")
        document_uuid = _text(document_uuid, "document_uuid") if document_uuid else None
        with self.lock:
            for task in self.tasks.values():
                if (task.database_id == database_id
                        and task.request_id.casefold() == request_id.casefold()):
                    if document_uuid and task.document_uuid and \
                            task.document_uuid.casefold() != document_uuid.casefold():
                        raise ExchangeConflict("Задание уже связано с другим документом 1С")
                    if document_uuid and not task.document_uuid:
                        task.document_uuid = document_uuid
                        self._save(task)
                    return task, False
            task = ExchangeTask(task_id=uuid.uuid4().hex, database_id=database_id,
                                request_id=request_id, created_at=time.time(),
                                document_uuid=document_uuid)
            self.tasks[task.task_id] = task
            self._save(task)
            return task, True

    def bind_job(self, task_id: str, job_id: str) -> ExchangeTask:
        if not isinstance(job_id, str) or not JOB_ID.fullmatch(job_id):
            raise ExchangeError("job_id: требуется 32 hex-символа")
        with self.lock:
            task = self.get(task_id)
            if task.job_id and task.job_id != job_id:
                raise ExchangeConflict("У задания уже есть другое фото")
            if task.job_id != job_id:
                task.job_id = job_id
                self._save(task)
            return task

    def for_job(self, job_id: str) -> ExchangeTask | None:
        """The 1C task bound to a normal job, if any (used for its database context)."""
        if not isinstance(job_id, str) or not JOB_ID.fullmatch(job_id):
            return None
        with self.lock:
            return next((task for task in self.tasks.values() if task.job_id == job_id), None)

    def result_path(self, task: ExchangeTask, version: int) -> Path:
        return self.root / task.task_id / f"result-v{version}.json"

    def load_result(self, task: ExchangeTask) -> dict | None:
        path = self.result_path(task, task.result_version)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def confirm(self, task_id: str, revision: str, input_revision: str,
                snapshot_factory, *, confirmed=True) -> tuple[ExchangeTask, bool]:
        """Freeze a new result version or return the matching confirmation.

        ``snapshot_factory(version)`` builds the full JSON snapshot while the
        registry lock is held, so the version and checksum cannot race another
        confirmation of the same task.
        """
        with self.lock:
            task = self.get(task_id)
            expected_status = 'confirmed' if confirmed else 'ready_unconfirmed'
            if (not task.stale_reason and task.result.get('status', 'confirmed') == expected_status
                    and task.confirmed_revision == revision
                    and task.input_revision == input_revision
                    and task.result.get("checksum")):
                return task, False
            if task.applied is not None:
                raise ExchangeConflict("Результат уже применён в 1С")
            version = task.result_version + 1
            snapshot = snapshot_factory(version)
            if snapshot.get("revision") != revision:
                raise ExchangeConflict("Снимок не соответствует подтверждаемой ревизии счёта")
            path = self.result_path(task, version)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
            temporary.replace(path)
            task.result_version = version
            task.confirmed_at = time.time() if confirmed else None
            task.confirmed_revision = revision
            task.input_revision = input_revision
            task.stale_reason = None
            task.result = {
                "version": version,
                "status": expected_status,
                "confirmation_origin": 'operator' if confirmed else None,
                "revision": revision,
                "checksum": snapshot["checksum"],
                "confirmed_at": snapshot.get("confirmed_at"),
                "items": len(snapshot.get("items") or []),
                "total": snapshot.get("total"),
            }
            self._save(task)
            return task, True

    def invalidate_job(self, job_id: str, reason: str) -> ExchangeTask | None:
        """Mark a confirmed result stale after the receipt or links changed."""
        with self.lock:
            task = self.for_job(job_id)
            if task is None or task.applied is not None:
                return None
            if not task.result.get("checksum") or task.stale_reason:
                return None
            task.stale_reason = reason or "Данные счёта изменились; подтвердите результат заново."
            self._save(task)
            return task

    def link_document(self, task_id: str, document_uuid: str) -> tuple[ExchangeTask, bool]:
        """Bind an actually recorded 1C document; repeat of the same link is safe."""
        document_uuid = _text(document_uuid, "document_uuid")
        with self.lock:
            task = self.get(task_id)
            if task.document_uuid:
                if task.document_uuid.casefold() == document_uuid.casefold():
                    return task, False
                raise ExchangeConflict("Задание уже связано с другим документом 1С")
            task.document_uuid = document_uuid
            self._save(task)
            return task, True

    def add_receipt(self, task_id: str, receipt: dict) -> tuple[ExchangeTask, bool]:
        """Idempotently store an ``applied``/``error`` receipt by ``operation_id``.

        The caller validates the receipt against the frozen result. Repeating
        the same operation with the same body returns the stored state; the
        same operation id with another body is a conflict.
        """
        operation_id = _text(receipt.get("operation_id"), "operation_id")
        fingerprint = _text(receipt.get("fingerprint"), "fingerprint", 128)
        status = receipt.get("status")
        if status not in {"applied", "error"}:
            raise ExchangeError("status: ожидается applied или error")
        with self.lock:
            task = self.get(task_id)
            for existing in task.receipts:
                if existing.get("operation_id", "").casefold() != operation_id.casefold():
                    continue
                if existing.get("fingerprint") == fingerprint:
                    return task, False
                raise ExchangeConflict(
                    "Квитанция с этим operation_id уже получена с другим телом")
            if status == "applied" and task.applied is not None:
                previous = str(task.applied.get("operation_id") or "")
                if previous.casefold() != operation_id.casefold():
                    raise ExchangeConflict(
                        "Задание уже применено в 1С другой операцией записи")
            stored = dict(receipt)
            stored["operation_id"] = operation_id
            stored["fingerprint"] = fingerprint
            stored["status"] = status
            if status == "applied":
                document_uuid = _text(receipt.get("document_uuid"), "document_uuid")
                if task.document_uuid and \
                        task.document_uuid.casefold() != document_uuid.casefold():
                    raise ExchangeConflict("Квитанция относится к другому документу 1С")
                task.document_uuid = document_uuid
                stored["document_uuid"] = document_uuid
            task.receipts.append(stored)
            self._save(task)
            return task, True
