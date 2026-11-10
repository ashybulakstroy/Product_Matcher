r"""Local receipt workspace; binding defaults are OCR_WEB_HOST/OCR_WEB_PORT in .env."""
from __future__ import annotations

import argparse
import copy
import json
import logging
import mimetypes
import re
import sys
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from time import perf_counter, time
from urllib.parse import parse_qs, unquote, urlparse

import cv2
from pydantic import ValidationError

from App.ocr.ocr_catalog_eval import PASS_SIMILARITY, ROOT
from App.exchange import (ExchangeConflict, ExchangeError, ExchangeNotFound, ExchangeStore,
                          contract, task_document)
from App.exchange import multipart
from App.ocr.web_pipeline import ReceiptPipeline
from App.ocr.receipt_editing import apply_edit, save_receipt, update_receipt
from App.server_config import ServerSettings, server_settings
from App.invoicing import invoice_service, catalog_1c, stock_sync

STATIC = Path(__file__).parent / "web"
JOB_ID = re.compile(r"^[a-f0-9]{32}$")
DOWNLOADS = {"receipt.json", "receipt.csv", "ocr_rectified.txt", "ocr_rectified.json", "ocr_header.json",
             "layout.json", "rectification.json", "rectification_map.npz", "quantity_cells.json", "numeric_split.json", "row_split.json", "upload.bin"}


class Job:
    def __init__(self, job_id, filename, directory):
        self.id, self.filename, self.directory = job_id, filename, directory
        self.status, self.progress, self.error = "queued", 0, ""
        self.result, self.artifacts, self.events = None, [], []
        self.created_at = time()
        self.condition = threading.Condition()
        self.active_stage = None
        self.active_stage_title = ""
        self.active_stage_started = None
        self.stage_sequence = 0

    def public(self):
        return dict(id=self.id, filename=self.filename, status=self.status, progress=self.progress,
                    error=self.error, result=self.result, artifacts=self.artifacts, created_at=self.created_at,
                    events=copy.deepcopy(self.events), last_event_id=len(self.events))

    def publish(self, event_type, **payload):
        with self.condition:
            if "progress" in payload:
                self.progress = max(self.progress, int(payload["progress"]))
                payload["progress"] = self.progress
            event = dict(type=event_type, id=len(self.events)+1, at=time(), **copy.deepcopy(payload))
            self.events.append(event)
            self.condition.notify_all()

    def phase(self, phase, progress, title):
        self.finish_stage("done")
        self.stage_sequence += 1
        self.active_stage = phase
        self.active_stage_title = title
        self.active_stage_started = perf_counter()
        effect = {"input":"decode","geometry":"perspective","layout":"grid","ocr":"scan",
                  "numeric":"numbers","matching":"table","assemble":"table","queued":"upload"}.get(phase,"")
        self.publish("stage", stage=phase, stage_id=self.stage_sequence, label=title,
                     effect=effect, status="running", elapsed_ms=None, progress=progress)
        self.publish("phase", phase=phase, progress=progress, title=title)

    def finish_stage(self, status="done", detail=""):
        if self.active_stage is None:
            return
        elapsed = round((perf_counter()-self.active_stage_started)*1000, 1)
        self.publish("stage", stage=self.active_stage, stage_id=self.stage_sequence,
                     label=self.active_stage_title, status=status, detail=detail,
                     elapsed_ms=elapsed, progress=self.progress)
        self.active_stage = None
        self.active_stage_title = ""
        self.active_stage_started = None

    def json_file(self, name, data):
        (self.directory / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def image(self, key, image, title, subtitle="", role="derived"):
        path = self.directory / f"{key}.png"
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise ValueError("Не удалось сохранить промежуточное изображение.")
        path.write_bytes(encoded.tobytes())
        asset = dict(key=key, filename=path.name, url=f"/api/jobs/{self.id}/files/{path.name}",
                     width=image.shape[1], height=image.shape[0], title=title, subtitle=subtitle, role=role)
        with self.condition:
            self.artifacts.append(asset)
        self.publish("artifact", artifact=asset)

    def persist(self):
        self.json_file("state.json", self.public())


class JobStore:
    def __init__(self, root, processor=None, max_upload=20*1024*1024):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_upload = max_upload
        self.jobs, self.lock = {}, threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="receipt")
        self.pipeline = ReceiptPipeline()
        self.processor = processor or self.pipeline.process
        # Completed results survive a server restart. Interrupted work is explicit.
        for state_file in sorted(self.root.glob("*/state.json"), key=lambda p:p.stat().st_mtime)[-100:]:
            if not JOB_ID.fullmatch(state_file.parent.name):
                continue
            try:
                state = json.loads(state_file.read_text(encoding="utf-8"))
                job = Job(state_file.parent.name, state["filename"], state_file.parent)
                for key in ("status", "progress", "error", "result", "artifacts", "created_at", "events"):
                    if key in state:
                        setattr(job, key, state[key])
                if job.status not in {"done", "failed"}:
                    job.status, job.error = "failed", "Обработка прервана перезапуском сервера. Загрузите фото ещё раз."
                if job.status == "done" and job.result and "rows" in job.result:
                    update_receipt(job.result)
                    save_receipt(job.directory, job.result)
                job.publish("complete" if job.status == "done" else "error", result=job.result, title=job.error, progress=job.progress)
                self.jobs[job.id] = job
            except (ValueError, KeyError, OSError):
                continue

    def submit(self, data, filename):
        job_id = uuid.uuid4().hex
        filename = re.sub(r"[\x00-\x1f]", "", filename.replace("\\", "/").split("/")[-1])[:200] or "receipt.jpg"
        directory = self.root / job_id
        directory.mkdir()
        (directory / "upload.bin").write_bytes(data)
        job = Job(job_id, filename, directory)
        job.persist()
        with self.lock:
            self.jobs[job_id] = job
        job.phase("queued", 1, "Фото загружено. Ожидаем обработку")
        self.pool.submit(self._process, job)
        return job

    def _process(self, job):
        job.status = "processing"
        try:
            result = self.processor(job)
            with job.condition:
                job.finish_stage("done")
                job.result, job.status = result, "done"
                job.publish("stage", stage="completed", stage_id=job.stage_sequence+1,
                            label="Электронный чек готов", effect="done", status="done",
                            elapsed_ms=0, progress=100)
                job.publish("complete", result=result, progress=100, title="Электронный чек готов")
        except Exception as error:
            (job.directory / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
            job.status = "failed"
            job.error = str(error) if isinstance(error, ValueError) else "Не удалось завершить обработку. Попробуйте другое фото."
            job.finish_stage("error", job.error)
            job.publish("error", title=job.error, progress=job.progress)
            traceback.print_exc()
        finally:
            job.persist()

    def get(self, job_id):
        with self.lock:
            return self.jobs.get(job_id) if JOB_ID.fullmatch(job_id) else None


class ReceiptHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def create_server(host=None, port=None, output_dir=None, processor=None, max_upload=20*1024*1024):
    if host is None or port is None:
        settings = server_settings()
        host = settings.host if host is None else host
        port = settings.port if port is None else port
    root_dir = Path(output_dir) if output_dir is not None else ROOT / "web_outputs"
    store = JobStore(root_dir, processor, max_upload)
    exchange = ExchangeStore(root_dir / "exchange")
    stock = stock_sync.StockStore(catalog_1c.CATALOG_DIR)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def send_data(self, data, content_type, status=200, attachment=None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-cache")
            if attachment:
                self.send_header("Content-Disposition", f'attachment; filename="{attachment}"')
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def send_json(self, data, status=200):
            self.send_data(json.dumps(data, ensure_ascii=False).encode(), "application/json; charset=utf-8", status)

        def base_url(self):
            host = (self.headers.get("Host") or "").strip()
            if not host or len(host) > 200:
                host = f"127.0.0.1:{self.server.server_port}"
            return "http://" + host

        def stock_status(self, query):
            try:
                params=parse_qs(query)
                ttl,_=stock_sync.settings()
                result=stock.status((params.get('organization_id') or [''])[0],
                                    (params.get('warehouse_id') or [''])[0],ttl,
                                    (params.get('database_id') or [None])[0])
                return self.send_json(result)
            except stock_sync.StockSyncError as error:
                return self.send_json(error.public(),error.status)
            except OSError:
                return self.send_json(dict(code='read_failed',message='Не удалось прочитать остатки.'),500)

        def stock_summary(self):
            # Operator page only: slice date and item count, never TTL or ids.
            try:
                ttl, _ = stock_sync.settings()
                now = datetime.now(timezone.utc)
                snapshots = []
                for path in stock_sync.paths(catalog_1c.CATALOG_DIR):
                    data = stock_sync.read(path)
                    age = (now - stock_sync.as_of(data['stock_as_of'])).total_seconds()
                    snapshots.append(dict(
                        database_id=data['database_id'],
                        organization_name=data['organization']['name'],
                        warehouse_name=data['warehouse']['name'],
                        stock_as_of=data['stock_as_of'],
                        record_count=data['record_count'],
                        status='fresh' if age < ttl else 'expired'))
                snapshots.sort(key=lambda item: stock_sync.as_of(item['stock_as_of']), reverse=True)
                return self.send_json(dict(snapshots=snapshots))
            except stock_sync.StockSyncError as error:
                return self.send_json(error.public(), error.status)
            except OSError:
                return self.send_json(dict(code='read_failed', message='Не удалось прочитать остатки.'), 500)

        def stock_upload(self):
            # User clarification: keys are deliberately ignored, including X-API-Key.
            try:
                _,limit=stock_sync.settings()
                if self.headers.get('Transfer-Encoding'):
                    raise stock_sync.StockSyncError('invalid_request','Передавайте один JSON с Content-Length.',400)
                try:
                    length=int(self.headers.get('Content-Length',''))
                except ValueError:
                    raise stock_sync.StockSyncError('invalid_length','Требуется Content-Length.',400)
                if length<=0:
                    raise stock_sync.StockSyncError('empty_body','Требуется полный JSON снимка.',400)
                if length>limit:
                    raise stock_sync.StockSyncError('too_large','Снимок превышает допустимый размер.',413)
                if self.headers.get('Content-Type','').split(';')[0].strip().lower()!='application/json':
                    raise stock_sync.StockSyncError('invalid_content_type','Требуется application/json UTF-8.',400)
                raw=self.rfile.read(length)
                if len(raw)!=length:
                    raise stock_sync.StockSyncError('incomplete_body','Тело загружено не полностью.',400)
                result=stock.publish(raw,self.headers.get('X-Content-SHA256'),catalog_1c.invalidate_stock_cache)
                return self.send_json(result)
            except stock_sync.StockSyncError as error:
                self.close_connection=True
                self.log_message('stock-sync отклонён: %s (%s); байт=%s, X-Content-SHA256=%s',
                                 error.code, error.message, self.headers.get('Content-Length'),
                                 'есть' if self.headers.get('X-Content-SHA256') else 'нет')
                return self.send_json(error.public(),error.status)
            except OSError:
                self.close_connection=True
                return self.send_json(dict(code='write_failed',message='Не удалось записать остатки.'),500)

        def exchange_create(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            if length <= 0:
                return self.send_json(dict(error="Пустой запрос задания"), 400)
            if length > 65536:
                self.close_connection = True
                return self.send_json(dict(error="Слишком большой запрос задания"), 413)
            data = self.rfile.read(length)
            if len(data) != length:
                return self.send_json(dict(error="Запрос задания загружен не полностью"), 400)
            try:
                payload = json.loads(data)
                if not isinstance(payload, dict):
                    raise ExchangeError("Тело задания должно быть JSON-объектом")
                task, created = exchange.create(payload.get("database_id"),
                                                payload.get("request_id"),
                                                payload.get("document_uuid"))
            except ExchangeConflict as error:
                return self.send_json(dict(error=str(error)), 409)
            except ExchangeError as error:
                return self.send_json(dict(error=str(error)), 400)
            except (ValueError, TypeError):
                return self.send_json(dict(error="Тело задания должно быть корректным JSON"), 400)
            return self.send_json(task_document(task, self.base_url()), 201 if created else 200)

        def exchange_photo(self, task_id):
            try:
                task = exchange.get(task_id)
            except ExchangeNotFound as error:
                self.close_connection = True  # the request body stays unread
                return self.send_json(dict(error=str(error)), 404)
            if task.job_id:
                self.close_connection = True
                return self.send_json(dict(error="У задания уже есть фото. Новое фото принимается только для нового задания из 1С."), 409)
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            if length <= 0:
                return self.send_json(dict(error="Пустая загрузка фото"), 400)
            if length > store.max_upload + 65536:
                self.close_connection = True
                return self.send_json(dict(error=f"Выберите изображение размером до {store.max_upload//1024//1024} МБ."), 413)
            data = self.rfile.read(length)
            if len(data) != length:
                return self.send_json(dict(error="Фото загружено не полностью"), 400)
            try:
                filename, photo = multipart.file_part(self.headers.get("Content-Type"), data)
            except multipart.MultipartError as error:
                return self.send_json(dict(error=str(error)), 400)
            if len(photo) > store.max_upload:
                self.close_connection = True
                return self.send_json(dict(error=f"Выберите изображение размером до {store.max_upload//1024//1024} МБ."), 413)
            job = store.submit(photo, filename)
            try:
                task = exchange.bind_job(task_id, job.id)
            except ExchangeConflict as error:
                return self.send_json(dict(error=str(error)), 409)
            return self.send_json(task_document(task, self.base_url()), 202)

        def invoice_context(self, job):
            """Database context of a 1C task bound to the job, if any."""
            task = exchange.for_job(job.id)
            return task.database_id if task else None

        def exchange_confirm(self, task_id):
            try:
                task = exchange.get(task_id)
            except ExchangeNotFound as error:
                self.close_connection = True
                return self.send_json(dict(error=str(error)), 404)
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            if length < 0:
                length = 0
            if length > 65536:
                self.close_connection = True
                return self.send_json(dict(error="Слишком большой запрос подтверждения"), 413)
            data = self.rfile.read(length) if length else b"{}"
            if len(data) != length:
                return self.send_json(dict(error="Запрос подтверждения загружен не полностью"), 400)
            try:
                payload = json.loads(data or b"{}")
                if not isinstance(payload, dict):
                    raise ValueError("not an object")
            except (ValueError, TypeError):
                return self.send_json(dict(error="Тело подтверждения должно быть JSON-объектом"), 400)
            revision = payload.get("revision")
            if revision is not None and not isinstance(revision, str):
                return self.send_json(dict(error="revision: требуется строка"), 400)
            for key, expected in (("database_id", task.database_id),
                                  ("request_id", task.request_id)):
                value = payload.get(key)
                if value not in (None, "") and str(value).strip().casefold() != expected.casefold():
                    return self.send_json(dict(error=f"Задание относится к другой базе/форме ({key})"), 409)
            if task.applied is not None:
                return self.send_json(dict(
                    error="Результат уже применён в 1С; повторное подтверждение недоступно.",
                    state=task.state), 409)
            if not task.job_id:
                return self.send_json(dict(error="Сначала загрузите фото в задание."), 409)
            job = store.get(task.job_id)
            if job is None:
                return self.send_json(dict(error="Задание обработки не найдено. Загрузите фото заново."), 409)
            with job.condition, catalog_1c.catalog_scope(self.invoice_context(job)):
                if job.status != "done" or not job.result:
                    return self.send_json(dict(error="Дождитесь завершения распознавания и проверки чека."), 409)
                receipt = job.result
                selections = invoice_service.load_selections(job)
                try:
                    model = invoice_service.invoice_model(receipt, selections=selections,
                                                          database_id=self.invoice_context(job))
                except ValueError as error:
                    return self.send_json(dict(error=str(error)), 409)
                except Exception:
                    return self.send_json(dict(error="Счёт не построен: внутренняя ошибка расчёта."), 500)
                reasons = invoice_service.exchange_reasons(model)
                if reasons:
                    return self.send_json(dict(error="Счёт ещё не готов для 1С.", reasons=reasons), 409)
                if revision and revision != model.get("revision"):
                    return self.send_json(dict(
                        error="Счёт изменился после загрузки страницы. Обновите страницу и подтвердите заново.",
                        reasons=["Ревизия счёта не совпадает с подтверждаемой."]), 409)
                try:
                    fingerprint = contract.input_revision(receipt, selections, database_id=self.invoice_context(job))
                    current = task
                    task, created = exchange.confirm(
                        task.task_id, model["revision"], fingerprint,
                        lambda version: contract.build_snapshot(current, model, receipt, version))
                except ValueError as error:
                    return self.send_json(dict(error=str(error)), 409)
                except ExchangeConflict as error:
                    return self.send_json(dict(error=str(error)), 409)
                except OSError:
                    return self.send_json(dict(error="Не удалось сохранить снимок результата."), 500)
            document = task_document(task, self.base_url())
            document["created"] = created
            return self.send_json(document)

        def result_snapshot(self, task_id, query):
            """Shared checks for JSON/XML results: ``(status, payload)``."""
            try:
                task = exchange.get(task_id)
            except ExchangeNotFound as error:
                return 404, dict(error=str(error))
            params = parse_qs(query)
            for key, expected in (("database_id", task.database_id),
                                  ("request_id", task.request_id)):
                value = (params.get(key) or [""])[0].strip()
                if value and value.casefold() != expected.casefold():
                    return 409, dict(
                        error="Задание относится к другой базе/форме 1С. Форму не менять.")
            allow_unconfirmed = (params.get('allow_unconfirmed') or [''])[0] == '1'
            if task.applied is None:
                job = store.get(task.job_id) if task.job_id else None
                if job is None:
                    return 202, dict(error="Счёт ещё не сформирован.", state=task.state)
                with job.condition, catalog_1c.catalog_scope(self.invoice_context(job)):
                    if job.status != "done" or not job.result:
                        return 202, dict(error="Счёт ещё формируется или требует исправления.", state=task.state)
                    try:
                        selections = invoice_service.load_selections(job)
                        fingerprint = contract.input_revision(job.result, selections, database_id=self.invoice_context(job))
                        current_confirmation = (task.confirmed_at is not None and not task.stale_reason
                            and task.input_revision == fingerprint
                            and task.result.get('confirmation_origin') == 'operator')
                        if not current_confirmation:
                            if task.input_revision and task.input_revision != fingerprint:
                                exchange.invalidate_job(job.id, 'Счёт изменён; требуется новое веб-подтверждение для автоимпорта.')
                            if not allow_unconfirmed:
                                return 202, dict(error="Ожидается веб-подтверждение для автоматического импорта.", state=task.state)
                            model = invoice_service.invoice_model(job.result, selections=selections,
                                                                  database_id=self.invoice_context(job))
                            reasons = invoice_service.exchange_reasons(model)
                            if reasons:
                                return 202, dict(error="Черновик не завершён.", reasons=reasons, state=task.state)
                            current = task
                            task, _ = exchange.confirm(
                                task.task_id, model['revision'], fingerprint,
                                lambda version: contract.build_snapshot(current, model, job.result, version,
                                                                        status='ready_unconfirmed'),
                                confirmed=False)
                    except (ValueError, ExchangeConflict) as error:
                        return 409, dict(error=str(error))
                    except OSError:
                        return 500, dict(error="Не удалось сохранить результат счёта.")
            snapshot = exchange.load_result(task)
            if (snapshot is None or snapshot.get("checksum") != task.result.get("checksum")
                    or contract.checksum(snapshot) != snapshot.get("checksum")):
                return 409, dict(
                    error="Снимок результата не найден или повреждён. Подтвердите результат заново.")
            return 200, snapshot

        def exchange_result(self, task_id, query):
            status, payload = self.result_snapshot(task_id, query)
            return self.send_json(payload, status)

        def exchange_result_xml(self, task_id, query):
            status, payload = self.result_snapshot(task_id, query)
            if status != 200:
                return self.send_json(payload, status)
            return self.send_data(contract.to_xml(payload), "application/xml; charset=utf-8")

        def read_small_json(self, limit=65536, label="запроса"):
            """Read a small JSON object; returns ``(payload, (status, error))``."""
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            if length < 0:
                length = 0
            if length > limit:
                self.close_connection = True
                return None, (413, dict(error=f"Слишком большой {label}"))
            data = self.rfile.read(length) if length else b"{}"
            if len(data) != length:
                return None, (400, dict(error=f"{label.capitalize()} загружен не полностью"))
            try:
                payload = json.loads(data or b"{}")
            except (ValueError, TypeError):
                return None, (400, dict(error=f"{label.capitalize()} должен быть корректным JSON"))
            if not isinstance(payload, dict):
                return None, (400, dict(error=f"{label.capitalize()} должен быть JSON-объектом"))
            return payload, None

        def exchange_document_link(self, task_id):
            payload, error = self.read_small_json(label="запроса привязки документа")
            if error:
                return self.send_json(error[1], error[0])
            try:
                task = exchange.get(task_id)
            except ExchangeNotFound as not_found:
                return self.send_json(dict(error=str(not_found)), 404)
            for key, expected in (("database_id", task.database_id),
                                  ("request_id", task.request_id)):
                value = payload.get(key)
                if value not in (None, "") and str(value).strip().casefold() != expected.casefold():
                    return self.send_json(dict(error=f"Задание относится к другой базе/форме ({key})"), 409)
            document_uuid = payload.get("document_uuid")
            if not isinstance(document_uuid, str) or not document_uuid.strip():
                return self.send_json(dict(error="document_uuid: требуется идентификатор документа 1С"), 400)
            try:
                task, linked = exchange.link_document(task_id, document_uuid)
            except ExchangeConflict as conflict:
                return self.send_json(dict(error=str(conflict)), 409)
            except ExchangeError as invalid:
                return self.send_json(dict(error=str(invalid)), 400)
            document = task_document(task, self.base_url())
            document["linked"] = linked
            return self.send_json(document)

        def exchange_import_receipt(self, task_id):
            payload, error = self.read_small_json(262144, "запроса квитанции")
            if error:
                return self.send_json(error[1], error[0])
            try:
                task = exchange.get(task_id)
            except ExchangeNotFound as not_found:
                return self.send_json(dict(error=str(not_found)), 404)
            database = payload.get("database_id")
            request = payload.get("request_id")
            operation = payload.get("operation_id")
            if not all(isinstance(value, str) and value.strip()
                       for value in (database, request, operation)):
                return self.send_json(dict(
                    error="database_id, request_id и operation_id обязательны"), 400)
            if database.strip().casefold() != task.database_id.casefold() or \
                    request.strip().casefold() != task.request_id.casefold():
                return self.send_json(dict(error="Квитанция относится к другой базе/форме 1С"), 409)
            status = payload.get("status")
            if status not in {"applied", "error"}:
                return self.send_json(dict(error="status: ожидается applied или error"), 400)
            version = payload.get("result_version")
            checksum = payload.get("checksum")
            document_uuid = payload.get("document_uuid")
            if status == "applied":
                if not isinstance(version, int) or isinstance(version, bool) or version <= 0:
                    return self.send_json(dict(error="result_version: требуется целое число версии"), 400)
                if not isinstance(checksum, str) or not checksum.strip():
                    return self.send_json(dict(error="checksum: требуется контрольная сумма"), 400)
                if not isinstance(document_uuid, str) or not document_uuid.strip():
                    return self.send_json(dict(error="document_uuid: требуется идентификатор документа 1С"), 400)
                if task.stale_reason:
                    return self.send_json(dict(
                        error="Результат изменён после подтверждения; квитанция не принята.",
                        reasons=[task.stale_reason]), 409)
                if not task.result.get("checksum"):
                    return self.send_json(dict(error="Результат ещё не подтверждён."), 409)
                if version != task.result_version:
                    return self.send_json(dict(
                        error=f"Квитанция относится к версии {version}, подтверждена версия {task.result_version}."), 409)
                if checksum.strip().casefold() != task.result["checksum"].casefold():
                    return self.send_json(dict(error="Контрольная сумма не совпадает с подтверждённым снимком."), 409)
            elif isinstance(version, bool) or (version is not None and not isinstance(version, int)):
                return self.send_json(dict(error="result_version: требуется целое число версии"), 400)
            document_number = payload.get("document_number")
            message = payload.get("message")
            warnings = payload.get("stock_warnings")
            for value, name, limit in ((document_number, "document_number", 200),
                                       (message, "message", 1000)):
                if value is not None and (not isinstance(value, str) or len(value) > limit):
                    return self.send_json(dict(error=f"{name}: некорректное значение"), 400)
            if warnings is not None and (not isinstance(warnings, list) or len(warnings) > 200):
                return self.send_json(dict(error="stock_warnings: ожидается список"), 400)
            receipt = {
                "operation_id": operation.strip(),
                "status": status,
                "database_id": database.strip(),
                "request_id": request.strip(),
                "document_uuid": document_uuid.strip() if isinstance(document_uuid, str) and document_uuid.strip() else None,
                "result_version": version,
                "checksum": checksum.strip() if isinstance(checksum, str) else None,
                "document_number": document_number.strip() if isinstance(document_number, str) and document_number.strip() else None,
                "message": message.strip() if isinstance(message, str) and message.strip() else None,
                "stock_warnings": warnings or [],
                "received_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
                "fingerprint": contract.fingerprint(payload),
            }
            try:
                task, created = exchange.add_receipt(task_id, receipt)
            except ExchangeConflict as conflict:
                return self.send_json(dict(error=str(conflict)), 409)
            except ExchangeError as invalid:
                return self.send_json(dict(error=str(invalid)), 400)
            document = task_document(task, self.base_url())
            document["created"] = created
            document["operation_id"] = operation.strip()
            document["receipt_status"] = status
            return self.send_json(document)

        def do_GET(self):
            parts = urlparse(self.path)
            path = unquote(parts.path)
            if path in {"/", "/index.html", "/app.js", "/styles.css", "/editing.css", "/ocr_activity.js", "/ocr_activity.css",
                        "/invoice.js", "/invoice.css", "/matching_compare.js", "/matching_compare.css", "/voting.js"}:
                filename = "index.html" if path == "/" else path.lstrip("/")
                mime = {"html":"text/html", "js":"text/javascript", "css":"text/css"}[filename.split(".")[-1]]
                return self.send_data((STATIC / filename).read_bytes(), mime+"; charset=utf-8")
            if path == '/api/1c/stock-sync/status':
                return self.stock_status(parts.query)
            if path == '/api/1c/stock-sync/summary':
                return self.stock_summary()
            if path == "/api/health":
                return self.send_json(dict(ok=True, pass_similarity=PASS_SIMILARITY, max_upload_bytes=store.max_upload,
                                           exchange=True))
            if path.startswith("/tasks"):
                page = re.fullmatch(r"/tasks/([a-f0-9]{32})/ui", path)
                if page:
                    return self.send_data((STATIC / "index.html").read_bytes(),
                                          "text/html; charset=utf-8")
                page_state = re.fullmatch(r"/tasks/([a-f0-9]{32})/ui/state", path)
                if page_state:
                    try:
                        task = exchange.get(page_state[1])
                    except ExchangeNotFound as error:
                        return self.send_json(dict(error=str(error)), 404)
                    return self.send_json(task_document(task, self.base_url()))
                result_route = re.fullmatch(r"/tasks/([a-f0-9]{32})/result(\.xml)?", path)
                if result_route:
                    if result_route[2]:
                        return self.exchange_result_xml(result_route[1], parts.query)
                    return self.exchange_result(result_route[1], parts.query)
                route = re.fullmatch(r"/tasks/([a-f0-9]{32})", path)
                if route:
                    try:
                        task = exchange.get(route[1])
                    except ExchangeNotFound as error:
                        return self.send_json(dict(error=str(error)), 404)
                    return self.send_json(task_document(task, self.base_url()))
                return self.send_json(dict(error="Адрес не найден"), 404)
            if path == "/api/catalog":
                query = parse_qs(parts.query).get("q", [""])[0]
                return self.send_json(dict(items=store.pipeline.search_catalog(query)))
            if path == "/api/catalog-1c":
                query = parse_qs(parts.query).get("q", [""])[0]
                params=parse_qs(parts.query)
                database=(params.get('database_id') or [None])[0]
                task_id=(params.get('task_id') or [None])[0]
                if task_id:
                    try:
                        database=exchange.get(task_id).database_id
                    except ExchangeNotFound as error:
                        return self.send_json(dict(error=str(error)),404)
                return self.send_json(invoice_service.search_catalog_1c(query,database_id=database,required_quantity=(params.get('required_quantity') or [None])[0]))
            if path == "/api/jobs":
                with store.lock:
                    jobs = sorted(store.jobs.values(), key=lambda job:job.created_at, reverse=True)[:12]
                return self.send_json([dict(id=job.id, filename=job.filename, status=job.status) for job in jobs])
            route = re.fullmatch(r"/api/jobs/([a-f0-9]{32})(?:/(events|files/[^/]+|invoice|invoice\.pdf))?", path)
            if not route:
                return self.send_json(dict(error="Адрес не найден"), 404)
            job = store.get(route[1])
            if job is None:
                return self.send_json(dict(error="Обработка не найдена"), 404)
            suffix = route[2]
            if suffix is None:
                return self.send_json(job.public())
            if suffix == "events":
                return self.events(job, parts.query)
            if suffix == "invoice":
                with job.condition:
                    if job.status != "done" or not job.result:
                        return self.send_json(dict(error="Дождитесь завершения распознавания."), 409)
                try:
                    model = invoice_service.invoice_model(job.result,
                                                          selections=invoice_service.load_selections(job),
                                                          database_id=self.invoice_context(job))
                except ValueError as error:
                    return self.send_json(dict(error=str(error)), 409)
                except Exception:
                    logging.exception("Invoice model request failed")
                    return self.send_json(dict(error="Счёт не построен: внутренняя ошибка расчёта."), 500)
                return self.send_json(model)
            if suffix == "invoice.pdf":
                query = parse_qs(parts.query)
                with job.condition:
                    if job.status != "done" or not job.result:
                        return self.send_json(dict(error="Дождитесь завершения распознавания."), 409)
                try:
                    model = invoice_service.invoice_model(job.result,
                                                          selections=invoice_service.load_selections(job),
                                                          database_id=self.invoice_context(job))
                except ValueError as error:
                    return self.send_json(dict(error=str(error)), 409)
                try:
                    payload = invoice_service.render_invoice(model, query.get("revision", [None])[0],
                                                             query.get("token", [None])[0])
                except RuntimeError as error:
                    return self.send_json(dict(error=str(error), reasons=model.get("blocked", [])), 409)
                except ValueError as error:
                    return self.send_json(dict(error=str(error)), 400)
                except Exception:
                    return self.send_json(dict(error="PDF не построен: внутренняя ошибка расчёта."), 500)
                number = re.sub(r"[^A-Za-z0-9._-]+", "_", model["document"].get("number") or "invoice")
                return self.send_data(payload, "application/pdf", attachment=f"invoice-{number}.pdf")
            filename = suffix.removeprefix("files/")
            allowed = DOWNLOADS | {asset["filename"] for asset in job.artifacts}
            if filename not in allowed or not (job.directory / filename).is_file():
                return self.send_json(dict(error="Файл не найден"), 404)
            data = (job.directory / filename).read_bytes()
            attachment = filename if filename in {"receipt.csv", "receipt.json", "ocr_rectified.txt", "rectification_map.npz"} else None
            mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
            return self.send_data(data, mime, attachment=attachment)

        def events(self, job, query):
            try:
                after = int(self.headers.get("Last-Event-ID") or parse_qs(query).get("after", [0])[0])
            except (TypeError, ValueError):
                after = 0
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                while True:
                    with job.condition:
                        pending = [event for event in job.events if event["id"] > after]
                        terminal = job.status in {"done", "failed"}
                        if not pending and not terminal:
                            job.condition.wait(timeout=12)
                    for event in pending:
                        payload = json.dumps(event, ensure_ascii=False)
                        self.wfile.write(f'id: {event["id"]}\nevent: pipeline\ndata: {payload}\n\n'.encode())
                        after = event["id"]
                    self.wfile.write(b"event: heartbeat\ndata: {}\n\n")
                    self.wfile.flush()
                    if terminal and after >= len(job.events):
                        break
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                pass
            self.close_connection = True

        def do_POST(self):
            parts = urlparse(self.path)
            if parts.path == '/api/1c/stock-sync':
                return self.stock_upload()
            if parts.path == "/tasks":
                return self.exchange_create()
            photo_route = re.fullmatch(r"/tasks/([a-f0-9]{32})/photos", parts.path)
            if photo_route:
                return self.exchange_photo(photo_route[1])
            confirm_route = re.fullmatch(r"/tasks/([a-f0-9]{32})/confirm", parts.path)
            if confirm_route:
                return self.exchange_confirm(confirm_route[1])
            link_route = re.fullmatch(r"/tasks/([a-f0-9]{32})/document-link", parts.path)
            if link_route:
                return self.exchange_document_link(link_route[1])
            receipt_route = re.fullmatch(r"/tasks/([a-f0-9]{32})/import-receipt", parts.path)
            if receipt_route:
                return self.exchange_import_receipt(receipt_route[1])
            edit_route = re.fullmatch(r"/api/jobs/([a-f0-9]{32})/edits", parts.path)
            invoice_route = re.fullmatch(r"/api/jobs/([a-f0-9]{32})/invoice/(preview|preview-pdf|selections|train)", parts.path)
            if parts.path != "/api/jobs" and edit_route is None and invoice_route is None:
                self.close_connection = True
                return self.send_json(dict(error="Адрес не найден"), 404)
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            if not 0 < length <= store.max_upload:
                self.close_connection = True
                return self.send_json(dict(error=f"Выберите изображение размером до {store.max_upload//1024//1024} МБ."), 413)
            if (edit_route or invoice_route) and length > 65536:
                self.close_connection = True
                return self.send_json(dict(error="Слишком большой запрос редактирования"), 413)
            data = self.rfile.read(length)
            if len(data) != length:
                return self.send_json(dict(error="Изображение загружено не полностью"), 400)
            if invoice_route:
                job = store.get(invoice_route[1])
                if job is None:
                    return self.send_json(dict(error="Чек не найден"), 404)
                return self.handle_invoice_request(job, invoice_route[2], data)
            if edit_route:
                job = store.get(edit_route[1])
                if job is None:
                    return self.send_json(dict(error="Чек не найден"), 404)
                try:
                    payload = json.loads(data)
                    if not isinstance(payload, dict):
                        raise ValueError("Некорректные данные редактирования.")
                    if not any(key in payload for key in ("row", "expected_total", "finalize")):
                        raise ValueError("Не указано изменение чека.")
                    with job.condition:
                        if job.status != "done" or not job.result:
                            return self.send_json(dict(error="Дождитесь завершения распознавания."), 409)
                        updated = apply_edit(job.result, payload, store.pipeline.catalog_entry)
                        learned = store.pipeline.remember_selection(job.result, updated, payload, job.id)
                        save_receipt(job.directory, updated)
                        job.result = updated
                        job.persist()
                    exchange.invalidate_job(
                        job.id, "Чек изменён после подтверждения для 1С; подтвердите результат заново.")
                    return self.send_json(dict(result=updated, mapping_saved=bool(learned)))
                except RuntimeError as error:
                    return self.send_json(dict(error=str(error)), 409)
                except (ValueError, TypeError) as error:
                    return self.send_json(dict(error=str(error)), 400)
            filename = parse_qs(parts.query).get("filename", ["receipt.jpg"])[0]
            job = store.submit(data, filename)
            return self.send_json(dict(id=job.id, status=job.status), 202)

        def handle_invoice_request(self, job, action, data):
            try:
                payload = json.loads(data)
                if not isinstance(payload, dict):
                    raise ValueError("Некорректные данные выбора номенклатуры.")
                selections = invoice_service.SelectionPayload(selections=payload.get("selections") or {})
                clean = {row_id: entry.model_dump() for row_id, entry in selections.selections.items()}
            except (ValueError, TypeError, ValidationError) as error:
                return self.send_json(dict(error=str(error)), 400)
            with job.condition:
                if job.status != "done" or not job.result:
                    return self.send_json(dict(error="Дождитесь завершения распознавания."), 409)
                if action == "train":
                    if payload.get("confirm") is not True:
                        return self.send_json(dict(error="Подтвердите обучение связям явно."), 400)
                    try:
                        from App.invoicing.training_pairs import training_lock
                        context = self.invoice_context(job)
                        with training_lock:
                            if context:
                                result = invoice_service.train_from_selections(
                                    job.result, clean, database_id=context)
                            else:
                                result = invoice_service.train_from_selections(job.result, clean)
                    except ValueError as error:
                        return self.send_json(dict(error=str(error)), 409)
                    except Exception:
                        return self.send_json(dict(error="Не удалось сохранить обучение; проверьте журнал и хранилища."), 500)
                    exchange.invalidate_job(
                        job.id, "Обучение связям изменило счёт; подтвердите результат заново.")
                    return self.send_json(dict(training=result))
            if action == "selections":
                with job.condition:
                    previous = invoice_service.load_selections(job)
                try:
                    clean = invoice_service.save_selections(job, clean)
                except OSError as error:
                    return self.send_json(dict(error=str(error)), 500)
                if clean != previous:
                    exchange.invalidate_job(
                        job.id, "Связи номенклатуры изменены после подтверждения; подтвердите результат заново.")
            try:
                model = invoice_service.invoice_model(job.result, selections=clean,
                                                      database_id=self.invoice_context(job))
                if action == "preview-pdf":
                    pdf = invoice_service.render_invoice(model, model["revision"], model["token"])
                    return self.send_data(pdf, "application/pdf")
            except ValueError as error:
                return self.send_json(dict(error=str(error)), 409)
            except Exception:
                return self.send_json(dict(error="Счёт не построен: внутренняя ошибка расчёта."), 500)
            return self.send_json(model)

    from App.web_debug import MatchingDebugHandler

    class CombinedHandler(Handler, MatchingDebugHandler):
        def do_GET(self):
            path = urlparse(self.path).path
            if path in {'/analog', '/analog/list', '/analog/list/data', '/health', '/api/matching/algorithms', '/api/stock/categories'}:
                self.close_connection = True
                return MatchingDebugHandler.do_GET(self)
            return super().do_GET()

        def do_POST(self):
            # OCR_ANALOG_BRIDGE_V1: catalog context comes only from this job's task.
            analog_route = re.fullmatch(r'/api/jobs/([a-f0-9]{32})/analogs', urlparse(self.path).path)
            if analog_route:
                job = store.get(analog_route[1])
                if job is None:
                    return self.send_json(dict(error='Обработка не найдена'), 404)
                try:
                    length = int(self.headers.get('Content-Length', '0'))
                    if length < 2 or length > 4096 or self.headers.get('Transfer-Encoding'):
                        self.close_connection = True
                        return self.send_json(dict(error='Неверный размер запроса'), 400)
                    payload = json.loads(self.rfile.read(length).decode('utf-8'))
                    if (not isinstance(payload, dict) or isinstance(payload.get('row_id'), bool)
                            or not isinstance(payload.get('row_id'), (str, int))):
                        return self.send_json(dict(error='Не указан row_id'), 400)
                    revision = payload.get('receipt_revision')
                    if revision is not None and (type(revision) is not int or revision < 0):
                        return self.send_json(dict(error='Неверная версия чека'), 400)
                    with job.condition:
                        if job.status != 'done' or not job.result:
                            return self.send_json(dict(error='OCR ещё не завершён'), 409)
                        receipt = copy.deepcopy(job.result)
                        if revision is not None and receipt.get('revision', 0) != revision:
                            return self.send_json(dict(error='Чек изменился. Повторите поиск.'), 409)
                    from App.invoicing.matching_bridge import for_invoice_row
                    result = for_invoice_row(receipt, payload['row_id'], self.invoice_context(job))
                    with job.condition:
                        if job.result.get('revision', 0) != receipt.get('revision', 0):
                            return self.send_json(dict(error='Чек изменился. Повторите поиск.'), 409)
                    return self.send_json(invoice_service._plain(result))
                except (ValueError, TypeError, UnicodeError) as error:
                    return self.send_json(dict(error=str(error)), 400)
                except Exception:
                    logging.exception('Analog voting request failed')
                    return self.send_json(dict(error='Поиск аналогов недоступен'), 503)
            if urlparse(self.path).path in {'/analog', '/api/matching/debug', '/api/matching/test'}:
                self.close_connection = True
                return MatchingDebugHandler.do_POST(self)
            return super().do_POST()

    server = ReceiptHTTPServer((host, port), CombinedHandler)
    server.store = store
    return server


def main():
    parser = argparse.ArgumentParser(description="Локальный веб-интерфейс чеков Торгсофт")
    try:
        settings = server_settings()
    except ValueError as error:
        parser.error(str(error))
    parser.add_argument("--host", default=settings.host)
    parser.add_argument("--port", type=int, default=settings.port)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "web_outputs")
    args = parser.parse_args()
    if Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
        parser.error("Используйте проектный .venv: .\\.venv\\Scripts\\python.exe -m App.web_server")
    server = create_server(args.host, args.port, args.output_dir)
    server.store.pipeline.knowledge.initialize()
    print(f"Receipt workspace: {ServerSettings(args.host, server.server_port).browser_url}", flush=True)
    print(f"Listening on {args.host}:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        server.store.pool.shutdown(wait=True)
        from App.matching.cpu_executor import shutdown_executors
        shutdown_executors()


if __name__ == "__main__":
    main()
