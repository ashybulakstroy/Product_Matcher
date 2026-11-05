"""Validated full 1C snapshots. No OCR, mapping writes, or API-key handling."""
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import threading
import time
from uuid import UUID

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]
PREFIX = 'catalog_1c_current'


def _replace_retry(source, destination, attempts=400, delay=0.01):
    """Windows: os.replace падает с PermissionError, пока файл открыт на чтение.

    Читатели (каталог, главная страница) могут держать файл открытым долю
    секунды, поэтому подмену повторяем вместо отказа от снимка.
    """
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == attempts-1:
                raise
            time.sleep(delay)


class StockSyncError(ValueError):
    def __init__(self, code, message, status=422):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status

    def public(self):
        return dict(code=self.code, message=self.message)


def settings(env_path=None, environ=None):
    values = {**dotenv_values(env_path or ROOT/'.env', encoding='utf-8-sig'),
              **(os.environ if environ is None else environ)}
    try:
        ttl = float(values.get('STOCK_TTL_HOURS', '1')) * 3600
        limit = int(values.get('STOCK_SYNC_MAX_BYTES', '10485760'))
        if not math.isfinite(ttl) or ttl <= 0 or limit <= 0:
            raise ValueError()
    except (ValueError, TypeError, OverflowError):
        raise StockSyncError('invalid_config', 'STOCK_TTL_HOURS и STOCK_SYNC_MAX_BYTES должны быть положительными.', 500)
    return ttl, limit


def uuid_text(value, field, status=422):
    try:
        if not isinstance(value, str):
            raise ValueError()
        return str(UUID(value))
    except (ValueError, TypeError, AttributeError):
        raise StockSyncError('invalid_uuid', f'{field}: требуется UUID.', status)


def as_of(value):
    try:
        if not isinstance(value, str):
            raise ValueError()
        moment = datetime.fromisoformat(value)
        if moment.tzinfo is None or moment.utcoffset() is None:
            raise ValueError()
        return moment.astimezone(timezone.utc)
    except (ValueError, TypeError, OverflowError):
        raise StockSyncError('invalid_date', 'stock_as_of: требуется дата ISO 8601 с часовым поясом.')


def _text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise StockSyncError('invalid_field', f'{field}: требуется непустая строка.')


def _decimal(value, field):
    if value is None:
        return
    try:
        if not isinstance(value, str) or not value.strip() or not Decimal(value).is_finite():
            raise ValueError()
    except (ValueError, InvalidOperation):
        raise StockSyncError('invalid_decimal', f'{field}: требуется строка Decimal либо null.')


def _object(pairs):
    result = {}
    duplicates = []
    for key, value in pairs:
        if key in result:
            duplicates.append(key)
            continue
        result[key] = value
    if duplicates:
        result = _Record(result)
        result.duplicates = tuple(duplicates)
    return result


class _Record(dict):
    """JSON-объект, в котором были повторяющиеся имена полей."""
    duplicates = ()


def _parse(raw):
    try:
        return json.loads(raw.decode('utf-8'), object_pairs_hook=_object,
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError()))
    except (ValueError, UnicodeError):
        raise StockSyncError('invalid_json', 'Требуется корректный JSON UTF-8 без BOM.')


def _header(data):
    """Общая часть снимка: без неё принимать нечего. Ошибка — весь файл."""
    if not isinstance(data, dict) or data.get('schema_version') != '1.0' or data.get('source') != '1c':
        raise StockSyncError('invalid_schema', 'Ожидается schema_version=1.0, source=1c.')
    if getattr(data, 'duplicates', ()):
        raise StockSyncError('duplicate_field', 'JSON содержит повторяющиеся имена полей.')
    uuid_text(data.get('snapshot_id'), 'snapshot_id')
    _text(data.get('database_id'), 'database_id')
    if len(data['database_id']) > 128:
        raise StockSyncError('invalid_field', 'database_id: максимум 128 символов.')
    for field in ('organization', 'warehouse'):
        obj = data.get(field)
        if not isinstance(obj, dict):
            raise StockSyncError('invalid_field', f'{field}: требуется объект.')
        if getattr(obj, 'duplicates', ()):
            raise StockSyncError('duplicate_field', f'{field}: повторяющиеся имена полей.')
        uuid_text(obj.get('id'), field+'.id')
        _text(obj.get('name'), field+'.name')
    as_of(data.get('stock_as_of'))
    items = data.get('items')
    count = data.get('record_count')
    if not isinstance(items, list) or not items or type(count) is not int or count != len(items):
        raise StockSyncError('invalid_count', 'record_count должен совпадать с непустым массивом items.')
    return items


def item_identity(item):
    """Проверка одной товарной записи; при ошибке — StockSyncError."""
    if not isinstance(item, dict):
        raise StockSyncError('invalid_item', 'items: ожидаются объекты товаров.')
    if getattr(item, 'duplicates', ()):
        raise StockSyncError('duplicate_field', 'Повторяющиеся имена полей записи.')
    identity = uuid_text(item.get('uuid'), 'items.uuid')
    for field in ('code', 'name', 'unit'):
        _text(item.get(field), 'items.'+field)
    if type(item.get('service')) is not bool:
        raise StockSyncError('invalid_item', 'service: требуется boolean.')
    for field in ('stock_quantity', 'stock_amount', 'accounting_price'):
        if field not in item:
            raise StockSyncError('invalid_item', f'Отсутствует {field}.')
        _decimal(item[field], field)
        if item['service'] and item[field] is not None:
            raise StockSyncError('invalid_service', 'Складские числовые поля услуг должны быть null.')
    if 'account_code' not in item or (item['account_code'] is not None and not isinstance(item['account_code'], str)):
        raise StockSyncError('invalid_item', 'account_code: требуется строка либо null.')
    return identity, item['code'].strip().casefold()


def select_items(items):
    """Корректные записи и отброшенные: одна плохая строка не отменяет снимок."""
    accepted, skipped, uuids, codes = [], [], set(), set()
    for index, item in enumerate(items):
        try:
            identity, code = item_identity(item)
            if identity in uuids or code in codes:
                raise StockSyncError('duplicate_item', 'UUID или код уже встречались в снимке.')
        except StockSyncError as error:
            skipped.append(dict(index=index,
                                uuid=item.get('uuid') if isinstance(item, dict) else None,
                                code=item.get('code') if isinstance(item, dict) else None,
                                code_error=error.code, reason=error.message))
            continue
        uuids.add(identity); codes.add(code)
        accepted.append(item)
    return accepted, skipped


def validate(raw):
    """Строгая проверка опубликованного файла целиком."""
    data = _parse(raw)
    items = _header(data)
    uuids, codes = set(), set()
    for item in items:
        identity, code = item_identity(item)
        if identity in uuids or code in codes:
            raise StockSyncError('duplicate_item', 'UUID и коды товаров должны быть уникальны.')
        uuids.add(identity); codes.add(code)
    return data



def path_for(directory, database_id):
    key = database_id.strip().casefold()
    slug = re.sub(r'[^\w.-]', '_', key)[:64]
    # Hash prevents collisions after sanitization/truncation and Windows case folding.
    suffix = hashlib.sha256(key.encode('utf-8')).hexdigest()[:12]
    return Path(directory) / f'{PREFIX}_{slug}_{suffix}.json'


def paths(directory):
    result=sorted(Path(directory).glob(PREFIX+'*.json'))
    legacy=Path(directory)/(PREFIX+'.json')
    if legacy in result:
        # A named snapshot supersedes the legacy singleton for that same base.
        data=read(legacy)
        if path_for(directory,data['database_id']) in result:
            result.remove(legacy)
    return result


def read(path):
    try:
        return validate(Path(path).read_bytes())
    except (OSError, StockSyncError):
        raise StockSyncError('corrupt_snapshot', 'Производственный JSON остатков повреждён или недоступен.', 500)


def select_path(directory, database_id=None):
    if database_id:
        named = path_for(directory, database_id)
        if named.exists():
            return named
        legacy = Path(directory)/(PREFIX+'.json')
        if legacy.exists() and read(legacy)['database_id'].strip().casefold() == database_id.strip().casefold():
            return legacy
        return None
    candidates = paths(directory)
    if len(candidates) > 1:
        raise StockSyncError('database_required', 'Снимков несколько: укажите database_id для выбора каталога.', 409)
    return candidates[0] if candidates else None


class StockStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.lock = threading.Lock()
        # Only known temporary stock files, never arbitrary files/directories.
        for tmp in self.directory.glob(PREFIX+'*.json.tmp'):
            tmp.unlink()

    def status(self, organization_id, warehouse_id, ttl_seconds, database_id=None, now=None):
        organization_id = uuid_text(organization_id, 'organization_id', 400)
        warehouse_id = uuid_text(warehouse_id, 'warehouse_id', 400)
        result = dict(status='missing', needs_update=True, organization_id=organization_id,
                      warehouse_id=warehouse_id, snapshot_id=None, stock_as_of=None,
                      ttl_remaining_seconds=0, record_count=0)
        matches = []
        candidates = paths(self.directory) if not database_id else [p for p in [select_path(self.directory, database_id)] if p]
        for path in candidates:
            data = read(path)
            if database_id and data['database_id'].strip().casefold()!=database_id.strip().casefold():
                continue
            if (uuid_text(data['organization']['id'], 'organization.id') == organization_id
                    and uuid_text(data['warehouse']['id'], 'warehouse.id') == warehouse_id):
                matches.append(data)
        if len(matches) > 1:
            raise StockSyncError('database_required', 'Пара встречается в нескольких базах: укажите database_id.', 409)
        if not matches:
            return result
        data = matches[0]
        moment = now or datetime.now(timezone.utc)
        age = (moment.astimezone(timezone.utc)-as_of(data['stock_as_of'])).total_seconds()
        remaining = max(0, int(ttl_seconds-age))
        result.update(status='fresh' if age < ttl_seconds else 'expired', needs_update=age >= ttl_seconds,
                      snapshot_id=data['snapshot_id'], stock_as_of=data['stock_as_of'],
                      ttl_remaining_seconds=remaining, record_count=data['record_count'])
        return result

    def publish(self, raw, expected_hash, after_commit=None):
        if not isinstance(expected_hash, str) or not re.fullmatch('[a-fA-F0-9]{64}', expected_hash):
            raise StockSyncError('invalid_sha256', 'X-Content-SHA256: требуется SHA-256 тела.')
        digest = hashlib.sha256(raw).hexdigest()
        if not hmac.compare_digest(digest, expected_hash.lower()):
            raise StockSyncError('sha256_mismatch', 'SHA-256 тела не совпадает.')
        # Routing is known only after validating the body. Nothing is published yet.
        data = _parse(raw)
        items = _header(data)
        accepted, skipped = select_items(items)
        if not accepted:
            reasons = sorted({row['reason'] for row in skipped})
            details = '; '.join(reasons[:3]) or 'причина не указана'
            raise StockSyncError('no_valid_items',
                                 'В снимке нет ни одной корректной записи товара. '
                                 f'Отклонено {len(skipped)}: {details}'[:500], 422)
        # A bad row must not discard the whole snapshot: publish only accepted rows.
        clean = dict(data, items=accepted, record_count=len(accepted))
        payload = json.dumps(clean, ensure_ascii=False, indent=2).encode('utf-8')
        payload_hash = hashlib.sha256(payload).hexdigest()
        destination = path_for(self.directory, data['database_id'])
        tmp = destination.with_suffix('.json.tmp')
        with self.lock:
            try:
                existing_path = select_path(self.directory, data['database_id'])
                if existing_path:
                    old = read(existing_path)
                    if old['database_id'].strip().casefold()!=data['database_id'].strip().casefold():
                        raise StockSyncError('origin_mismatch','Существующий JSON относится к другой базе.',500)
                    if uuid_text(old['snapshot_id'], 'snapshot_id') == uuid_text(data['snapshot_id'], 'snapshot_id'):
                        published = hashlib.sha256(existing_path.read_bytes()).hexdigest()
                        if not hmac.compare_digest(published, payload_hash):
                            raise StockSyncError('snapshot_conflict', 'snapshot_id уже опубликован с другим содержимым.', 409)
                        if after_commit:
                            after_commit(clean)
                        return self._committed(clean, digest, skipped)
                    if as_of(data['stock_as_of']) < as_of(old['stock_as_of']):
                        raise StockSyncError('older_snapshot', 'Более старый stock_as_of не заменяет текущий снимок базы.', 409)
                self.directory.mkdir(parents=True, exist_ok=True)
                with tmp.open('wb') as target:
                    target.write(payload); target.flush(); os.fsync(target.fileno())
                if not hmac.compare_digest(hashlib.sha256(tmp.read_bytes()).hexdigest(), payload_hash):
                    raise StockSyncError('disk_integrity', 'Проверка целостности временного файла не пройдена.', 500)
                # The published file must pass the strict reader check before the switch.
                validate(tmp.read_bytes())
                _replace_retry(tmp, destination)
                if after_commit:
                    after_commit(clean)
                return self._committed(clean, digest, skipped)
            except OSError:
                raise StockSyncError('write_failed', 'Не удалось записать снимок остатков.', 500)
            finally:
                tmp.unlink(missing_ok=True)

    @staticmethod
    def _committed(data, digest, skipped=()):
        return dict(status='committed', snapshot_id=data['snapshot_id'],
                    organization_id=data['organization']['id'], warehouse_id=data['warehouse']['id'],
                    received_count=data['record_count'], skipped_count=len(skipped),
                    skipped=list(skipped)[:50], integrity='verified', sha256=digest,
                    stock_as_of=data['stock_as_of'])
