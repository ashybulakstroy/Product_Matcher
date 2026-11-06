"""Central numeric OCR settings read from the EXISTING project-root .env only.

Use the same python-dotenv dependency already used by App.ocr.ocr_catalog_eval.
No separate JSON files are created/read. Missing or blank VAT_RATE_PERCENT -> 16.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from pathlib import Path

from dotenv import dotenv_values

DEFAULT_RATE = '16'
DEFAULT_MAX_IMAGES = 250
ENV_FILE = '.env'


def load_numeric_settings(root=None):
    project_root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    env_path = project_root / ENV_FILE
    # Read only the one established project .env. Do not modify os.environ or
    # borrow values from another file/system environment in these settings.
    values = dotenv_values(env_path, encoding='utf-8-sig') if env_path.is_file() else {}
    raw = values.get('VAT_RATE_PERCENT')
    source = 'default_16_percent'
    if raw is None or not str(raw).strip():
        rate = DEFAULT_RATE
    else:
        try:
            value = Decimal(str(raw).strip())
        except InvalidOperation as error:
            raise ValueError(f'Некорректная ставка VAT_RATE_PERCENT в {env_path}: {raw}') from error
        if not value.is_finite() or not (Decimal('0') <= value <= Decimal('100')):
            raise ValueError(f'VAT_RATE_PERCENT в {env_path} должен быть в диапазоне 0..100')
        rate = format(value, 'f')
        source = 'env_file'
    raw_max = values.get('OCR_NUMERIC_MAX_IMAGES')
    if raw_max is None or not str(raw_max).strip():
        max_images = DEFAULT_MAX_IMAGES
    else:
        try:
            max_images = int(str(raw_max).strip())
        except ValueError as error:
            raise ValueError(f'OCR_NUMERIC_MAX_IMAGES в {env_path} должен быть целым числом') from error
        if not 2 <= max_images <= 1000:
            raise ValueError(f'OCR_NUMERIC_MAX_IMAGES в {env_path} должен быть в диапазоне 2..1000')
    return dict(vat_rate_percent=rate, vat_rate_source=source, max_images=max_images)


def get_vat_rate_percent(root=None):
    return load_numeric_settings(root)['vat_rate_percent']
