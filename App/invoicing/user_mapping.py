"""Confirmed OCR→Torgsoft reference names for the invoice stage.

Keeps the donor file store `Data/reference/User_Mapping/OCR_Torgsoft_mapping.txt`
(read-only for lookups, TSV). This is NOT the OCR pipeline's own knowledge
store (`Data/knowledge/manual_mappings.json`), and the two must stay separate.
Only exact normalised names match; nothing here is guessed from similarity.
"""
from __future__ import annotations

import csv
import os
import re
import threading
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAPPING_DIR = ROOT / "Data" / "reference" / "User_Mapping"
MAPPING_FILE = MAPPING_DIR / "OCR_Torgsoft_mapping.txt"
_write_lock = threading.Lock()


def normalize_mapping_key(value: str) -> str:
    """Project-wide key policy: ё→е, lowercase, word tokens only."""
    value = value.casefold().replace("ё", "е")
    value = re.sub(r"(?<=[а-я])[-]\s+(?=[а-я])", "", value)
    return " ".join(re.findall(r"[a-zа-я0-9]+", value))


def _signature(path: Path) -> tuple[str, int, int]:
    if not path.exists():
        return str(path.resolve()), 0, 0
    stat = path.stat()
    return str(path.resolve()), stat.st_mtime_ns, stat.st_size


@lru_cache(maxsize=4)
def _load_mapping(signature: tuple[str, int, int]) -> dict[str, tuple[str, str]]:
    path = Path(signature[0])
    mappings = {}
    if not path.exists():
        return mappings
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        for row in csv.reader(source, delimiter="\t"):
            if len(row) < 2:
                continue
            ocr_name, reference_name = row[0].strip(), row[1].strip()
            key = normalize_mapping_key(ocr_name)
            if key and reference_name:
                mappings[key] = (ocr_name, reference_name)
    return mappings


def lookup_user_mapping(ocr_name: str, mapping_file: Path = MAPPING_FILE) -> str | None:
    entry = _load_mapping(_signature(mapping_file)).get(normalize_mapping_key(ocr_name))
    return entry[1] if entry else None


def save_user_mappings(
    pairs: list[tuple[str, str]],
    mapping_file: Path = MAPPING_FILE,
) -> int:
    with _write_lock:
        mappings = dict(_load_mapping(_signature(mapping_file)))
        saved = 0
        for ocr_name, reference_name in pairs:
            ocr_name = ocr_name.strip()
            reference_name = reference_name.strip()
            key = normalize_mapping_key(ocr_name)
            if not key or not reference_name:
                continue
            if mappings.get(key) != (ocr_name, reference_name):
                mappings[key] = (ocr_name, reference_name)
                saved += 1

        mapping_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = mapping_file.with_suffix(mapping_file.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8-sig", newline="") as target:
            writer = csv.writer(target, delimiter="\t", lineterminator="\n")
            writer.writerows(mappings[key] for key in sorted(mappings))
        os.replace(temporary, mapping_file)
        _load_mapping.cache_clear()
        return saved
