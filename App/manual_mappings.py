"""Persistent, explicit OCR-to-catalog decisions. No fuzzy reuse of learned text."""
from __future__ import annotations

import json
import copy
import os
import tempfile
import threading
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data/knowledge/manual_mappings.json"
_locks_guard = threading.Lock()
_locks = {}


def ocr_key(text):
    # Keep punctuation, units and numbers: similar text is not the same decision.
    return "".join(unicodedata.normalize("NFKC", str(text)).casefold().split())


class ManualMappings:
    def __init__(self, path=DEFAULT_PATH):
        self.path = Path(path).resolve()
        with _locks_guard:
            self.lock = _locks.setdefault(str(self.path).casefold(), threading.RLock())
        self._signature, self._document = None, None

    def _read(self):
        if not self.path.exists():
            return dict(schema_version=1, mappings={}, history=[])
        stat = self.path.stat()
        signature = (stat.st_mtime_ns, stat.st_size, stat.st_ino)
        if signature == self._signature:
            return self._document
        document = json.loads(self.path.read_text(encoding="utf-8-sig"))
        if not isinstance(document, dict) or not isinstance(document.get("mappings"), dict) or not isinstance(document.get("history"), list):
            raise ValueError("Некорректный формат файла ручных сопоставлений.")
        canonical = {}
        for record in document["mappings"].values():
            if not isinstance(record, dict) or not all(field in record for field in ("ocr_raw", "catalog_key", "catalog_name", "article")):
                raise ValueError("Некорректная запись ручного сопоставления.")
            key = ocr_key(record["ocr_raw"])
            if not key:
                raise ValueError("Пустой OCR в файле ручных сопоставлений.")
            if key not in canonical or record.get("updated_at", "") >= canonical[key].get("updated_at", ""):
                canonical[key] = dict(record, ocr_key=key)
        document["mappings"] = canonical
        self._signature, self._document = signature, document
        return document

    def _write(self, document):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent,
                                             prefix=self.path.name+".", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(document, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            self._signature = None
        finally:
            if temporary and temporary.exists():
                temporary.unlink()

    def initialize(self):
        with self.lock:
            if not self.path.exists():
                self._write(dict(schema_version=1, mappings={}, history=[]))
            else:
                self._read()

    def lookup(self, raw_text):
        with self.lock:
            return copy.deepcopy(self._read()["mappings"].get(ocr_key(raw_text)))

    def remember(self, raw_text, entry, article="", job_id="", row_id=None):
        key = ocr_key(raw_text)
        if not key:
            raise ValueError("Нельзя запомнить пустое название OCR.")
        record = dict(ocr_key=key, ocr_raw=raw_text, catalog_key=entry["normalized"],
                      catalog_name=entry["name"], article=article,
                      updated_at=datetime.now(timezone.utc).isoformat(), job_id=job_id, row_id=row_id)
        with self.lock:
            document = copy.deepcopy(self._read())
            before = document["mappings"].get(key)
            document["history"].append(dict(ocr_key=key, recorded_at=record["updated_at"], before=before, after=record))
            document["mappings"][key] = record
            self._write(document)
        return record

    def resolve(self, raw_text, catalog_index, automatic):
        if automatic.get("best") and automatic["best"]["score"] >= 100:
            return automatic
        saved = self.lookup(raw_text)
        if not saved:
            return automatic
        entry = catalog_index.get(saved["catalog_key"])
        if not entry:
            return automatic
        articles = {source.get("article", "") for source in entry["sources"]}
        article = saved["article"]
        if (article and article not in articles) or (not article and len(articles - {""}) > 1):
            return automatic
        # 100 here is an exact knowledge-key hit, not the text similarity score.
        best = dict(name=entry["name"], normalized=entry["normalized"], sources=entry["sources"],
                    score=100.0, raw_similarity=automatic.get("best", {}).get("raw_similarity") if automatic.get("best") else None,
                    missing_query_numbers=[], source="manual_mapping", article=article)
        others = [c for c in automatic.get("top_matches", []) if c["normalized"] != entry["normalized"]]
        return dict(status="manual_mapping", best=best, top_matches=[best]+others[:4],
                    gap=None, source="manual_mapping", manual_mapping=saved,
                    automatic_match=automatic)
