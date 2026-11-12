# -*- coding: utf-8 -*-
"""Bounded OCR ensemble for suspicious numeric cells only.

RapidOCR reuses the loaded engine. EasyOCR is optional and initialized on its
first *necessary* numeric fallback; missing weights are downloaded once into
models/easyocr, while an existing valid model is reused. No amounts are
fabricated from receipt arithmetic.
"""
from __future__ import annotations

import copy
from collections import Counter, defaultdict
from contextlib import contextmanager, nullcontext
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import itertools
import os
import json
from pathlib import Path
import re
import shutil
import subprocess
import threading
from typing import Any

import cv2
import numpy as np

VERSION = "numeric-ensemble-1.1"
PAYABLE = {"product", "service", "tax"}
CENTS = Decimal(".01")


def parse_numeric(s: Any) -> Decimal | None:
    if isinstance(s, Decimal):
        return s if s.is_finite() and s >= 0 else None
    if s is None or isinstance(s, (dict, list, bool)):
        return None
    text = re.sub(r"\s+", "", str(s).strip()).strip(",.")
    text = re.sub(r"(?:₸|ТГ|KZT|[ТT])$", "", text, flags=re.I)
    if not text or not re.fullmatch(r"\d+(?:[.,]\d+)*", text):
        return None
    if "," in text and "." in text:
        decimal_sep = "," if text.rfind(",") > text.rfind(".") else "."
        text = text.replace("." if decimal_sep == "," else ",", "")
    text = text.replace(",", ".")
    if text.count(".") > 1:
        return None
    try:
        d = Decimal(text)
        return d if d.is_finite() and Decimal(0) <= d < Decimal("1e13") else None
    except InvalidOperation:
        return None


def canon(value: Decimal) -> str:
    return format(value.normalize(), "f") if value != 0 else "0"


def digit_edits(a: Any, b: Any) -> int:
    a = re.sub(r"\D", "", str(a)); b = re.sub(r"\D", "", str(b))
    previous = list(range(len(b) + 1))
    for i, c in enumerate(a, 1):
        current = [i]
        for j, d in enumerate(b, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+int(c != d)))
        previous = current
    return previous[-1]


def _value(row, field):
    return parse_numeric(row.get(field, row.get(field + "_raw", "")))


def rank_suspect_cells(receipt: dict, calculation: dict, limit: int = 4, *, image=None, layout=None) -> list[dict]:
    """Arithmetic provides *where to look*, never the replacement OCR value."""
    expected = parse_numeric(calculation.get("expected_total"))
    computed = parse_numeric(calculation.get("computed_total"))
    gap = expected - computed if expected is not None and computed is not None and calculation.get("status") == "mismatch" else None
    tasks = []
    for row in receipt.get("rows", []):
        if row.get("kind") not in PAYABLE or "y_range" not in row:
            continue
        for field in ("price", "quantity"):
            if str(row.get(field + "_source", "")).lower() in {"user", "manual"}:
                continue
            old = _value(row, field)
            other = _value(row, "quantity" if field == "price" else "price")
            reason, priority, implied = "weak_image", 100, None
            if old is None:
                reason, priority = "missing_or_invalid", -100
            elif gap is None and calculation.get("vat", {}).get("status") == "mismatch" and row["kind"] == "tax":
                reason, priority = "vat_mismatch", -50
            elif gap is not None and other not in (None, 0):
                implied = old + gap / other
                if implied >= 0 and implied != old:
                    # One-cell exact arithmetic is a *priority clue*, not a candidate.
                    edits = digit_edits(canon(old), canon(implied))
                    if edits <= 1 and implied == implied.quantize(CENTS):
                        priority = 0 + (0 if field == "price" else 5) + edits
                        reason = "one_digit_arithmetic_clue"
                    elif edits <= 2:
                        priority = 25 + (0 if field == "price" else 5) + edits
                        reason = "two_digit_arithmetic_clue"
            # Confidence is only a tie-breaker. It is not a calibrated probability.
            lines = row.get("raw_lines", [])
            scores = []
            if isinstance(lines, list):
                for line in lines:
                    box = np.asarray(line.get("box", []))
                    if box.shape == (4, 2):
                        cx = float(box[:, 0].mean())
                        if np.isfinite(cx) and (layout is None or (
                            (layout["quantity_right"] <= cx <= layout["right"]) if field == "price"
                            else (layout["name_right"] <= cx <= layout["quantity_right"]))):
                            scores.append(float(line.get("confidence", 1)))
            low_confidence = 1 - min(scores) if scores else 0.2
            # Cheap OpenCV blur cue only; no OCR invoked to rank cells.
            sharpness = 0.0
            if image is not None and layout is not None:
                region = crop_cell(image, layout, row, field)
                if region is not None and min(region.shape[:2]) >= 3:
                    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if region.ndim==3 else region
                    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            tasks.append({"row": row["row"], "field": field, "kind": row["kind"],
                          "before": canon(old) if old is not None else None,
                          "reason": reason, "priority": priority,
                          "arithmetic_hint": canon(implied) if implied is not None and implied >= 0 else None,
                          "_order": (priority, -low_confidence, sharpness, 0 if field == "price" else 1, int(row["row"]))})
    tasks.sort(key=lambda task: task["_order"])
    return [{k:v for k,v in task.items() if k != "_order"} for task in tasks[:max(0, limit)]]


def crop_cell(image: np.ndarray, layout: dict, row: dict, field: str) -> np.ndarray | None:
    if field not in {"price", "quantity"}:
        return None
    y0, y1 = (int(round(float(v))) for v in row["y_range"])
    left, right = ((layout["quantity_right"], layout["right"]) if field == "price" else
                   (layout["name_right"], layout["quantity_right"]))
    h, w = image.shape[:2]
    width = float(right)-float(left)
    # Currency sign is often the rightmost printed component. Two crops are
    # produced later (full and narrow); never erase the original evidence.
    x0 = max(0, min(w, int(round(left + max(1, width*.06)))))
    x1 = max(0, min(w, int(round(right - max(1, width*.04)))))
    top = max(0, min(h, y0+2)); bottom = max(0, min(h, y1-2))
    if x1-x0 < 9 or bottom-top < 7:
        return None
    return image[top:bottom, x0:x1].copy()


def _tight_ink(gray: np.ndarray) -> np.ndarray:
    """Locate text, avoid cell borders, and preserve a clean white border."""
    h, w = gray.shape
    if min(h, w) < 6:
        return gray
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    bw = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(8, int(w*.65)), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(8, int(h*.85))))
    lines = cv2.bitwise_or(cv2.morphologyEx(bw, cv2.MORPH_OPEN, hk),
                           cv2.morphologyEx(bw, cv2.MORPH_OPEN, vk))
    ink = cv2.bitwise_and(bw, cv2.bitwise_not(lines))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    glyphs = []
    for x,y,cw,ch,area in stats[1:]:
        if area < 2 or ch < 3 or cw >= w*.55:
            continue
        if x <= 1 or x+cw >= w-1:
            continue
        if ch >= h*.89:
            continue
        glyphs.append((x,y,cw,ch))
    if not glyphs:
        return gray
    x0 = max(0, min(x for x,_,_,_ in glyphs)-2)
    x1 = min(w, max(x+cw for x,_,cw,_ in glyphs)+2)
    y0 = max(0, min(y for _,y,_,_ in glyphs)-2)
    y1 = min(h, max(y+ch for _,y,_,ch in glyphs)+2)
    if x1-x0 < 5 or y1-y0 < 5:
        return gray
    return gray[y0:y1, x0:x1]


def prepared_views(crop: np.ndarray, field: str, max_views: int = 2) -> list[tuple[str,np.ndarray]]:
    """Return preserved-aspect BGR images; avoids the 1-channel RapidOCR bug."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop.copy()
    h,w = gray.shape
    originals = [("original",gray)]
    if field == "price" and w > 19:
        # Try without the far-right currency sign. The original remains too.
        narrowed = gray[:,:max(10,int(w*.86))]
        originals.append(("currency_trim",narrowed))
    variations = []
    for label, raw in originals:
        tight = _tight_ink(raw)
        clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(2,2)).apply(tight)
        for suffix,arr in (("gray",tight),("clahe",clahe)):
            if arr.shape[0] < 3 or arr.shape[1] < 3:
                continue
            fac = max(2.0,min(7.0,65.0/max(1,arr.shape[0])))
            interp = cv2.INTER_LINEAR if suffix == "gray" else cv2.INTER_CUBIC
            bigger = cv2.resize(arr,(max(12,round(arr.shape[1]*fac)),max(12,round(arr.shape[0]*fac))),interpolation=interp)
            bigger = cv2.copyMakeBorder(bigger,18,18,18,18,cv2.BORDER_CONSTANT,value=255)
            variations.append((label+"_"+suffix,cv2.cvtColor(bigger,cv2.COLOR_GRAY2BGR)))
    # Prioritize two distinct crops/filters; at most two default views per cell.
    priority = ["currency_trim_gray","currency_trim_clahe","original_gray","original_clahe"] if field=="price" else ["original_gray","original_clahe"]
    variations.sort(key=lambda pair: priority.index(pair[0]) if pair[0] in priority else 99)
    return variations[:max(1,max_views)]


def _rapidocr_read(prepared: np.ndarray, recognize, lock=None) -> tuple[str,float] | None:
    if recognize is None:
        return None
    with (lock if lock is not None else nullcontext()):
        outcome=recognize([prepared])
    if len(outcome)!=1:
        return None
    text, conf=outcome[0]
    return str(text),float(conf)


def _tesseract_read(prepared: np.ndarray, psm: int=7, timeout_seconds: float=3, exe: str="tesseract") -> tuple[str,float] | None:
    if shutil.which(exe) is None:
        return None
    ok, encoded=cv2.imencode(".png",prepared)
    if not ok:
        return None
    cmd=[exe,"stdin","stdout","--oem","1","--psm",str(psm),"-l","eng",
         "-c","tessedit_char_whitelist=0123456789.,", "-c","load_system_dawg=0", "-c","load_freq_dawg=0", "tsv"]
    try:
        out=subprocess.run(cmd,input=encoded.tobytes(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                           timeout=timeout_seconds,check=False)
    except (OSError,subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    tokens=[]; confidence=[]
    for line in out.stdout.decode("utf-8","replace").splitlines()[1:]:
        cells=line.split("\t",11)
        if len(cells)<12:
            continue
        txt=cells[11].strip()
        if not txt:
            continue
        try:score=float(cells[10])/100.0
        except ValueError: score=0.0
        if score<0:continue
        tokens.append(txt);confidence.append(score)
    return ("".join(tokens),sum(confidence)/len(confidence)) if tokens else None


def _easyocr_read(prepared, reader) -> tuple[str,float] | None:
    if reader is None:
        return None
    # Do not hide initialization or model-download failures. The caller logs
    # them per engine and preserves the original receipt.
    result=reader.recognize(prepared, allowlist="0123456789.,", detail=1)
    if not result:return None
    text="".join(str(item[1]) for item in result)
    conf=min(float(item[2]) for item in result)
    return text,conf


def _ocr_candidates(views, recognize, ocr_lock, *, use_tesseract:bool, easyocr_reader=None,
                    callbacks=None, tesseract_exe="tesseract", task=None) -> tuple[dict, list[dict]]:
    evidence=[]
    for i,(variant,image) in enumerate(views):
        for engine in ("rapidocr","tesseract"):
            if engine=="tesseract" and not use_tesseract: continue
            try:
                if callbacks and engine in callbacks:
                    reading=callbacks[engine](image,variant,i,task)
                elif engine=="rapidocr":
                    reading=_rapidocr_read(image,recognize,ocr_lock)
                else:
                    reading=_tesseract_read(image, 7 if i==0 else 8, exe=tesseract_exe)
            except Exception as exc:
                evidence.append(dict(engine=engine, variant=variant, error=type(exc).__name__))
                continue
            if not reading:continue
            raw, confidence=reading
            parsed=parse_numeric(raw)
            evidence.append(dict(engine=engine,variant=variant,text=raw,confidence=round(confidence,4),
                                 number=canon(parsed) if parsed is not None else None))
    groups_before=defaultdict(list)
    for item in evidence:
        if item.get("number") is not None:
            groups_before[item["number"]].append(item)
    # A consensus on the ORIGINAL erroneous value cannot fix the arithmetic.
    # Only request the third engine if primary results do not have an
    # independently corroborated *alternative* for this suspect cell.
    original=(task or {}).get("before")
    alternative_confirmed=any(
        number!=original and len({v["engine"] for v in readings if v.get("confidence", 0)>=.35})>=2
        for number,readings in groups_before.items()
    )
    if easyocr_reader is not None and not alternative_confirmed:
        for variant,image in views[:1]:
            try:
                reading=(callbacks["easyocr"](image,variant,0,task)
                         if callbacks and "easyocr" in callbacks
                         else _easyocr_read(image,easyocr_reader))
            except Exception as exc:
                evidence.append(dict(engine="easyocr",variant=variant,
                                     error=type(exc).__name__,detail=str(exc)[:300]))
                break
            if reading:
                raw,confidence=reading
                number=parse_numeric(raw)
                evidence.append(dict(engine="easyocr",variant=variant,text=raw,
                                     confidence=round(confidence,4),
                                     number=canon(number) if number is not None else None))
    groups=defaultdict(list)
    for item in evidence:
        if item.get("number") is not None:
            groups[item["number"]].append(item)
    candidates={}
    for number,items in groups.items():
        engines={item["engine"] for item in items if item["confidence"]>=0.35}
        strong=sum(item["confidence"]>=0.55 for item in items)
        candidates[number]={"number":number,"engines":sorted(engines),"engines_agree":len(engines)>=2,
                            "strong_votes":strong,"votes":len(items),
                            "max_confidence":max(item["confidence"] for item in items)}
    return candidates,evidence


def recover_mismatched_numbers(receipt:dict, image:np.ndarray, layout:dict, recognize,
                               *, ocr_lock=None, max_cells:int=4, max_views:int=2,
                               use_tesseract:bool=True, easyocr_reader=None,
                               callbacks=None, validate_fn=None,
                               require_two_engines:bool=True, tesseract_exe="tesseract") -> tuple[dict,dict]:
    """Select max 4 suspect cells, then test at most 2-cell OCR corrections.

    Calculation is a validator only. If no independently supported interpretation
    yields an exact total + VAT, do not change the original receipt.
    """
    if validate_fn is None:
        from App.ocr.receipt_editing import update_receipt
        validate_fn=update_receipt
    current=copy.deepcopy(receipt)
    calculation=validate_fn(copy.deepcopy(current))["calculation"]
    report=dict(version=VERSION, triggered=False, reason="", initial_total_status=calculation["status"],
                initial_vat_status=calculation["vat"]["status"],selected=[],candidates=[],applied=[],
                final_total_status=calculation["status"],final_vat_status=calculation["vat"]["status"],
                rapidocr_calls=0,tesseract_calls=0,
                easyocr={"state":"disabled" if easyocr_reader is None else "not_requested"})
    if parse_numeric(calculation.get("expected_total")) is None:
        report["reason"]="printed_total_unavailable"
        return current,report
    if calculation["status"] == "match" and calculation["vat"]["status"] in {"match","not_applicable"}:
        report["reason"]="mathematics_already_valid"
        return current,report
    if calculation["status"] not in {"incomplete","mismatch"} and calculation["vat"]["status"] != "mismatch":
        report["reason"]="no_readable_printed_total_or_nonrecoverable_status"
        return current,report
    report["triggered"]=True
    rows_by_id={r["row"]:r for r in current.get("rows",[])}
    selected=rank_suspect_cells(current,calculation,limit=max_cells,image=image,layout=layout)
    report["selected"]=selected
    options=[]
    for task in selected:
        row=rows_by_id[task["row"]]
        crop=crop_cell(image,layout,row,task["field"])
        if crop is None:continue
        views=prepared_views(crop,task["field"],max_views=max_views)
        if not views:continue
        readings,evidence=_ocr_candidates(views,recognize,ocr_lock,use_tesseract=use_tesseract,
                                         easyocr_reader=easyocr_reader,callbacks=callbacks,tesseract_exe=tesseract_exe,task=task)
        report["rapidocr_calls"]+=len([e for e in evidence if e.get("engine")=="rapidocr"])
        report["tesseract_calls"]+=len([e for e in evidence if e.get("engine")=="tesseract"])
        report["candidates"].append(dict(row=task["row"],field=task["field"],before=task["before"],
                                         evidence=evidence,choices=list(readings.values())))
        for value,metadata in readings.items():
            if value==task["before"]:continue
            if require_two_engines and not metadata["engines_agree"]:continue
            if not require_two_engines and metadata["strong_votes"] < 2:continue
            options.append((task,value,metadata))

    report["easyocr"] = easyocr_reader.diagnostics() if hasattr(easyocr_reader,"diagnostics") else (
        {"state":"provided_reader"} if easyocr_reader is not None else {"state":"disabled"})
    if report["easyocr"].get("state") in {"library_missing","download_failed","initialization_failed"}:
        report.setdefault("warnings",[]).append(
            "EasyOCR включён, но недоступен: " + str(report["easyocr"].get("error", "неизвестная ошибка"))
        )

    # Evaluate all supported single-cell corrections; if multiple distinct
    # assignments balance the receipt, reject ambiguity rather than choose.
    accepted=[]
    for size in (1,2):
        if accepted: break
        for combo in itertools.combinations(options,size):
            if len({(t["row"],t["field"]) for t,_,_ in combo})!=size:continue
            trial=copy.deepcopy(current)
            for task,new,_ in combo:
                target=next(r for r in trial["rows"] if r["row"]==task["row"])
                target[task["field"]]=new
            check=validate_fn(trial)["calculation"]
            if check["status"]=="match" and check["vat"]["status"] in {"match","not_applicable"}:
                accepted.append((combo,trial))
            if len(accepted)>1:break
        if len(accepted)>1:break
    if len(accepted)!=1:
        report["reason"]="ambiguous_mathematical_solutions" if len(accepted)>1 else "no_independently_confirmed_math_solution"
        return current,report
    chosen,trial=accepted[0]
    for task,new,metadata in chosen:
        target=next(r for r in trial["rows"] if r["row"]==task["row"])
        target[task["field"]+"_source"]="numeric_ensemble"
        target[task["field"]+"_ocr"]={"engines":metadata["engines"],"votes":metadata["votes"]}
        report["applied"].append(dict(row=task["row"],field=task["field"],before=task["before"],
                                      after=new,engines=metadata["engines"],reason="ocr_evidence_and_exact_math"))
    final=validate_fn(copy.deepcopy(trial))["calculation"]
    report.update(reason="confirmed",final_total_status=final["status"],final_vat_status=final["vat"]["status"])
    return trial,report


def load_ensemble_options(root=None) -> dict:
    """Read only the existing project-root .env; never create another config."""
    from dotenv import dotenv_values
    project=Path(root).resolve() if root is not None else Path(__file__).resolve().parents[2]
    path=project/".env"
    values=dotenv_values(path,encoding="utf-8-sig") if path.is_file() else {}
    def boolean(name,default):
        raw=str(values.get(name, "1" if default else "0")).strip().lower()
        if raw not in {"1","0","true","false","yes","no"}:
            raise ValueError(f"{name} в .env должен быть 0 или 1")
        return raw in {"1","true","yes"}
    def integer(name,default,lo,hi):
        try:v=int(str(values.get(name,default)))
        except ValueError as exc:raise ValueError(f"{name} должен быть целым") from exc
        if v<lo or v>hi:raise ValueError(f"{name} должен быть {lo}..{hi}")
        return v
    return dict(enabled=boolean("OCR_NUMERIC_ENSEMBLE_ENABLED",True),
                max_cells=integer("OCR_NUMERIC_ENSEMBLE_MAX_CELLS",4,1,8),
                max_views=integer("OCR_NUMERIC_ENSEMBLE_MAX_VIEWS",2,1,4),
                use_tesseract=boolean("OCR_NUMERIC_ENSEMBLE_TESSERACT",True),
                enable_easyocr=boolean("OCR_NUMERIC_ENSEMBLE_EASYOCR",False),
                require_two_engines=boolean("OCR_NUMERIC_ENSEMBLE_REQUIRE_TWO_ENGINES",True),
                tesseract_exe=str(values.get("OCR_NUMERIC_ENSEMBLE_TESSERACT_EXE") or "tesseract"))


# Cache the heavy EasyOCR model in the application process, across requests.
# Only one process-local initializer may run at once; a file lock protects the
# on-disk model files from another worker/process downloading simultaneously.
_EASYOCR_CACHE: dict[str, Any] = {}
_EASYOCR_INIT_LOCK = threading.RLock()


@contextmanager
def _model_download_lock(directory: Path):
    """Cross-process lock for the model directory, using only the stdlib.

    On Windows use msvcrt.locking; on POSIX use fcntl.flock. The tiny lock file
    is intentionally retained so it remains the same inode for future workers.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ".easyocr_download.lock"
    with path.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            handle.seek(0)
            if handle.read(1) == b"":
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class LazyEasyOCRReader:
    """Deferred English-digit OCR with verified on-demand weight downloads.

    Constructing this object is cheap and network-free. The first real
    .recognize call imports easyocr and calls its Reader(download_enabled=True).
    EasyOCR handles integrity checking and re-download of corrupted weights.
    The package itself (pip install easyocr) is NOT installed in a web request.
    """

    def __init__(self, project_root):
        self.model_dir = Path(project_root).resolve() / "models" / "easyocr"
        self.state = "not_requested"
        self.error = None
        self.download_attempted = False
        self.reader_initialized = False
        self._reader = None
        self._failed = False

    def _initialize(self):
        if self._reader is not None:
            return self._reader
        if self._failed:
            raise RuntimeError(f"EasyOCR {self.state}: {self.error or chr(63)}")
        key = str(self.model_dir)
        with _EASYOCR_INIT_LOCK:
            if key in _EASYOCR_CACHE:
                self._reader = _EASYOCR_CACHE[key]
                self.state = "reused_in_memory"
                return self._reader
            try:
                import easyocr
            except ImportError as exc:
                self.state = "library_missing"
                self.error = "Библиотека easyocr не установлена в .venv: python -m pip install easyocr"
                self._failed = True
                raise RuntimeError(self.error) from exc
            try:
                with _model_download_lock(self.model_dir):
                    # This path is EasyOCR's standard model for Reader(['en']).
                    # It is a diagnostic hint, not a substitute for EasyOCR's
                    # own checksum validation (inside Reader).
                    weight = self.model_dir / "english_g2.pth"
                    existing = weight.stat() if weight.is_file() else None
                    before = (existing.st_size, existing.st_mtime_ns) if existing else None
                    self.download_attempted = existing is None
                    self.state = "loading_or_downloading"
                    self._reader = easyocr.Reader(
                        ["en"], gpu=False,
                        model_storage_directory=str(self.model_dir),
                        user_network_directory=str(self.model_dir / "_user_network"),
                        detector=False, recognizer=True,
                        download_enabled=True, verbose=False,
                    )
                    updated = weight.stat() if weight.is_file() else None
                    after = (updated.st_size, updated.st_mtime_ns) if updated else None
                    # EasyOCR also re-downloads an existing but corrupt model.
                    self.download_attempted = self.download_attempted or before != after
                _EASYOCR_CACHE[key] = self._reader
                self.reader_initialized = True
                self.state = "downloaded_and_ready" if self.download_attempted else "loaded_from_disk"
                return self._reader
            except Exception as exc:
                self.state = "download_failed" if self.download_attempted else "initialization_failed"
                self.error = f"{type(exc).__name__}: {exc}"[:500]
                self._failed = True
                raise RuntimeError(f"EasyOCR {self.state}: {self.error}") from exc

    def recognize(self, image, *args, **kwargs):
        return self._initialize().recognize(image, *args, **kwargs)

    def diagnostics(self):
        return dict(state=self.state, model_path=str(self.model_dir),
                    download_attempted=self.download_attempted,
                    reader_initialized=self.reader_initialized,
                    error=self.error)


def optional_easyocr_reader(project_root, enabled=False):
    """Return a lazy proxy only when enabled; no work on mere construction.

    When actually needed, a missing model is downloaded once and cached in
    models/easyocr. Missing Python dependency or download failures are reported
    explicitly in numeric_recovery.json, never silently treated as success.
    """
    return LazyEasyOCRReader(project_root) if enabled else None
