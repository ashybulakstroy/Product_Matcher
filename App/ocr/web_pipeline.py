"""Single-photo pipeline shared by the local web server and integration checks."""
from __future__ import annotations

import csv
import io
import json
import re
import threading
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from PIL import Image, ImageOps
from rapidocr_onnxruntime import RapidOCR
from rapidfuzz import fuzz, process

from App.ocr.ocr_activity import preliminary_row_count, completed_text_rows, row_totals, emit_status
from App.ocr.visual_sync import emit as emit_visual, raw_row as visual_raw_row, ready_cells
from App.ocr.receipt_rectification import rectify_table
from App.ocr.receipt_orientation_detector import orient_receipt
from App.ocr.rotation_rectification import rotate_table_crop
from App.ocr.ocr_catalog_eval import (ROOT, PASS_SIMILARITY, CatalogMatcher, aligned_lines,
                                  choose_row_boundaries, enforce_cell_boundaries, load_catalog,
                                  needs_review, parse_rows, review_reason, table_layout)
from App.ocr.ocr_run import MODEL
from App.ocr.table_boundary import detect_table_boundary
from App.ocr.full_grid_boundary import detect_full_grid_boundary
from App.ocr.receipt_roi_recovery import recover_table_boundary, recover_compact_table
from App.ocr.table_verification import verify_table_candidate
from App.ocr.receipt_editing import initialize_row, save_receipt, update_receipt
from App.ocr.manual_mappings import ManualMappings
from App.ocr.quantity_cells import recover_quantities
from App.ocr.numeric_cell_split import recover as recover_numeric_cells
from App.ocr.row_cell_split import repair as split_merged_product_rows
from App.ocr.numeric_recovery import recover_numeric_receipt
from App.ocr.selective_number_ocr import recover_number_cells
from App.ocr.selective_text_recovery import recover_text_names

MAX_PIXELS = 25_000_000


class _RetryOrientationAfterNoText(ValueError):
    """Request one full retry from the EXIF-normalized source image."""


def decode_photo(data):
    try:
        with Image.open(io.BytesIO(data)) as photo:
            if photo.format not in {"JPEG", "PNG", "WEBP"}:
                raise ValueError("Поддерживаются фотографии JPG, PNG и WEBP.")
            if photo.width * photo.height > MAX_PIXELS:
                raise ValueError("Слишком большое фото. Максимальное разрешение — 25 мегапикселей.")
            rotated = photo.getexif().get(274, 1) != 1
            oriented = ImageOps.exif_transpose(photo)
            if oriented.mode in {"RGBA", "LA"} or "transparency" in oriented.info:
                rgba = oriented.convert("RGBA")
                background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                oriented = Image.alpha_composite(background, rgba)
            image = cv2.cvtColor(np.asarray(oriented.convert("RGB")), cv2.COLOR_RGB2BGR)
    except (OSError, Image.DecompressionBombError) as error:
        raise ValueError("Не удалось прочитать изображение. Выберите исправный JPG, PNG или WEBP.") from error
    if min(image.shape[:2]) < 80:
        raise ValueError("Изображение слишком маленькое для распознавания чека.")
    return image, rotated


def grid_cells(layout, rows=None):
    xs = [layout.get("body_left", 16), layout["name_left"], layout["name_right"], layout["quantity_right"], layout["right"]]
    bands = [(index+1, start, end, "cell") for index, (start, end) in enumerate(zip(layout["row_boundaries"], layout["row_boundaries"][1:]))]
    if rows is not None:
        bands = [(row["row"], *row["y_range"], row["kind"]) for row in rows]
    return [dict(id=f'{row_id}-{role}', row=row_id, role=role, kind=kind,
                 box=[[left, start], [right, start], [right, end], [left, end]])
            for row_id, start, end, kind in bands for role, left, right in
            zip(("number", "name", "quantity", "price"), xs, xs[1:])]


class _ObservedCall:
    def __init__(self, delegate, callback):
        self.delegate, self.callback = delegate, callback

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def __call__(self, *args, **kwargs):
        result = self.delegate(*args, **kwargs)
        self.callback(result)
        return result


class ReceiptPipeline:
    def __init__(self, knowledge=None, *, split_product_rows=True):
        self.split_product_rows = split_product_rows
        self.knowledge = knowledge if knowledge is not None else ManualMappings()
        self.engine = None
        self.matcher = None
        self.catalog_signature = None
        self.catalog_count = 0
        self.ocr_lock = threading.Lock()
        self.catalog_lock = threading.RLock()
        self.catalog_index = {}

    def ensure_catalog(self):
        directory = ROOT / "data/reference/torgsoft_ref"
        signature = tuple((p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in sorted(directory.glob("*.xlsx")) if not p.name.startswith("~$"))
        with self.catalog_lock:
            if signature != self.catalog_signature:
                entries, workbooks = load_catalog(directory)
                self.matcher, self.catalog_count = CatalogMatcher(entries, knowledge=self.knowledge), len(entries)
                self.catalog_index = {entry["normalized"]:entry for entry in entries}
                self.catalog_signature = signature

    def catalog_entry(self, key):
        self.ensure_catalog()
        return self.catalog_index.get(key)

    def remember_selection(self, before, after, payload, job_id):
        if not payload.get("catalog_key") or not payload.get("confirm_name"):
            return None
        previous = next((row for row in before["rows"] if row["row"] == payload.get("row")), None)
        if not previous or previous["kind"] != "product":
            return None
        if not (payload.get("learn_mapping") or previous.get("needs_name_review") or
                previous.get("catalog_key") != payload["catalog_key"]):
            return None
        selected = next(row for row in after["rows"] if row["row"] == previous["row"])
        entry = self.catalog_entry(selected["catalog_key"])
        return self.knowledge.remember(previous["name_raw"], entry, selected["article"], job_id, previous["row"])

    def search_catalog(self, query):
        self.ensure_catalog()
        query = str(query).strip()[:200]
        if len(query) < 2:
            return []
        from App.ocr.ocr_catalog_eval import normalize_name
        direct = [entry for entry in self.matcher.entries if any(query.casefold() == source["article"].casefold() for source in entry["sources"])]
        ranked = [self.matcher.entries[index] for _, _, index in process.extract(normalize_name(query), self.matcher.names, scorer=fuzz.WRatio, limit=8)]
        found = {}
        for entry in direct + ranked:
            found.setdefault(entry["normalized"], dict(key=entry["normalized"], name=entry["name"],
                articles=sorted({source["article"] for source in entry["sources"] if source["article"]})))
        return list(found.values())[:8]

    def resources(self):
        self.ensure_catalog()
        if self.engine is None:
            self.engine = RapidOCR(rec_model_path=str(MODEL), rec_img_shape=[3, 48, 320], text_score=.3)

    def recognize(self, path, job, target, progress, on_partial=None, row_boundaries=None):
        with self.ocr_lock:
            detector, recognizer = self.engine.text_detector, self.engine.text_recognizer

            detected = []
            def boxes_ready(result):
                boxes = result[0]
                detected[:] = list(self.engine.sorted_boxes(boxes)) if boxes is not None else []
                job.publish("boxes", target=target, boxes=boxes.tolist() if boxes is not None else [],
                            title="Обнаружены текстовые блоки", progress=progress)

            self.engine.text_detector = _ObservedCall(detector, boxes_ready)
            class StreamingRecognizer:
                def __getattr__(self, name):
                    return getattr(recognizer, name)

                def __call__(inner, images):
                    results, elapsed = [], 0.
                    size = max(1, recognizer.rec_batch_num)
                    for start in range(0, len(images), size):
                        batch, seconds = recognizer(images[start:start+size])
                        results.extend(batch)
                        elapsed += seconds
                        if on_partial and detected:
                            partial = [dict(text=str(text), confidence=float(confidence), box=np.asarray(box).tolist(),
                                            box_original=np.asarray(box).tolist())
                                       for box, (text, confidence) in zip(detected, results)
                                       if confidence >= self.engine.text_score]
                            on_partial(partial, pending_boxes=detected[len(results):])
                            rows_done, rows_total = completed_text_rows(detected, len(results), row_boundaries) if row_boundaries is not None else (0, 0)
                            if rows_total:
                                emit_status(job, "ocr", "Распознаём товары, количество и цены",
                                            rows_done=rows_done, rows_total=rows_total,
                                            progress=progress+int(6*len(results)/len(images)))
                            else:
                                emit_status(job, "ocr", "Считываем товарные данные",
                                            details="Продолжаем распознавание текста",
                                            progress=progress+int(6*len(results)/len(images)))
                    return results, elapsed
            self.engine.text_recognizer = StreamingRecognizer()
            try:
                result, times = self.engine(str(path))
            finally:
                self.engine.text_detector, self.engine.text_recognizer = detector, recognizer
        return result or []

    def process(self, job):
        try:
            return self._process_attempt(job, orientation_mode="normal")
        except _RetryOrientationAfterNoText:
            job.publish("message", title="Текст или таблица не обнаружены; проверяем 4 поворота чека")
            return self._process_attempt(job, orientation_mode="recovery")

    def _process_attempt(self, job, *, orientation_mode):
        started = perf_counter()
        job.phase("input", 5, "Подготавливаем фотографию")
        emit_status(job, "input", "Проверяем ориентацию и качество фотографии")
        image, exif_rotated = decode_photo((job.directory / "upload.bin").read_bytes())
        # Portrait normal mode is intentionally free: no orientation OCR and
        # no early model initialization. Landscape and recovery reuse RapidOCR.
        if orientation_mode == "recovery" or image.shape[0] < image.shape[1]:
            self.resources()
        image, orientation_info = orient_receipt(
            image, engine=self.engine, ocr_lock=self.ocr_lock,
            mode=orientation_mode,
        )
        job.json_file(
            "orientation.json" if orientation_mode == "normal" else "orientation_recovery.json",
            orientation_info,
        )
        if orientation_mode == "recovery" and (
            orientation_info["status"] != "OK"
        ):
            raise ValueError(
                "Не удалось достоверно установить ориентацию фотографии. "
                "Требуется ручная проверка фото."
            )
        job.image(
            "original", image, "Ориентированное фото",
            f"Режим: {orientation_mode}; EXIF: {exif_rotated}; "
            f"поворот: {orientation_info['rotation_clockwise']}°; "
            f"статус: {orientation_info['status']}",
            role="original",
        )
        job.phase("geometry", 13, "Ищем границы товарной таблицы")
        boundary = detect_table_boundary(image)
        boundary, paper_roi_report = recover_table_boundary(
            image, detect_table_boundary, boundary, detect_full_grid_boundary,
        )
        if paper_roi_report['status'] != 'not_needed':
            job.json_file("paper_roi_recovery.json", paper_roi_report)
            if paper_roi_report["status"] == "recovered":
                job.publish("message", title="Товарная таблица найдена на узком чеке без повторного OCR",
                            progress=21)
        header_ocr_calls = 0
        def recognize_table_header(header):
            nonlocal header_ocr_calls
            self.resources()
            with self.ocr_lock:
                header_ocr_calls += 1
                return self.engine(header)[0]
        emit_status(job, 'geometry', 'Проверяем сетку и заголовок таблицы',
                    details='Подтверждаем область перед распознаванием товаров')
        verification = verify_table_candidate(image, boundary, recognize_table_header)
        selection = dict(initial=verification, fallback=None)
        if verification['status'] == 'rejected':
            alternative, compact_report = recover_compact_table(
                image, detect_table_boundary, boundary, recognize_table_header)
            selection['fallback'] = compact_report
            if compact_report['status'] == 'recovered':
                boundary = alternative
                verification = compact_report['verification']
                job.publish('message', title='Таблица найдена и проверена по заголовку и сетке', progress=22)
            elif verification['status'] == 'rejected':
                boundary = dict(boundary, polygon=None)
        selection.update(final=verification, header_ocr_calls=header_ocr_calls,
                         existing_boundary_preserved=verification['status'] == 'uncertain')
        job.json_file('table_verification.json', selection)
        polygon = boundary["polygon"]
        if polygon is None:
            if orientation_mode == "normal":
                raise _RetryOrientationAfterNoText(
                    "Не обнаружена таблица; возможна ошибочная ориентация"
                )
            raise ValueError(
                "Не удалось найти рамку таблицы после проверки ориентации. "
                "Нужна ручная проверка снимка."
            )
        job.publish("contour", target="original", polygon=polygon.tolist(), title="Граница таблицы найдена", progress=24)
        emit_visual(job, "geometry_found", target="original", message="Контур таблицы найден")
        emit_status(job, "geometry", "Найдена товарная таблица", details="Определяем её границы")
        warnings = list(boundary["warnings"])
        header_bottom = max(0, int(polygon[:, 1].min())-4)
        if header_bottom >= 80:
            pad = int(max(15, np.ptp(polygon[:, 0])*.1))
            left = max(0, int(polygon[:, 0].min())-pad)
            right = min(image.shape[1], int(polygon[:, 0].max())+pad)
            job.image("header", image[:header_bottom, left:right], "Crop · шапка документа", "Номер заказа и дата, если они читаются")
        lo = np.maximum(0, np.floor(polygon.min(axis=0)).astype(int)-16)
        hi = np.minimum([image.shape[1], image.shape[0]], np.ceil(polygon.max(axis=0)).astype(int)+17)
        crop = image[lo[1]:hi[1], lo[0]:hi[0]]
        job.image("crop", crop, "Crop · область таблицы", "Отдельное изображение, исходное фото сохранено")
        job.publish("region", target="original", box=[lo.tolist(), [int(hi[0]), int(lo[1])], hi.tolist(), [int(lo[0]), int(hi[1])]], progress=29)
        job.phase("geometry", 32, "Проверяем волны бумаги и выпрямляем таблицу")
        if "partial_contour_replaced_by_full_grid_hypothesis" in boundary["warnings"]:
            rectified, map_x, map_y, metadata = rotate_table_crop(image, polygon)
        else:
            rectified, map_x, map_y, metadata = rectify_table(
                image, polygon, trust_top=boundary["status"]=="detected")
        warnings.extend(metadata["warnings"])
        subtitle = {"wave_grid":"Выпрямлены волны по внутренней сетке; ячейка итога сохранена",
                    "original_crop_fallback":"Сохранена исходная геометрия: требуется проверка"}.get(metadata["method"],"Выпрямлены границы; нижняя ячейка итога сохранена")
        job.publish("message",title=subtitle,progress=37)
        emit_status(job, "geometry", "Выпрямляем таблицу", details=subtitle)
        job.image("rectified", rectified, "Rectify · рабочая таблица", subtitle)
        emit_visual(job, "geometry_ready", target="rectified", message="Таблица выпрямлена · готовим распознавание")
        np.savez_compressed(job.directory / "rectification_map.npz", map_x=map_x, map_y=map_y)
        job.json_file("rectification.json", metadata)
        job.phase("layout", 40, "Находим строки и столбцы")
        layout = table_layout(rectified, metadata)
        layout.setdefault("body_left",metadata["margin"])
        estimated_rows = preliminary_row_count(layout.get("row_boundaries", []))
        emit_status(job, "layout", "Определяем товарные строки",
                    details=(f"Предварительно выделено {estimated_rows} строк таблицы"
                             if estimated_rows else "Границы строк уточняются"))
        job.publish("cells", target="rectified", cells=grid_cells(layout), title="Предварительная разметка ячеек", progress=44)
        job.phase("ocr", 47, "Готовим локальный OCR и справочники")
        self.resources()

        # Header OCR is separate from the table. It cannot contaminate product rows.
        header_lines = []
        header_bottom = max(0, int(polygon[:, 1].min())-4)
        if header_bottom >= 80:
            emit_visual(job, "scan_start", target="header", message="Читаем шапку чека")
            try:
                header_result = self.recognize(job.directory / "header.png", job, "header", 50)
            finally:
                emit_visual(job, "scan_stop", target="header")
            header_lines = [dict(text=str(text), confidence=float(confidence), box=box) for box, text, confidence in header_result]
            job.json_file("ocr_header.json", dict(lines=header_lines))
            header_text = "\n".join(line["text"] for line in header_lines)
            date = re.search(r"\b\d{2}[./]\d{2}[./]20\d{2}\b", header_text)
            number = re.search(r"(?:тапсырыс|заказ|№|\bno\b|\bne\b)[^\d\n]{0,15}(\d{6,})", header_text, re.IGNORECASE)
            job.publish("header", date=date.group(0) if date else "", order_number=number.group(1) if number else "")
        job.phase("ocr", 58, "Сканируем текст в ячейках")
        emit_status(job, "ocr", "Распознаём товары, количество и цены",
                    details="Читаем строки таблицы")
        visual_seen_rows = set()
        def draft_ready(partial, pending_boxes=()):
            provisional = aligned_lines(dict(lines=partial), metadata)
            boundaries, _ = choose_row_boundaries(provisional, layout)
            draft = parse_rows(provisional, layout, boundaries)
            for row in draft:
                row.update(name=row["name_raw"], quantity=row["quantity_raw"], price=row["price_raw"],
                           stage="ocr", match=None, similarity=None, articles=[], needs_review=False,
                           ocr_name_confidence_pct=round(row["ocr_confidence"]*100, 2))
            job.publish("draft_rows", rows=draft, cells=grid_cells(layout, draft),
                        ready_cells=ready_cells(draft, layout, pending_boxes),
                        title="Чек собирается из OCR", progress=64)
            for draft_row in draft:
                row_id = draft_row.get("row")
                if (row_id not in visual_seen_rows and draft_row.get("kind") in {"product", "tax", "service"}
                        and str(draft_row.get("name_raw") or "").strip()):
                    visual_seen_rows.add(row_id)
                    emit_visual(job, "row_ready", row=visual_raw_row(draft_row))
        emit_visual(job, "scan_start", target="rectified", message="Распознаём товарные строки")
        try:
            raw_result = self.recognize(job.directory / "rectified.png", job, "rectified", 64,
                                        on_partial=draft_ready, row_boundaries=layout.get("row_boundaries"))
        finally:
            emit_visual(job, "scan_stop", target="rectified")
        lines = []
        for box, text, confidence in raw_result:
            box = np.asarray(box, np.float32)
            mx = np.clip(box[:, 0], 0, map_x.shape[1]-1).reshape(1, -1)
            my = np.clip(box[:, 1], 0, map_x.shape[0]-1).reshape(1, -1)
            original_box = np.column_stack([cv2.remap(mapping, mx, my, cv2.INTER_LINEAR).reshape(-1) for mapping in (map_x, map_y)])
            lines.append(dict(text=str(text), confidence=float(confidence), box=box.tolist(), box_original=original_box.tolist()))
        if not lines:
            if orientation_mode == "normal":
                raise _RetryOrientationAfterNoText(
                    "RapidOCR не прочёл ни одного текстового блока"
                )
            raise ValueError(
                "Текст не распознан даже после повторного определения ориентации."
            )
        job.json_file("ocr_rectified.json", dict(lines=lines, variant="rectified"))
        (job.directory / "ocr_rectified.txt").write_text("\n".join(line["text"] for line in lines), encoding="utf-8")
        job.phase("layout", 74, "Объединяем строки товара и отделяем НДС")
        emit_visual(job, "checking", message="Собираем товары, проверяем количество, цены и НДС")
        emit_status(job, "layout", "Собираем товарные строки", details="Отделяем НДС от товаров")
        aligned = aligned_lines(dict(lines=lines), metadata)
        boundaries, segmentation = choose_row_boundaries(aligned, layout)
        layout["row_boundaries"] = enforce_cell_boundaries(aligned, layout, boundaries)
        rows = parse_rows(aligned, layout, layout["row_boundaries"])
        total_candidates = [line for line in aligned if line["center"][1] >= layout["body_bottom"]-5
                            and line["center"][0] > layout["name_right"] and re.search(r"\d[.,]\d{2}", line["text"])]
        total = max(total_candidates,key=lambda line:line["center"][1])["text"] if total_candidates else ""
        if self.split_product_rows:
            split_rows,row_split_check = split_merged_product_rows(dict(rows=rows,total_raw=total),layout,lines,rectified)
        else:
            split_rows,row_split_check = dict(rows=rows),[dict(applied=False,reason='disabled_for_independent_baseline')]
        rows = split_rows['rows']
        rt = row_totals(rows)
        emit_status(job, "layout", "Определяем товары и НДС",
                    details=f"Найдено {rt['products']} товаров, строк НДС: {rt['tax_rows']}")
        job.json_file('row_split.json',row_split_check)
        restored_count = sum(item['applied'] for item in row_split_check)
        if restored_count:
            job.publish('message',title=f'Склеенные товарные строки разделены: {restored_count}; печатная граница, числа и итог проверены',progress=76)
        rows, number_report = recover_number_cells(
            rectified, rows, layout,
            lambda crops: self.engine.text_recognizer(crops)[0],
            ocr_lock=self.ocr_lock,
        )
        job.json_file('selective_number_ocr.json', number_report)
        if number_report['recovered']:
            job.publish(
                'message',
                title=f"Номера строк прочитаны отдельно: {number_report['recovered']}",
                progress=76,
            )
        emit_status(job, "check", "Проверяем количество и цены", details="Уточняем неоднозначные значения")
        with self.ocr_lock:
            recovered,quantity_check,_ = recover_quantities(dict(rows=rows,total_raw=total),rectified,layout,
                lambda crops:self.engine.text_recognizer(crops)[0])
        job.json_file("quantity_cells.json",quantity_check)
        if quantity_check["attempts"]:
            if quantity_check["applied"]:
                recovered_rows = {row["row"]:row for row in recovered["rows"] if row.get("quantity_source")=="cell_ocr"}
                for row in rows:
                    if row["row"] in recovered_rows:
                        found = recovered_rows[row["row"]]
                        row.update(quantity=found["quantity"],quantity_source="cell_ocr",quantity_ocr=found["quantity_ocr"])
                        if found.get('quantity_unverified'):
                            row['quantity_unverified'] = True
                quantity_message = (f"Количество прочитано по ячейкам: {len(recovered_rows)}; итог и НДС требуют проверки"
                                    if quantity_check.get('provisional_rows') else
                                    f"Количество прочитано отдельно по ячейкам: {len(recovered_rows)}; сумма и НДС проверены")
                job.publish("message",title=quantity_message,progress=77)
            else:
                job.publish("message",title="Повторное чтение количеств не подтверждено; строки оставлены на проверку",progress=77)
        split_receipt,numeric_split = recover_numeric_cells(dict(rows=rows,total_raw=total),layout,lines)
        rows = split_receipt['rows']
        job.json_file('numeric_split.json',numeric_split)
        split_count = sum(item['applied'] for item in numeric_split)
        if split_count:
            job.publish('message',title=f'Количество и цена разделены по исходному OCR: {split_count}; сумма и НДС проверены',progress=78)
        job.phase('numeric',78,'Проверяем сумму НДС и числовые поля')
        job.publish('message',title='Проверяем сумму НДС и итог чека',progress=78)
        recovered_receipt=None
        try:
            recovered_receipt,numeric_report=recover_numeric_receipt(
                dict(rows=rows,total_raw=total),rectified,layout,aligned,
                lambda images:self.engine.text_recognizer(images)[0],ocr_lock=self.ocr_lock)
            rows,total=recovered_receipt['rows'],recovered_receipt['total_raw']
            job.json_file('numeric_recovery.json',numeric_report)
            if numeric_report['vat_triggered']:
                job.publish('message',title='Дополнительно проверена сумма НДС',progress=78)
            if numeric_report.get('ensemble', {}).get('triggered'):
                job.publish('message',title='Дополнительно проверены подозрительные числовые поля',progress=79)
            if numeric_report['total_triggered']:
                job.publish('message',title='Повторно проверены количество, цены, НДС и итог',progress=79)
            if numeric_report['applied_count']:
                job.publish('message',title=f"Уточнено числовых полей: {numeric_report['applied_count']}",progress=79)
        except Exception as error:
            job.json_file('numeric_recovery.json',dict(status='error',detail=str(error)))
            job.publish('message',title='Повторная проверка чисел недоступна; исходный OCR сохранён',progress=79)
        try:
            rows, text_recovery_report = recover_text_names(
                rows, aligned, rectified, layout,
                rapidocr=lambda crop: self.engine(crop)[0], ocr_lock=self.ocr_lock,
                output_dir=job.directory / 'text_recovery',
            )
            job.json_file('selective_text_recovery.json', text_recovery_report)
            count = len(text_recovery_report['stage1_applied']) + len(text_recovery_report['stage2_applied'])
            if count:
                job.publish('message', title=f'Уточнено наименований по OCR: {count}', progress=79)
        except Exception as error:
            job.json_file('selective_text_recovery.json', dict(status='error', detail=str(error)))
            job.publish('message', title='Повторная проверка названий недоступна; исходный текст сохранён', progress=79)
        cells = grid_cells(layout, rows)
        job.json_file("layout.json", dict(layout=layout, cells=cells, segmentation=segmentation))
        job.publish("cells", target="rectified", cells=cells, title="Ячейки выделены, НДС — отдельная строка", progress=79)
        for row in rows:
            row.update(name=row.get('name_recovered', row["name_raw"]), quantity=row.get("quantity",row["quantity_raw"]), price=row.get("price",row["price_raw"]), stage="ocr",
                       match=None, articles=[], similarity=None, needs_review=False,
                       ocr_name_confidence_pct=round(row["ocr_confidence"]*100, 2))
        job.publish("draft_rows", rows=rows, ready_cells=ready_cells(rows, layout),
                    pass_similarity=PASS_SIMILARITY, title="OCR готов. Начинаем подбор названий", progress=80)
        for complete_row in rows:
            if (complete_row.get("row") not in visual_seen_rows
                    and complete_row.get("kind") in {"product", "tax", "service"}
                    and str(complete_row.get("name_raw") or "").strip()):
                visual_seen_rows.add(complete_row["row"])
                emit_visual(job, "row_ready", row=visual_raw_row(complete_row))
        emit_visual(job, "rows_ready", message="Товарные строки собраны · значения предварительные")
        job.phase("matching", 81, "Сопоставляем товары со справочником")
        emit_visual(job, "matching", message="Сопоставляем товары с номенклатурой")
        total_products = sum(row.get("kind") == "product" for row in rows)
        matched_products = 0
        emit_status(job, "matching", "Сопоставляем товары со справочником",
                    details=f"Всего товаров: {total_products}")
        for index, row in enumerate(rows):
            row["match"] = self.matcher.match(row.get('name_recovered', row["name_raw"])) if row["kind"] == "product" else dict(status="excluded_"+row["kind"], best=None, top_matches=[])
            row["ocr_name_confidence_pct"] = round(row["ocr_confidence"]*100, 2) if row["name_raw"].strip() else None
            best = row["match"]["best"] or {}
            row["articles"] = sorted({source["article"] for source in best.get("sources", []) if source["article"]})
            row["similarity"] = best.get("score")
            row["review_reason"] = review_reason(row)
            row["needs_review"] = needs_review(dict(kind=row["kind"], status=row["match"]["status"],
                                                    similarity=best.get("score", ""), articles="; ".join(row["articles"])))
            row["stage"] = "matched"
            initialize_row(row, PASS_SIMILARITY)
            job.publish("row", row=row, progress=81+int(15*(index+1)/max(1, len(rows))), title=f"Обработана строка {index+1} из {len(rows)}")
            if row["kind"] == "product":
                matched_products += 1
                emit_status(job, "matching", "Сопоставляем названия товаров",
                            matched_products=matched_products, total_products=total_products)

        header_text = "\n".join(line["text"] for line in header_lines)
        date = re.search(r"\b\d{2}[./]\d{2}[./]20\d{2}\b", header_text)
        number = re.search(r"(?:тапсырыс|заказ|№|\bno\b|\bne\b)[^\d\n]{0,15}(\d{6,})", header_text, re.IGNORECASE)
        products = [row for row in rows if row["kind"] == "product"]
        result = dict(schema_version="torgsoft-review-1.0", filename=job.filename, job_id=job.id,
                      order_number=number.group(1) if number else "", date=date.group(0) if date else "",
                      rows=rows, total_raw=total, product_count=len(products),
                      review_count=sum(row["needs_review"] for row in products), pass_similarity=PASS_SIMILARITY,
                      catalog_names=self.catalog_count, warnings=warnings, elapsed_ms=round((perf_counter()-started)*1000),
                      geometry_status=boundary["status"], rectification_method=metadata["method"],
                      raw_text="\n".join(line["text"] for line in lines),
                      downloads={key:f"/api/jobs/{job.id}/files/{name}" for key, name in
                                 (("json", "receipt.json"), ("csv", "receipt.csv"), ("text", "ocr_rectified.txt"))})
        if recovered_receipt and recovered_receipt.get('expected_total_source')=='numeric_recovery':
            for key in ('expected_total','expected_total_source','expected_total_ocr'):
                result[key]=recovered_receipt[key]
        job.phase('assemble',97,'Собираем электронный чек')
        emit_status(job, "check", "Проверяем цены, НДС и итог",
                    details="Вычисляем суммы по прочитанным строкам")
        update_receipt(result)
        save_receipt(job.directory, result)
        emit_status(job, "check", "Данные чека подготовлены",
                    details=f"{len(products)} товаров, требуют проверки: {result['review_count']}")
        return result
