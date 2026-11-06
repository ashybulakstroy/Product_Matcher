import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from App.ocr.manual_mappings import ManualMappings, ocr_key
from App.ocr.ocr_catalog_eval import CatalogMatcher, normalize_name, review_reason, needs_review
from App.ocr.receipt_editing import apply_edit, update_receipt
from App.ocr.web_pipeline import ReceiptPipeline


def entry(name="Правильный товар", articles=("123",)):
    return dict(name=name, normalized=normalize_name(name), sources=[dict(article=a) for a in articles])


def automatic(item=None, score=75):
    item = item or entry("Неправильный товар")
    best = dict(item, score=score, raw_similarity=score, missing_query_numbers=[])
    return dict(best=best, top_matches=[best], status="review", gap=2)


def test_persistent_learning_and_replacement_keep_history(tmp_path):
    path = tmp_path/"knowledge.json"
    knowledge = ManualMappings(path)
    assert knowledge.lookup("OCR") is None and not path.exists()
    knowledge.remember("OCR товар 5", entry(), "123", "job1", 2)
    assert ManualMappings(path).lookup(" ocr  ТОВАР\n5 ")["catalog_name"] == "Правильный товар"
    assert ManualMappings(path).lookup("OCRТОВАР5")["catalog_name"] == "Правильный товар"
    knowledge.remember("OCR товар 5", entry("Другой товар"), "123", "job2", 3)
    assert knowledge.lookup("OCR товар 5")["catalog_name"] == "Другой товар"
    document = json.loads(path.read_text(encoding="utf-8"))
    assert len(document["mappings"]) == 1
    assert len(document["history"]) == 2
    assert document["history"][1]["before"]["catalog_name"] == "Правильный товар"


def test_key_does_not_conflate_other_numbers_or_punctuation():
    assert ocr_key("OCR  A\n5") == ocr_key("ocr a 5")
    assert ocr_key("Пила CT 15074") == ocr_key("пилаct15074")
    assert ocr_key(" П И Л А\tCT15074") == ocr_key("пилаct15074")
    assert ocr_key("OCR A5") != ocr_key("OCR A6")
    assert ocr_key("40/32") != ocr_key("40/33")
    assert ocr_key("Товар Ø5") != ocr_key("Товар 05")


def test_manual_choice_overrides_only_nonperfect_automatic_match(tmp_path):
    knowledge = ManualMappings(tmp_path/"knowledge.json")
    selected = entry(articles=("123", "456"))
    knowledge.remember("OCR", selected, "456")
    index = {selected["normalized"]:selected}
    result = knowledge.resolve("OCR", index, automatic())
    assert result["source"] == "manual_mapping"
    assert result["best"]["article"] == "456"
    assert result["automatic_match"]["best"]["score"] == 75
    row = dict(row=1,kind="product",name_raw="OCR",quantity_raw="1",price_raw="100",match=result)
    receipt = update_receipt(dict(rows=[row],total_raw="100"))
    assert row["name"] == selected["name"] and row["article"] == "456"
    assert row["name_origin"] == "manual_mapping" and receipt["can_finalize"]
    assert review_reason(row) == ""
    assert not needs_review(dict(kind="product",status="manual_mapping",similarity=100,articles="123;456"))
    perfect = automatic(score=100)
    assert knowledge.resolve("OCR", index, perfect) is perfect


def test_removed_catalog_entry_or_changed_article_does_not_apply(tmp_path):
    knowledge = ManualMappings(tmp_path/"knowledge.json")
    item = entry()
    knowledge.remember("OCR", item, "123")
    fallback = automatic()
    assert knowledge.resolve("OCR", {}, fallback) is fallback
    changed = entry(articles=("999",))
    assert knowledge.resolve("OCR", {changed["normalized"]:changed}, fallback) is fallback
    assert knowledge.resolve("Other OCR", {item["normalized"]:item}, fallback) is fallback


def test_matcher_checks_new_knowledge_even_with_cached_automatic_match(tmp_path):
    knowledge = ManualMappings(tmp_path/"knowledge.json")
    items = [entry("Пила CROWN CT15074"),entry("Дверь Лион 50951")]
    matcher = CatalogMatcher(items, knowledge=knowledge)
    raw = "Непонятный OCR жжж 987654"
    before = matcher.match(raw)
    assert before["best"]["score"] < 100
    knowledge.remember(raw,items[1],"123")
    assert matcher.match(raw)["best"]["name"] == items[1]["name"]
    knowledge.remember(raw,items[0],"123")
    assert matcher.match(raw)["best"]["name"] == items[0]["name"]
    assert before.get("source") is None


def test_only_confirmed_catalog_selection_trains_on_raw_ocr(tmp_path):
    knowledge = ManualMappings(tmp_path/"knowledge.json")
    pipeline = ReceiptPipeline(knowledge)
    selected = entry()
    pipeline.catalog_entry = lambda key:selected if key == selected["normalized"] else None
    raw = "Ошибка OCR 123"
    receipt = update_receipt(dict(rows=[dict(row=2,kind="product",name_raw=raw,quantity_raw="1",price_raw="10",match=automatic(score=55))],total_raw="10"))
    payload = dict(revision=0,row=2,catalog_key=selected["normalized"],confirm_name=True)
    edited = apply_edit(receipt,payload,pipeline.catalog_entry)
    assert pipeline.remember_selection(receipt,edited,payload,"job")
    assert knowledge.lookup(raw)["catalog_key"] == selected["normalized"]
    assert edited["rows"][0]["name_raw"] == raw
    assert pipeline.remember_selection(receipt,edited,dict(row=2,price="11"),"job") is None
    assert pipeline.remember_selection(receipt,edited,dict(row=2,name="Ручной текст",confirm_name=True),"job") is None


def test_empty_ocr_rejected(tmp_path):
    with pytest.raises(ValueError):
        ManualMappings(tmp_path/"k.json").remember("  ",entry())


def test_parallel_writers_and_manual_text_file_changes(tmp_path):
    path = tmp_path/"knowledge.json"
    first = ManualMappings(path)
    first.initialize()
    def save(index):
        ManualMappings(path).remember(f"OCR {index}",entry(),"123")
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(save,range(12)))
    document = json.loads(path.read_text(encoding="utf-8"))
    assert len(document["mappings"]) == len(document["history"]) == 12
    assert first.lookup("ocr0")["article"] == "123"
    document["mappings"]["ocr0"]["catalog_name"] = "Исправлено в текстовом файле"
    path.write_text(json.dumps(document,ensure_ascii=False,indent=2),encoding="utf-8")
    assert first.lookup("O C R 0")["catalog_name"] == "Исправлено в текстовом файле"


def test_invalid_json_is_not_overwritten(tmp_path):
    path = tmp_path/"knowledge.json"
    path.write_text("invalid json",encoding="utf-8")
    with pytest.raises(ValueError):
        ManualMappings(path).remember("OCR",entry())
    assert path.read_text(encoding="utf-8") == "invalid json"
