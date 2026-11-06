import io
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from App.ocr.web_pipeline import decode_photo, grid_cells
from App.web_server import create_server


def test_combined_server_serves_receipt_and_matching_routes(server):
    _, base = server
    for path in ('/', '/matching_compare.js', '/matching_compare.css', '/voting.js', '/api/matching/algorithms', '/api/health', '/api/catalog?q=', '/api/1c/stock-sync/summary', '/analog', '/health'):
        with urllib.request.urlopen(base + path, timeout=10) as response:
            assert response.status == 200
            assert response.read()
    payload = dict(checked=[dict(id='t1', name='Молоко')],
                   stock=[dict(id='s1', name='Молоко', quantity=20)], min_stock=0)
    request = urllib.request.Request(base + '/api/matching/test',
        data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=10) as response:
        result = json.load(response)
    assert result['best_candidate']['stock_id'] == 's1'
    assert result['voting']['positions'][0]['recommended_stock_id'] == 's1'
    assert result['voting_elapsed_ms'] >= 0
    url = result['results_by_algorithm'][0]['full_results_url']
    with urllib.request.urlopen(base + url, timeout=10) as response:
        assert response.read()


@pytest.fixture
def server(tmp_path):
    def process(job):
        job.phase("geometry", 20, "geometry")
        job.image("original", np.full((100, 100, 3), 255, np.uint8), "original", role="original")
        result = dict(rows=[], product_count=0)
        job.json_file("receipt.json", result)
        return result
    instance = create_server("127.0.0.1", 0, tmp_path, process, max_upload=1024)
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    yield instance, f"http://127.0.0.1:{instance.server_port}"
    instance.shutdown()
    instance.server_close()
    instance.store.pool.shutdown(wait=True)
    thread.join(timeout=3)


def test_http_upload_sse_and_download(server):
    instance, base = server
    request = urllib.request.Request(base+"/api/jobs?filename=receipt.png", data=b"image", method="POST")
    with urllib.request.urlopen(request) as response:
        assert response.status == 202
        job_id = json.load(response)["id"]
    with urllib.request.urlopen(base+f"/api/jobs/{job_id}/events") as response:
        content = response.read().decode()
    payloads = [json.loads(line[6:]) for line in content.splitlines() if line.startswith("data: ")]
    events = [payload for payload in payloads if "type" in payload]
    assert any(event["type"] == "artifact" for event in events)
    stage_events = [event for event in events if event["type"] == "stage"]
    assert any(event["stage"] == "queued" and event["status"] == "running" for event in stage_events)
    assert any(event["stage"] == "geometry" and event["status"] == "done"
               and event["elapsed_ms"] >= 0 for event in stage_events)
    assert any(event["stage"] == "completed" and event["status"] == "done" for event in stage_events)
    assert events[-1]["type"] == "complete"
    assert "event: heartbeat\ndata: {}" in content
    assert events[-1]["progress"] == 100
    assert [event["id"] for event in events] == sorted({event["id"] for event in events})
    with urllib.request.urlopen(base+f"/api/jobs/{job_id}") as response:
        snapshot = json.load(response)
        assert snapshot["status"] == "done"
        assert snapshot["last_event_id"] == len(snapshot["events"])
        assert any(event["type"] == "stage" for event in snapshot["events"])
    with urllib.request.urlopen(base+f"/api/jobs/{job_id}/files/receipt.json") as response:
        assert json.load(response)["product_count"] == 0
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(base+f"/api/jobs/{job_id}/files/state.json")
    assert error.value.code == 404


def test_upload_size_limit_and_unknown_job(server):
    _, base = server
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(urllib.request.Request(base+"/api/jobs", data=b"x"*1025, method="POST"))
    assert error.value.code == 413
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(base+"/api/jobs/"+"0"*32)
    assert error.value.code == 404


def test_uploaded_filename_cannot_escape_job_directory(server):
    instance, base = server
    job = instance.store.submit(b"data", "../../outside.png")
    assert job.filename == "outside.png"
    assert job.directory.parent == instance.store.root


def test_photo_decoder_handles_exif_and_rejects_invalid_images():
    photo = Image.new("RGB", (180, 100), (255, 255, 255))
    exif = photo.getexif()
    exif[274] = 6
    stream = io.BytesIO()
    photo.save(stream, format="JPEG", exif=exif)
    image, rotated = decode_photo(stream.getvalue())
    assert rotated and image.shape[:2] == (180, 100)
    with pytest.raises(ValueError):
        decode_photo(b"not an image")


def test_layout_cells_include_separate_tax_row():
    layout = dict(name_left=30, name_right=160, quantity_right=200, right=260, row_boundaries=[16, 100, 180])
    rows = [dict(row=1, y_range=[16, 100], kind="product"), dict(row=2, y_range=[100, 180], kind="tax")]
    cells = grid_cells(layout, rows)
    assert len(cells) == 8
    assert len([cell for cell in cells if cell["kind"] == "tax"]) == 4
    assert {cell["role"] for cell in cells} == {"number", "name", "quantity", "price"}


def test_edit_api_validates_revision_and_persists_exports(server):
    from App.ocr.receipt_editing import update_receipt
    instance, base = server
    job = instance.store.submit(b"image", "receipt.png")
    with urllib.request.urlopen(base+f"/api/jobs/{job.id}/events") as response:
        response.read()
    job.result = update_receipt(dict(rows=[dict(row=1,kind="service",name_raw="Доставка",quantity_raw="10",price_raw="2")], total_raw="20"))
    url = base+f"/api/jobs/{job.id}/edits"
    def post(payload):
        return urllib.request.urlopen(urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type":"application/json"}, method="POST"))
    with post(dict(revision=0,row=1,quantity="5")) as response:
        result = json.load(response)["result"]
    assert result["calculation"]["difference"] == "-10.00"
    assert json.loads((job.directory/"receipt.json").read_text(encoding="utf-8"))["revision"] == 1
    with pytest.raises(urllib.error.HTTPError) as error:
        post(dict(revision=0,row=1,quantity="10"))
    assert error.value.code == 409
    with pytest.raises(urllib.error.HTTPError) as error:
        post(dict(revision=1,row=1,quantity="O"))
    assert error.value.code == 400
    with post(dict(revision=1,row=1,quantity="10",finalize=True)) as response:
        assert json.load(response)["result"]["finalized"]
    assert (job.directory/"receipt.csv").exists()


def test_live_events_preserve_ocr_snapshot(tmp_path):
    from App.web_server import Job
    job = Job("0"*32, "test.png", tmp_path)
    rows = [dict(row=1,name="Исходный OCR",stage="ocr")]
    job.publish("draft_rows", rows=rows)
    rows[0].update(name="Товар справочника",stage="matched")
    job.publish("row", row=rows[0])
    assert job.events[0]["rows"][0]["name"] == "Исходный OCR"
    assert job.events[1]["row"]["name"] == "Товар справочника"


def test_progress_ui_unlocks_after_accepted_202_and_has_real_history():
    from App.web_server import STATIC
    script = (STATIC/"app.js").read_text(encoding="utf-8")
    page = (STATIC/"index.html").read_text(encoding="utf-8")
    assert "ready=true;transition('ready')" in script
    assert "event.type==='stage'" in script
    assert "updateProgressPreview(event.artifact)" in script
    assert 'id="progress-history"' in page
    assert 'id="progress-preview"' in page


def test_receipt_uses_targeted_number_before_visual_sequence_fallback():
    script = (Path(__file__).resolve().parents[2] / 'App' / 'web' / 'app.js').read_text(encoding='utf-8')
    assert 'row.item_number_recovered' in script
    assert 'Номер повторно прочитан только из ячейки №.' in script


def test_catalog_selection_api_trains_persistent_knowledge(server, tmp_path):
    from App.ocr.manual_mappings import ManualMappings
    from App.ocr.ocr_catalog_eval import CatalogMatcher
    from App.ocr.receipt_editing import update_receipt
    instance, base = server
    knowledge = ManualMappings(tmp_path/"manual.json")
    pipeline = instance.store.pipeline
    pipeline.knowledge = knowledge
    target = dict(name="Выбранный товар",normalized="выбранный товар",sources=[dict(article="42")])
    pipeline.catalog_entry = lambda key:target if key == target["normalized"] else None
    job = instance.store.submit(b"image", "receipt.png")
    with urllib.request.urlopen(base+f"/api/jobs/{job.id}/events") as response:
        response.read()
    raw = "Непонятный OCR жжж 987654"
    job.result = update_receipt(dict(rows=[dict(row=1,kind="product",name_raw=raw,quantity_raw="1",price_raw="2")],total_raw="2"))
    payload = dict(revision=0,row=1,catalog_key=target["normalized"],article="42",confirm_name=True,learn_mapping=True)
    request = urllib.request.Request(base+f"/api/jobs/{job.id}/edits",data=json.dumps(payload).encode(),method="POST",headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(request) as response:
        result = json.load(response)
    assert result["mapping_saved"]
    assert knowledge.lookup(raw)["job_id"] == job.id
    assert result["result"]["rows"][0]["name_raw"] == raw
    restarted_matcher = CatalogMatcher([target],knowledge=ManualMappings(knowledge.path))
    reused = restarted_matcher.match(raw)
    assert reused["source"] == "manual_mapping" and reused["best"]["name"] == target["name"]
