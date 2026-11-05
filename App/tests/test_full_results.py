import json
import re
import shutil
import subprocess
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

from App.web_debug import (
    FULL_RESULTS_PAGE,
    HTML_PAGE,
    MatchingDebugHandler,
    _build_algorithm_result,
    _load_full_result,
    clear_full_result_cache,
)


def _stock_items(count=10):
    return [
        {"id": f"s{index:02d}", "name": f"Товар {index:02d}", "quantity": 100}
        for index in range(count)
    ]


def test_full_results_url_keeps_all_candidates():
    clear_full_result_cache()
    try:
        result = _build_algorithm_result(
            [{"id": "q", "name": "Товар"}],
            _stock_items(25),
            top_n=3,
            min_stock=0,
            algorithm_id="copilot_1",
        )

        url = result["full_results_url"]
        assert "result_id=" in url
        assert "payload=" not in url
        assert len(url) < 200
        result_id = parse_qs(urlparse(url).query)["result_id"][0]
        full_result = _load_full_result(result_id)

        assert full_result is not None
        assert len(result["items"][0]["candidates"]) == 3
        assert len(full_result["items"][0]["candidates"]) == 25
    finally:
        clear_full_result_cache()


def test_full_results_keeps_all_checked_items():
    clear_full_result_cache()
    try:
        result = _build_algorithm_result(
            [
                {"id": "q1", "name": "Товар 01"},
                {"id": "q2", "name": "Товар 02"},
            ],
            _stock_items(10),
            top_n=2,
            min_stock=0,
            algorithm_id="copilot_1",
        )

        result_id = parse_qs(urlparse(result["full_results_url"]).query)["result_id"][0]
        full_result = _load_full_result(result_id)

        assert full_result is not None
        assert len(full_result["items"]) == 2
        assert [item["checked_name"] for item in full_result["items"]] == ["Товар 01", "Товар 02"]
        assert all(len(item["candidates"]) == 10 for item in full_result["items"])
    finally:
        clear_full_result_cache()


def test_full_results_http_endpoint_returns_all_items_and_candidates():
    clear_full_result_cache()
    server = ThreadingHTTPServer(("127.0.0.1", 0), MatchingDebugHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    try:
        payload = {
            "checked": [
                {"id": "q1", "name": "Товар 01"},
                {"id": "q2", "name": "Товар 02"},
            ],
            "stock": _stock_items(10),
            "top_n": 2,
            "min_stock": 0,
            "algorithms": ["copilot_1"],
        }
        request = urllib.request.Request(
            base + "/analog",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))

        algorithm = data["results_by_algorithm"][0]
        assert len(data["items"]) == 2
        assert len(data["items"][0]["candidates"]) == 2

        result_id = parse_qs(urlparse(algorithm["full_results_url"]).query)["result_id"][0]

        with urllib.request.urlopen(base + algorithm["full_results_url"], timeout=30) as response:
            assert response.status == 200
            assert "Полный список кандидатов" in response.read().decode("utf-8")

        with urllib.request.urlopen(
            base + f"/analog/list/data?result_id={result_id}", timeout=30
        ) as response:
            full_result = json.loads(response.read().decode("utf-8"))

        assert len(full_result["items"]) == 2
        assert all(len(item["candidates"]) == 10 for item in full_result["items"])

        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(base + "/analog/list/data?result_id=missing", timeout=30)
        assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        clear_full_result_cache()


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
@pytest.mark.parametrize("page", [HTML_PAGE, FULL_RESULTS_PAGE], ids=["main_page", "full_results_page"])
def test_page_javascript_has_valid_syntax(page, tmp_path):
    match = re.search(r"<script>(.*?)</script>", page, re.S)
    assert match is not None

    script_path = tmp_path / "page.js"
    script_path.write_text(match.group(1), encoding="utf-8")
    completed = subprocess.run(
        ["node", "--check", str(script_path)],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
