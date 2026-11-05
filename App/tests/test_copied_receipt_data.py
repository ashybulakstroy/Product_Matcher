import json
from pathlib import Path


def test_copied_parent_test_data_is_present():
    root = Path(__file__).resolve().parents[2]
    golden_path = root / "data" / "test_data" / "golden" / "cases.json"
    assert golden_path.exists(), "Golden input/output fixture is missing"

    payload = json.loads(golden_path.read_text(encoding="utf-8"))
    receipts = payload.get("receipts") or []
    assert len(receipts) >= 1
    assert "items" in receipts[0]

    image_dir = root / "data" / "test_data" / "torgsoft_receipts"
    assert image_dir.exists(), "Receipt sample directory is missing"
    image_files = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.jpeg"))
    assert len(image_files) >= 1
