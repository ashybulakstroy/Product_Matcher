import json
from pathlib import Path


def build_test_report() -> dict:
    root = Path(__file__).resolve().parents[1]
    golden_path = root / "data" / "test_data" / "golden" / "cases.json"
    receipt_dir = root / "data" / "test_data" / "torgsoft_receipts"

    payload = json.loads(golden_path.read_text(encoding="utf-8")) if golden_path.exists() else {"receipts": []}
    receipts = payload.get("receipts") or []

    rows = []
    for idx, receipt in enumerate(receipts[:5], start=1):
        item_names = [item.get("name", "") for item in receipt.get("items", [])]
        rows.append(
            {
                "position": idx,
                "input": {
                    "receipt_id": receipt.get("id"),
                    "expected_items": item_names,
                    "input_files": sorted([p.name for p in receipt_dir.iterdir() if p.is_file()])[:10],
                },
                "output": {
                    "status": "golden_fixture_loaded",
                    "expected_basis": receipt.get("expected_basis"),
                    "order_number": receipt.get("order_number")
                },
            }
        )

    return {
        "summary": {
            "golden_cases": len(receipts),
            "receipt_files": len(list(receipt_dir.glob("*"))) if receipt_dir.exists() else 0,
            "status": "ok",
            "message": "This report shows the input sample and the expected output fixture for the matching module.",
        },
        "rows": rows,
    }


def save_test_report(path: str | None = None) -> str:
    root = Path(__file__).resolve().parents[1]
    target = Path(path) if path else root / "data" / "test_data" / "report.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(build_test_report(), ensure_ascii=False, indent=2), encoding="utf-8")
    return str(target)
