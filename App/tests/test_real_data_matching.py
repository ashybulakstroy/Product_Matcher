import csv
from pathlib import Path

from App.services.matching import match_products


def _read_first_real_match_row():
    csv_path = Path(__file__).resolve().parents[2] / "data" / "in_data" / "Единый справочник соответствий Торгсофт-1С 24722.csv"
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter=';')
        rows = list(reader)

    header = rows[0]
    first_data_row = rows[1]
    data = dict(zip(header, first_data_row))
    return data


def test_match_on_real_parent_data():
    row = _read_first_real_match_row()
    checked_name = row["Название Торгсофт"]
    stock_name = row["Рекомендуемая номенклатура 1С"]

    result = match_products(
        [{"id": "real-check", "name": checked_name}],
        [{"id": "real-stock", "name": stock_name}],
        top_n=3,
    )

    assert len(result) == 1
    assert result[0]["position_no"] == 1
    assert result[0]["candidates"][0]["stock_id"] == "real-stock"
    assert result[0]["candidates"][0]["similarity_pct"] >= 90
