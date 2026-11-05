from App.services.matching import match_products, normalize_name


def test_normalize_name_removes_extra_symbols():
    raw = "Молоко 1.5% ультрапастеризованное, 1 л"
    result = normalize_name(raw)
    assert "молоко" in result
    assert "1" in result
    assert "л" in result


def test_match_products_returns_top_candidates_sorted_by_similarity():
    checked = [{"id": "t1", "name": "Молоко 1.5% Ультрапастеризованное 1 л"}]
    stock = [
        {"id": "s1", "name": "Молоко ультрапастеризованное 1.5% 1 л"},
        {"id": "s2", "name": "Сметана 20% 500 г"},
        {"id": "s3", "name": "Молоко 2.5% пастеризованное 1 л"},
    ]

    result = match_products(checked, stock, top_n=2)

    assert len(result) == 1
    assert result[0]["position_no"] == 1
    assert result[0]["checked_id"] == "t1"
    assert len(result[0]["candidates"]) == 2
    assert result[0]["candidates"][0]["stock_id"] == "s1"
    assert result[0]["candidates"][0]["similarity_pct"] >= result[0]["candidates"][1]["similarity_pct"]
