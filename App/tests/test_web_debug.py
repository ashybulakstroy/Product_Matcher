from pathlib import Path

from App.services.matching import _sql_like_score
from App.web_debug import ALGORITHM_REGISTRY, FULL_RESULTS_PAGE, HTML_PAGE, STATIC_ROOT, match_debug_payload


def test_match_debug_payload_returns_top_candidates():
    checked = [{"id": "t1", "name": "Молоко 1.5% Ультрапастеризованное 1 л"}]
    stock = [
        {"id": "s1", "name": "Молоко ультрапастеризованное 1.5% 1 л"},
        {"id": "s2", "name": "Сметана 20% 500 г"},
    ]

    result = match_debug_payload(checked, stock, top_n=2)

    assert result["items"][0]["position_no"] == 1
    assert len(result["items"][0]["candidates"]) == 2
    assert result["items"][0]["candidates"][0]["stock_id"] == "s1"


def test_multi_algorithm_payload_contains_algorithm_metadata():
    checked = [{"id": "t1", "name": "Молоко 1.5% Ультрапастеризованное 1 л"}]
    stock = [
        {"id": "s1", "name": "Молоко ультрапастеризованное 1.5% 1 л"},
        {"id": "s2", "name": "Сметана 20% 500 г"},
    ]

    result = match_debug_payload(checked, stock, top_n=2, algorithm_id="copilot_1")

    assert result["algorithm_id"] == "copilot_1"
    assert result["best_candidate"]["stock_id"] == "s1"
    assert result["results_by_algorithm"][0]["algorithm_id"] == "copilot_1"
    assert result["results_by_algorithm"][0]["full_results_url"].startswith("/analog/list?algorithm_id=copilot_1")


def test_parent_web_root_is_present():
    index_file = STATIC_ROOT / "index.html"

    assert index_file.exists(), "родительский web root должен быть скопирован в проект"
    assert "Ашыбулак строй" in index_file.read_text(encoding="utf-8")


def test_full_results_page_separates_sufficient_and_insufficient_lists():
    assert "Достаточный остаток" in FULL_RESULTS_PAGE
    assert "Недостаточный остаток" in FULL_RESULTS_PAGE
    assert "color:#b91c1c" in FULL_RESULTS_PAGE


def test_main_page_low_stock_quantity_is_red():
    assert "color:#b91c1c" in HTML_PAGE


def test_sql_str_2_algorithm_is_available():
    assert "sql_str_2" in ALGORITHM_REGISTRY

    checked = [{"id": "t1", "name": "Молоко 1.5%"}]
    stock = [
        {"id": "s1", "name": "Молоко ультрапастеризованное 1.5% 1 л"},
        {"id": "s2", "name": "Сметана 20% 500 г"},
    ]

    result = match_debug_payload(checked, stock, top_n=2, algorithm_id="sql_str_2")

    assert result["algorithm_id"] == "sql_str_2"
    assert result["results_by_algorithm"][0]["algorithm_id"] == "sql_str_2"
    assert result["best_candidate"]["stock_name"] == "Молоко ультрапастеризованное 1.5% 1 л"


def test_search_runs_all_algorithms_in_parallel_by_default():
    checked = [{"id": "t1", "name": "Молоко 1.5%"}]
    stock = [
        {"id": "s1", "name": "Молоко ультрапастеризованное 1.5% 1 л"},
        {"id": "s2", "name": "Сметана 20% 500 г"},
    ]

    result = match_debug_payload(checked, stock, top_n=2, algorithm_id=None, algorithms=None)

    assert len(result["results_by_algorithm"]) == len(ALGORITHM_REGISTRY)
    assert {item["algorithm_id"] for item in result["results_by_algorithm"]} == set(ALGORITHM_REGISTRY)


def test_sql_str_2_scores_by_token_coverage():
    left = "Тачка садовая одноколесная усиленная эксперт"
    right = "Тачка садово-строительная одноколесная усиленная желтая ЭКСПЕРТ"
    partial = "Тачка эксперт"
    unrelated = "Электроды МР-3 АРС Арсенал д.3 мм пачка 2,5 кг"

    full_score = _sql_like_score(left, right)
    partial_score = _sql_like_score(left, partial)
    unrelated_score = _sql_like_score("тачка", unrelated)

    assert full_score > partial_score
    assert full_score > 60
    assert partial_score > 0
    assert unrelated_score == 0.0


def test_match_debug_payload_keeps_search_query_for_full_results_page():
    checked = [{"id": "t1", "name": "Тачка садовая одноколесная усиленная эксперт"}]
    stock = [{"id": "s1", "name": "Тачка садово-строительная одноколесная усиленная желтая ЭКСПЕРТ", "quantity": 20}]

    result = match_debug_payload(checked, stock, top_n=2, algorithm_id="sql_str_2")

    assert result["query"] == "Тачка садовая одноколесная усиленная эксперт"
    assert result["results_by_algorithm"][0]["query"] == "Тачка садовая одноколесная усиленная эксперт"
