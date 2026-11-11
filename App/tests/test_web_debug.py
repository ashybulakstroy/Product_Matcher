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


def test_price_search_filters_literal_name_price_and_stock_without_top_limit():
    from App.web_debug import price_search
    stock = [dict(name='Кабель медный', accounting_price=price, quantity=5) for price in range(20, 0, -1)]
    stock += [dict(name='Провод', accounting_price=12, quantity=5),
              dict(name='Кабель', accounting_price=None, quantity=5),
              dict(name='Кабель', accounting_price=12, quantity=1)]
    result = price_search('КАБЕЛЬ', 5, stock, 3)
    assert [item['accounting_price'] for item in result['items']] == list(range(4, 21))


def test_price_search_without_name_returns_all_names_from_inclusive_floor():
    from App.web_debug import price_search
    stock=[dict(id='a',name='Кабель',accounting_price=120,quantity=5),
           dict(id='b',name='Болт',accounting_price=100,quantity=5),
           dict(id='c',name='Гайка',accounting_price=99,quantity=5),
           dict(id='d',name='Провод',accounting_price=None,quantity=5),
           dict(id='e',name='Шайба',accounting_price=110,quantity=1)]
    assert [item['id'] for item in price_search('',100,stock,3)['items']]==['c','b','a']
    assert [item['id'] for item in price_search('кабель',100,stock,3)['items']]==['a']


def test_price_catalog_pages_are_sorted_bounded_and_json_serializable():
    from decimal import Decimal
    import json
    from App.web_debug import price_catalog_page
    stock=[dict(id=str(i), name='item', accounting_price=Decimal(i), quantity=Decimal('3.25'),
                stock_quantity=Decimal('3.25')) for i in range(250,0,-1)]
    pages=[price_catalog_page('', 10, stock, 3.25, page) for page in (1,2,3)]
    assert [len(page['items']) for page in pages]==[100,100,43]
    assert all(page['total']==243 and page['pages']==3 for page in pages)
    ids=[int(item['id']) for page in pages for item in page['items']]
    assert ids==list(range(8,251))
    json.dumps(pages)
    assert price_catalog_page('', 10, stock, 4)['total']==0


def test_price_catalog_sort_orders_all_pages_before_pagination():
    from App.web_debug import price_catalog_page
    stock = [dict(id=str(i),name='Товар',accounting_price=i,quantity=5) for i in range(205)]
    first = price_catalog_page('',0,stock,3,1,'desc')
    second = price_catalog_page('',0,stock,3,2,'desc')
    assert [int(item['id']) for item in first['items']] == list(range(204,104,-1))
    assert [int(item['id']) for item in second['items']] == list(range(104,4,-1))


def test_price_search_uses_exact_75_percent_floor_before_display_rounding():
    from App.web_debug import price_search
    stock = [dict(id=str(price),name='товар',accounting_price=price,quantity=4)
             for price in ('75.74','75.75','75.76','101','1000')]
    result = price_search('', '101', stock, 3)
    assert result['minimum_price']=='75.75'
    assert [item['id'] for item in result['items']] == ['75.75','75.76','101','1000']
