from App.invoicing import category_index
from App.web_debug import price_catalog_page


def test_category_search_scoped_hierarchical_and_preserves_unknowns(monkeypatch, tmp_path):
    source = tmp_path / 'classified.xlsx'
    source.touch()
    monkeypatch.setattr(category_index, '_load', lambda *args: {
        '01': ('кабель', ('Электрика', 'Провода', 'Кабель')),
        '02': ('провод', ('Электрика', 'Провода', 'Провод')),
        '03': ('болт', ('Крепёж', 'Метизы', 'Болт')),
        '04': None,
        '05': ('устаревшее название', ('Электрика', 'Провода', 'Кабель')),
    })
    stock = [dict(id='id'+code,code=code,name=name,quantity=10,accounting_price=price) for code,name,price in [
        ('01','Кабель',200),('02','Провод',100),('03','Болт',300),('04','Неясно',150),('05','Новый товар',250)]]
    select = lambda category: category_index.filter_stock(stock,'buh_kz_01',category,source)
    assert [x['code'] for x in select(['Электрика'])] == ['01','02']
    assert [x['code'] for x in select(['Электрика','Провода','Кабель'])] == ['01']
    assert [x['code'] for x in select([category_index.UNCLASSIFIED])] == ['04','05']
    assert [x['code'] for x in category_index.filter_stock(stock,'other', [category_index.UNCLASSIFIED],source)] == ['01','02','03','04','05']
    result = price_catalog_page('',100,select(['Электрика']),3)
    assert [x['name'] for x in result['items']] == ['Провод','Кабель']


def test_reject_invalid_category(monkeypatch, tmp_path):
    import pytest
    source = tmp_path / 'classified.xlsx'
    source.touch()
    monkeypatch.setattr(category_index, '_load', lambda *args: {})
    with pytest.raises(ValueError):
        category_index.filter_stock([], 'buh_kz_01', ['Не существует'], source)


def test_keyword_matches_all_category_levels_across_branches(monkeypatch, tmp_path):
    source = tmp_path / 'classified.xlsx'
    source.touch()
    monkeypatch.setattr(category_index, '_load', lambda *args: {
        '1': ('средство', ('Быт', 'Автохимия', 'Моющее средство')),
        '2': ('очиститель', ('Автохимия', 'Уход', 'Очиститель')),
        '3': ('клей', ('Клей', 'Материалы', 'Стройхимия')),
        '4': ('болт', ('Крепёж', 'Метизы', 'Болты')),
    })
    stock = [dict(id=code, code=code, name=name) for code, name in [('1','Средство'),('2','Очиститель'),('3','Клей'),('4','Болт')]]
    assert [item['id'] for item in category_index.filter_stock(stock,'buh_kz_01','ХиМиЯ',source)] == ['1','2','3']
    assert category_index.filter_stock(stock,'buh_kz_01','несуществующая',source) == []
    assert [item['id'] for item in category_index.filter_stock(stock,'buh_kz_01',['Быт','Автохимия'],source)] == ['1']
