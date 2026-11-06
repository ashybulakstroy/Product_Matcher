import pytest

from App.ocr.ocr_catalog_eval import aligned_lines, parse_rows


LAYOUT = dict(body_left=10, name_left=60, name_right=320, quantity_right=410,
              right=520, body_top=10, body_bottom=160)


def line(text, x, y, width=40):
    box = [[x-width/2, y-10], [x+width/2, y-10], [x+width/2, y+10], [x-width/2, y+10]]
    return dict(text=text, confidence=.9, box=box, box_original=box)


@pytest.mark.parametrize('price_label', ['Багасы', 'Бағасы', 'Цена'])
@pytest.mark.parametrize('boundaries', [[10, 60, 120, 160], [10, 120, 160]])
def test_header_number_misread_as_n_does_not_become_product(price_label, boundaries):
    payload = [line('N', 85, 40), line('Тауардың атауы', 220, 40, 150),
               line('Саны', 370, 40), line(price_label, 460, 40),
               line('Краска N', 180, 90), line('2', 370, 90), line('100', 460, 90),
               line('НДС 16 %', 180, 140), line('32', 370, 140), line('1', 460, 140)]
    rows = parse_rows(aligned_lines(dict(lines=payload), {}), LAYOUT, boundaries)
    assert [row['kind'] for row in rows] == ['product', 'tax']
    assert rows[0]['name_raw'] == 'Краска N'
    assert (rows[0]['quantity_raw'], rows[0]['price_raw']) == ('2', '100')
    assert all(item['text'] not in {'Саны', price_label, 'N'} for row in rows for item in row['raw_lines'])


def test_product_named_n_with_numeric_cells_is_kept():
    payload = [line('N', 85, 40), line('2', 370, 40), line('100', 460, 40)]
    rows = parse_rows(aligned_lines(dict(lines=payload), {}), LAYOUT, [10, 60])
    assert len(rows) == 1 and rows[0]['name_raw'] == 'N'


def test_quantity_and_price_labels_identify_header_even_when_name_is_unreadable():
    payload = [line('N', 85, 40), line('Саны', 370, 40), line('Багасы', 460, 40),
               line('Товар', 180, 90), line('2', 370, 90), line('100', 460, 90)]
    rows = parse_rows(aligned_lines(dict(lines=payload), {}), LAYOUT, [10, 60, 120])
    assert len(rows) == 1 and rows[0]['name_raw'] == 'Товар'


@pytest.mark.parametrize('marker',['N8','No','№','1'])
def test_header_excludes_numeric_noise_and_all_content_above_it(marker):
    payload=[line('Кассир 100',180,15),line(marker,30,40),line('Тауардың атауы',180,40),
             line('Саны',370,40),line('Багасы',460,40),line('Доска',180,90),
             line('51',370,90),line('1880',460,90)]
    rows=parse_rows(aligned_lines(dict(lines=payload),{}),LAYOUT,[0,25,60,120])
    assert len(rows)==1 and rows[0]['name_raw']=='Доска'
    assert (rows[0]['quantity_raw'],rows[0]['price_raw'])==('51','1880')
