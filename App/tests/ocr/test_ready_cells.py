import copy

from App.ocr.visual_sync import ready_cells


LAYOUT = dict(body_left=10, name_left=40, name_right=240, quantity_right=300, right=380)


def box(left, top, right, bottom):
    return [[left,top],[right,top],[right,bottom],[left,bottom]]


def row(number=1):
    return dict(row=number,kind='product',y_range=[30,100],item_number_raw='1',
                name_raw='Товар в две строки',quantity_raw='2',price_raw='850')


def test_cell_waits_for_all_detected_blocks_in_its_own_area():
    original=[row()];before=copy.deepcopy(original)
    pending=[box(50,75,180,90),box(310,50,370,70)]
    ready=ready_cells(original,LAYOUT,pending)
    assert {cell['field'] for cell in ready} == {'number','quantity'}
    assert original == before
    ready=ready_cells(original,LAYOUT)
    assert {cell['field'] for cell in ready} == {'number','name','quantity','price'}
    assert next(cell['value'] for cell in ready if cell['field']=='name') == original[0]['name_raw']


def test_pending_blocks_in_another_row_do_not_delay_current_cells():
    ready=ready_cells([row()],LAYOUT,[box(60,130,180,150)])
    assert len(ready)==4


def test_no_invented_values_for_empty_cells_or_total_row():
    empty=row();empty['quantity_raw']='';empty['item_number_raw']=''
    total=dict(row(),kind='total')
    ready=ready_cells([empty,total],LAYOUT)
    assert {cell['field'] for cell in ready} == {'name','price'}


def test_recovered_printed_number_is_ready_without_an_inferred_ordinal():
    source=row();source['item_number_raw']='';source['item_number_recovered']='12'
    ready=ready_cells([source],LAYOUT)
    assert next(cell['value'] for cell in ready if cell['field']=='number') == '12'
