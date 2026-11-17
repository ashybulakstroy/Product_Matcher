import cv2,numpy as np
from geometry_lab.physical_grid_graph_v3 import detect_physical_graph
from geometry_lab.synthetic import receipt_grid
from geometry_lab.structure_diagnostics import reference_data
from geometry_lab.linked_grid import validate_linked_annotation


def test_short_four_column_table_not_rejected_by_page_height():
    im=np.full((1200,600,3),245,np.uint8);xs=[60,100,310,370,460];ys=[280,320,360,400]
    for x in xs:cv2.line(im,(x,ys[0]),(x,ys[-1]),(90,90,90),1)
    for y in ys:cv2.line(im,(xs[0],y),(xs[-1],y),(90,90,90),1)
    data,report=detect_physical_graph(im)
    assert data is not None,report
    assert len(data['vertical'])==5
    assert all(min(abs(r.position-y) for r in data['horizontal'])<4 for y in ys)


def test_text_and_soft_shadow_do_not_create_printed_grid():
    image=np.full((700,600,3),240,np.uint8)
    for y in [200,300,400]:
        for x in [60,130,350,470]:cv2.putText(image,'TEXT',(x,y),cv2.FONT_HERSHEY_SIMPLEX,.4,(60,60,60),1)
    assert detect_physical_graph(image)[0] is None


def test_extra_text_between_rules_is_not_an_extra_horizontal():
    im,xs,ys=receipt_grid()
    for x in [130,220,550,650]:cv2.putText(im,'123',(x,260),cv2.FONT_HERSHEY_SIMPLEX,.45,(60,60,60),1)
    data,report=detect_physical_graph(im)
    assert data is not None,report
    assert all(min(abs(r.position-y) for y in ys)<4 for r in data['horizontal'])


def test_graph_export_with_bend_is_compatible_with_evaluation():
    points={'a':{'x':0,'y':10},'b':{'x':50,'y':12},'c':{'x':100,'y':10},'d':{'x':0,'y':50}}
    g=dict(schema='geometry_grid_graph_v1',graph_nodes=points,horizontal_node_ids=[['a','b','c']],vertical_node_ids=[['a','d']],
        horizontal_polylines=[[[0,10],[50,12],[100,10]]],vertical_polylines=[[[0,10],[0,50]]],nodes=[])
    validate_linked_annotation(g)
    d=reference_data(g,(100,100))
    assert d['horizontal'][0].eval(0)==12
