import cv2,numpy as np,pytest
from geometry_lab.physical_grid_graph_local import detect_physical_graph,ridge_response,trace_seed
from geometry_lab.synthetic import receipt_grid
from geometry_lab.linked_grid import export_lines,validate_linked_annotation


def test_blank_has_no_horizontal_evidence():
    image=np.full((600,500,3),240,np.uint8)
    for x in [50,90,300,360,450]:cv2.line(image,(x,100),(x,500),(50,50,50),1)
    data,_=detect_physical_graph(image)
    assert data is None


def test_faint_broken_separator_requires_distributed_ink():
    image,xs,ys=receipt_grid()
    cv2.line(image,(xs[0],ys[3]),(xs[-1],ys[3]),(244,244,244),5)
    cv2.line(image,(xs[0],ys[3]),(xs[-1],ys[3]),(205,205,205),1)
    cv2.line(image,(300,ys[3]),(320,ys[3]),(244,244,244),3)
    data,report=detect_physical_graph(image)
    assert data is not None,report
    assert any(abs(r.position-ys[3])<4 for r in data['horizontal'])
    assert all(e['ink_coverage']>=.64 and sum(e['intersection_evidence'])>=3 for e in report['edges'])


def test_local_tracker_does_not_jump_to_neighbouring_strong_rule():
    response=np.zeros((120,200),np.float32)
    response[50,:]=20;response[62,:]=60
    xs=np.arange(10,191,2,dtype=float)
    ys,_=trace_seed(response,(20,50,180,50),xs)
    assert np.median(abs(ys-50))<2


def test_shared_node_changes_both_lines():
    nodes=[[[10+20*c,10+30*r] for c in range(5)] for r in range(4)]
    nodes[1][2]=[53,44]
    lines=export_lines(nodes)
    assert lines['horizontal_polylines'][1][2]==[53.,44.]
    assert lines['vertical_polylines'][2][1]==[53.,44.]
    assert validate_linked_annotation(lines)==lines
    lines['vertical_polylines'][2][1][0]+=2
    with pytest.raises(ValueError):validate_linked_annotation(lines)


def test_crossed_grid_export_is_rejected():
    nodes=[[[10+20*c,10+30*r] for c in range(5)] for r in range(4)]
    nodes[2][1][1]=5
    with pytest.raises(ValueError):export_lines(nodes)
