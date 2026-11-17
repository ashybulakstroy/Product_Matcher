import cv2
import numpy as np
from geometry_lab.physical_grid_graph import connected_fragments,detect_physical_graph
from geometry_lab.structure_diagnostics import reference_data,match_lines,serialize
from geometry_lab.synthetic import receipt_grid
from stage2 import confirmed_reference


def test_nearby_distinct_rules_are_not_merged_by_page_height():
    parts=[(200.,(10,200,300,200),290),(212.,(10,212,300,212),290),
           (200.,(320,200,600,200),280)]
    groups=connected_fragments(parts,840)
    assert sorted(map(len,groups))==[1,2]


def test_unsupported_large_gap_not_connected():
    assert len(connected_fragments([(100,(0,100,50,100),50),
        (100,(700,100,800,100),100)],840))==2


def test_text_and_partial_footer_are_not_full_grid_rows():
    im,xs,ys=receipt_grid()
    cv2.line(im,(210,250),(400,250),(60,60,60),2)
    cv2.line(im,(xs[2],820),(xs[-1],820),(60,60,60),2)
    data,report=detect_physical_graph(im)
    assert data is not None,report
    measured=[r.position for r in data['horizontal']]
    assert all(min(abs(y-v) for v in measured)<4 for y in ys)
    assert not any(abs(y-250)<5 or abs(y-820)<5 for y in measured)


def test_partial_gap_needs_observed_ink_and_crossings():
    im,xs,ys=receipt_grid()
    cv2.rectangle(im,(280,ys[3]-3),(310,ys[3]+3),(244,244,244),-1)
    data,report=detect_physical_graph(im)
    assert data is not None
    edge=min(report['edges'],key=lambda e:abs(e['position']-ys[3]))
    assert abs(edge['position']-ys[3])<4 and edge['inferred_gap_samples']>0
    assert sum(edge['intersection_evidence'])>=3


def test_equal_counts_do_not_hide_missing_and_false_line():
    gt=dict(horizontal_polylines=[[[0,10],[100,10]],[[0,30],[100,30]]],vertical_polylines=[])
    pred=dict(horizontal=[[[0,10],[100,10]],[[0,60],[100,60]]],vertical=[])
    m=match_lines(pred,gt,(100,100))['horizontal']
    assert m['missing']==[1] and m['false']==[1]


def test_reference_adapter_preserves_bend_without_polynomial_fit():
    gt=dict(horizontal_polylines=[[[0,10],[50,30],[100,10]]],vertical_polylines=[])
    d=reference_data(gt,(100,100))
    assert d['horizontal'][0].eval(0)==30
    assert d['horizontal'][0].eval(-.5)==10


def test_human_confirmation_requires_provenance_and_all_lines():
    c=dict(sha256='a',shape=[100,100]);gt=dict(human_verified=True,source_checked=True,
        reviewer='human',reviewed_at='2026-11-16',sha256='a',source_shape=[100,100],
        vertical_polylines=[[]]*5,horizontal_polylines=[[]]*4,
        approvals=dict(vertical=[True]*5,horizontal=[True]*4))
    assert confirmed_reference(gt,c)
    gt['approvals']['horizontal'][0]=False
    assert not confirmed_reference(gt,c)
    assert not confirmed_reference(dict(human_verified=True),c)
