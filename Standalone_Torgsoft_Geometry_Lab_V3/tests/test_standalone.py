from pathlib import Path
import json
import numpy as np
import cv2
import pytest
from geometry_lab.cases import discover,Case,load_baseline
from geometry_lab.synthetic import receipt_grid,deform
from geometry_lab.detection import detect_grid
from geometry_lab.algorithms import detect_and_run
from geometry_lab.metrics import safety,projected_lines,evaluate_ground_truth


def test_imports_do_not_require_project_or_prior_trial():
    import geometry_lab.algorithms,geometry_lab.cases,geometry_lab.metrics
    assert not any('trial_geometry' in x for x in geometry_lab.algorithms.__dict__)


def test_unique_sha_with_bytes_and_read_only(tmp_path):
    src=tmp_path/'web_outputs'
    img,*_=receipt_grid()
    for d in ['a','b']:
        p=src/d;p.mkdir(parents=True)
        assert cv2.imwrite(str(p/'original.png'),img)
    assert len(discover(src))==1
    assert (src/'a'/'original.png').is_file()


def test_source_not_modified_and_baseline_optional(tmp_path):
    src=tmp_path/'web_outputs'/'job1';src.mkdir(parents=True)
    img,*_=receipt_grid()
    cv2.imwrite(str(src/'original.png'),img)
    case=discover(src.parent)[0]
    assert load_baseline(case)==(None,'main_ocr_artifacts_missing')
    before=(src/'original.png').read_bytes()
    for name in ['perspective','grid_mesh']:
        _=detect_and_run(img,name)
    assert (src/'original.png').read_bytes()==before


def test_synthetic_detects_five_columns():
    image,*_=receipt_grid()
    data,rep=detect_grid(image)
    assert data is not None,rep
    assert len(data['vertical'])==5,rep
    assert len(data['horizontal'])>=3,rep


@pytest.mark.parametrize('method',['perspective','grid_mesh','tps_scipy'])
def test_synthetic_rectification(method):
    image,*_=receipt_grid()
    result,rep=detect_and_run(image,method)
    assert result is not None,rep
    met=safety(result,image.shape)
    assert met['source_valid_fraction']>.9,met
    assert met['folded_fraction']<.01,met
    v,h=projected_lines(result)
    assert len(v)==5
    assert len(h)>=3


def test_gt_mismatch_not_reported_as_zero_error():
    img,*_=receipt_grid()
    result,rep=detect_and_run(img,'grid_mesh')
    assert result is not None,rep
    val=evaluate_ground_truth(result,{'vertical_polylines':[], 'horizontal_polylines':[]},img.shape)
    assert val['status']=='GT_MISMATCH'
    assert 'mean_px' not in val


def test_synthetic_wavy_grid_is_not_split_into_duplicate_rows():
    image,*_=receipt_grid()
    warped=deform(image,'wave')
    data,report=detect_grid(warped)
    assert data is not None,report
    assert len(data['vertical'])==5
    assert len(data['horizontal'])==8
    result,diagnostic=detect_and_run(warped,'grid_mesh')
    assert result is not None,diagnostic
    assert safety(result,warped.shape)['folded_fraction']<.01
