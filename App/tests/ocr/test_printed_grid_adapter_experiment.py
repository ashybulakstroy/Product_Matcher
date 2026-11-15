import json
from pathlib import Path

import cv2
import numpy as np

from App.ocr import printed_grid_adapter as adapter

ROOT=Path(__file__).resolve().parents[3]
PHOTO=ROOT/'data/ocr_regression/printed_grid_v3/photos/receipt_20261115.png'


def test_shadow_does_not_change_receipt(monkeypatch, tmp_path):
    image=cv2.imread(str(PHOTO))
    current=(image,None,None,dict(method='existing_contour',warnings=[]))
    monkeypatch.setattr(adapter,'mode',lambda:'shadow')
    out, report=adapter.maybe_use_printed_grid(image,current,output_dir=tmp_path)
    assert out is current
    assert not report['selected']
    assert (tmp_path/'candidate_printed_grid_rectified.png').is_file()


def test_select_only_on_observed_geometry(monkeypatch):
    image=cv2.imread(str(PHOTO))
    current=(image,None,None,dict(method='original_crop_fallback',warnings=[]))
    monkeypatch.setattr(adapter,'mode',lambda:'select')
    out,report=adapter.maybe_use_printed_grid(image,current)
    assert report['selected'] and report['status']=='selected',report
    assert out[3]['method']=='printed_grid_rectified'
    assert out[0].shape[:2]==out[1].shape==out[2].shape
    assert np.array_equal(image,current[0])


def test_blank_does_not_change_geometry_even_select(monkeypatch):
    image=np.full((1024,768,3),255,np.uint8)
    baseline=(image,None,None,dict(method='baseline',warnings=[]))
    monkeypatch.setattr(adapter,'mode',lambda:'select')
    result,report=adapter.maybe_use_printed_grid(image,baseline)
    assert result is baseline and not report['selected']

def test_select_preserves_established_and_special_geometry(monkeypatch):
    monkeypatch.setattr(adapter,'mode',lambda:'select')
    monkeypatch.setattr(adapter,'rectify_by_printed_grid',lambda *_: (_ for _ in ()).throw(
        AssertionError('verified geometry must not be replaced')))
    image=np.zeros((1000,700,3),np.uint8)
    for method in ['grid_curvature_v2','wave_grid','contour_coons_patch','source_grid_crop','rotation_crop']:
        current=(image,None,None,dict(method=method))
        output,report=adapter.maybe_use_printed_grid(image,current)
        assert output is current
        assert report['reason']=='preserve_established_geometry'
