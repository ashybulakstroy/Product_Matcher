from pathlib import Path

import cv2
import numpy as np
import pytest

from App.ocr.ocr_catalog_eval import aligned_lines, choose_row_boundaries, parse_rows, table_layout
from App.ocr.receipt_rectification import rectify_table
from App.ocr.table_boundary import detect_table_boundary

FIXTURE = Path(__file__).parent/"fixtures/wavy_receipt_596753.png"
CURVED = Path(__file__).parent/"fixtures/curved_receipt_table.png"


def _rectify_curved(monkeypatch, mode):
    monkeypatch.setenv("OCR_GRID_CURVATURE_V2", mode)
    image = cv2.imdecode(np.frombuffer(CURVED.read_bytes(),np.uint8),cv2.IMREAD_COLOR)
    height,width = image.shape[:2]
    polygon = np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]],np.float32)
    return image,rectify_table(image,polygon)


def test_wave_receipt_selects_verified_grid_and_preserves_source_map(monkeypatch):
    monkeypatch.setenv("OCR_GRID_CURVATURE_V2","select")
    image = cv2.imdecode(np.frombuffer(FIXTURE.read_bytes(),np.uint8),cv2.IMREAD_COLOR)
    boundary = detect_table_boundary(image)
    output,mx,my,metadata = rectify_table(image,boundary["polygon"],trust_top=boundary["status"]=="detected")
    assert metadata["method"] == "wave_grid"
    assert metadata["wave_assessment"]["selected"]
    assert metadata["wave_assessment"]["straightness_check"]["horizontal"]["after_px"]<1
    assert np.array_equal(output,cv2.remap(image,mx,my,cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT,borderValue=(255,255,255)))
    layout = table_layout(output,metadata)
    assert layout["grid_confirmed"] and len(layout["row_boundaries"])==4
    assert layout["body_left"]<layout["name_left"]<layout["name_right"]<layout["quantity_right"]<layout["right"]
    assert layout["body_bottom"]<output.shape[0]-15  # Footer strip is retained.


def test_curved_receipt_selects_v2_when_established_wave_is_not_selected(monkeypatch):
    image,(output,mx,my,metadata) = _rectify_curved(monkeypatch,"select")
    assert metadata["method"] == "grid_curvature_v2"
    assert metadata["grid_curvature"]["decision"] == "DEWARP"
    assert metadata["grid_curvature_candidate"]["after"]["decision"] != "DEWARP"
    assessment = metadata["grid_curvature_assessment"]
    assert assessment["selected"] is True and assessment["status"] == "selected"
    assert metadata["wave_assessment"]["selected"] is False
    assert np.array_equal(output,cv2.remap(image,mx,my,cv2.INTER_CUBIC,
                                           borderMode=cv2.BORDER_CONSTANT,borderValue=(255,255,255)))
    layout = table_layout(output,metadata)
    assert layout["grid_confirmed"] and len(layout["row_boundaries"]) >= 4
    assert layout["body_left"] < layout["name_left"] < layout["name_right"] < layout["quantity_right"] < layout["right"]


def test_shadow_mode_reports_v2_without_changing_established_output(monkeypatch):
    image,(off_output,off_x,off_y,off_metadata) = _rectify_curved(monkeypatch,"off")
    assert "grid_curvature" not in off_metadata
    assert "grid_curvature_assessment" not in off_metadata
    _, (shadow_output,shadow_x,shadow_y,shadow_metadata) = _rectify_curved(monkeypatch,"shadow")
    assert shadow_metadata["method"] == off_metadata["method"]
    assert np.array_equal(shadow_output,off_output)
    assert np.array_equal(shadow_x,off_x) and np.array_equal(shadow_y,off_y)
    assessment = shadow_metadata["grid_curvature_assessment"]
    assert assessment["selected"] is False and assessment["status"] == "candidate_ready"


def test_wave_failure_returns_established_contour_result(monkeypatch):
    from App.ocr import receipt_rectification as module
    image = np.full((400,600,3),255,np.uint8)
    polygon = np.array([[20,20],[580,20],[580,370],[20,370]],np.float32)
    mx,my,expected = module.build_contour_map(polygon)
    def failed(*args,**kwargs):
        raise ValueError("Unreliable grid")
    monkeypatch.setattr(module,"assess_dewarp_necessity",lambda image: {
        "decision":"DEWARP","needs_dewarp":True,"reason":"test"})
    monkeypatch.setattr(module,"rectify",failed)
    output,actual_x,actual_y,metadata = module.rectify_table(image,polygon)
    assert metadata["method"] == expected["method"]
    assert np.array_equal(actual_x,mx) and np.array_equal(actual_y,my)
    assert metadata["wave_assessment"]["status"] == "rejected"


def test_wave_and_contour_failure_keep_original_crop(monkeypatch):
    from App.ocr import receipt_rectification as module
    image = np.full((400,600,3),255,np.uint8)
    polygon = np.array([[20,20],[580,20],[580,370],[20,370]],np.float32)
    def failed(*args,**kwargs):
        raise ValueError("Unreliable geometry")
    monkeypatch.setattr(module,"assess_dewarp_necessity",lambda image: {
        "decision":"DEWARP","needs_dewarp":True,"reason":"test"})
    monkeypatch.setattr(module,"rectify",failed)
    monkeypatch.setattr(module,"build_contour_map",failed)
    output,mx,my,metadata = module.rectify_table(image,polygon)
    assert metadata["method"] == "original_crop_fallback"
    assert np.array_equal(output,image[4:387,4:597])
    assert mx[0,0]==4 and my[0,0]==4


def test_dewarp_gate_keeps_contour_result_when_curvature_is_not_proven(monkeypatch):
    from App.ocr import receipt_rectification as module
    image = np.full((400,600,3),255,np.uint8)
    polygon = np.array([[20,20],[580,20],[580,370],[20,370]],np.float32)
    monkeypatch.setattr(module,"assess_dewarp_necessity",lambda candidate: {
        "decision":"SKIP","needs_dewarp":False,"reason":"already straight"})
    monkeypatch.setattr(module,"rectify",lambda *args,**kwargs: pytest.fail("wave dewarp must be skipped"))
    output,mx,my,metadata = module.rectify_table(image,polygon)
    assert output.shape == mx.shape + (3,)
    assert my.shape == mx.shape
    assert metadata["dewarp_necessity"]["decision"] == "SKIP"
    assert metadata["wave_assessment"] == {
        "status":"skipped_not_needed","selected":False,"reason":"already straight"}


def test_verified_grid_does_not_split_wrapped_names_by_multiple_price_blocks():
    layout = dict(body_left=10,name_left=30,name_right=220,quantity_right=260,right=330,
                  body_top=10,body_bottom=240,row_boundaries=[10,60,160,240],grid_confirmed=True)
    payload = []
    for text,x,y in (("Тауардың атауы",100,30),("Масло моторное",100,90),("ники Huter, 1л. @",100,145),
                     ("5780",290,100),("1| 242000",290,120),("Электрогенератор",100,180),("5,5кВт @",100,220)):
        box = [[x-10,y-5],[x+10,y-5],[x+10,y+5],[x-10,y+5]]
        payload.append(dict(text=text,confidence=.9,box=box,box_original=box))
    lines = aligned_lines(dict(lines=payload),{})
    boundaries,source = choose_row_boundaries(lines,layout)
    assert source == "verified_printed_grid" and boundaries==layout["row_boundaries"]
    rows = parse_rows(lines,layout,boundaries)
    assert len(rows)==2
    assert rows[0]["name_raw"].endswith("ники Huter, 1л. @")
    assert rows[1]["name_raw"].endswith("5,5кВт @")
