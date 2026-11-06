import cv2
import numpy as np
import pytest
from dataclasses import replace

from App.ocr.wave_dewarp import Rule, find_rules, map_quality, ordered_axis_map, rectify


def grid():
    image = np.full((500,800,3),245,np.uint8)
    for y in (30,100,260,420):
        cv2.line(image,(30,y),(770,y),(30,30,30),2)
    for x in (30,90,480,610,770):
        cv2.line(image,(x,30),(x,455 if x>=480 else 420),(30,30,30),2)
    cv2.line(image,(480,455),(770,455),(30,30,30),2)
    cv2.putText(image,"Product ABC 40x20",(110,160),0,.7,(20,20,20),1)
    cv2.putText(image,"Second product",(110,320),0,.7,(20,20,20),1)
    cv2.putText(image,"247780.00",(500,445),0,.7,(0,0,190),2)
    return image


def test_wavy_grid_is_straightened_without_folding_or_losing_total():
    image = grid()
    yy,xx = np.indices(image.shape[:2],dtype=np.float32)
    warped = cv2.remap(image,xx-6*np.sin(np.pi*yy/500),yy-18*np.sin(2*np.pi*xx/800)*(.4+.6*yy/500),
                       cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
    output,mx,my,meta,_ = rectify(warped,[[0,0],[799,0],[799,499],[0,499]],margin=0)
    assert meta["status"] == "rectified",meta
    assert meta["quality"]["positive"]
    assert mx.dtype == my.dtype == np.float32
    before,_ = find_rules(warped)
    after,_ = find_rules(output)
    assert len(after)>=len(before)>=3
    assert max(r.curvature for r in after) < max(r.curvature for r in before)*.3
    red = lambda im:np.count_nonzero((im[:,:,2]>100)&(im[:,:,2]>im[:,:,0].astype(float)*1.8))
    assert red(output)>.8*red(warped)
    assert output.shape == image.shape


def test_flat_grid_is_left_unchanged():
    image = grid()
    output,_,_,meta,_ = rectify(image,[[0,0],[799,0],[799,499],[0,499]],margin=0)
    assert meta["status"] == "no_change",meta
    assert np.array_equal(output,image)


def test_insufficient_rules_return_exact_crop():
    image = np.full((200,300,3),255,np.uint8)
    output,_,_,meta,_ = rectify(image,[[0,0],[299,0],[299,199],[0,199]],margin=0)
    assert meta["status"] == "rejected"
    assert meta["warnings"]
    assert np.array_equal(output,image)


def test_crossing_constraints_are_rejected():
    curves = [np.linspace(10,100,100),np.full(100,50),np.full(100,120)]
    rules = [Rule(values,np.empty((0,2)),1,0,0) for values in curves]
    with pytest.raises(ValueError,match="пересекаются"):
        ordered_axis_map(rules,150,100)


def test_negative_jacobian_is_not_accepted():
    yy,xx = np.indices((100,150),dtype=np.float32)
    assert not map_quality(xx[:,::-1],yy)["positive"]


def test_failed_second_check_restores_original_pixels(monkeypatch):
    from App.ocr import wave_dewarp as module
    image = grid()
    yy,xx = np.indices(image.shape[:2],dtype=np.float32)
    warped = cv2.remap(image,xx,yy-15*np.sin(2*np.pi*xx/800),cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
    original = module.find_rules
    horizontal_calls = 0
    def measured(image,vertical=False):
        nonlocal horizontal_calls
        rules,mask = original(image,vertical)
        if not vertical:
            horizontal_calls += 1
            if horizontal_calls == 2:
                rules = [replace(rule,curvature=100) for rule in rules]
        return rules,mask
    monkeypatch.setattr(module,"find_rules",measured)
    output,mx,my,meta,diagnostics = module.rectify(warped,[[0,0],[799,0],[799,499],[0,499]],margin=0)
    assert meta["status"] == "rejected_quality"
    assert "rejected_candidate" in diagnostics
    assert np.array_equal(output,warped)
    assert np.array_equal(mx,xx) and np.array_equal(my,yy)
