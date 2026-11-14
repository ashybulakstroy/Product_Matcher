import json

import cv2
import numpy as np
import pytest

from App.ocr.cell_preview import preview_png


def fixture(root, *, curved=False):
    yy,xx=np.indices((100,120))
    original=np.stack([xx,yy,np.full_like(xx,77)],axis=-1).astype(np.uint8)
    cv2.imwrite(str(root/'original.png'),original)
    # A different working image must never be used for the original evidence.
    cv2.imwrite(str(root/'rectified.png'),np.zeros((40,50,3),np.uint8))
    my,mx=np.indices((40,50),dtype=np.float32)
    mx+=20
    my+=30
    if curved:my+=10*np.sin(np.pi*(mx-20)/49)
    np.savez_compressed(root/'rectification_map.npz',map_x=mx,map_y=my)
    (root/'layout.json').write_text(json.dumps(dict(cells=[dict(row=1,role='quantity',
        box=[[5,5],[25,5],[25,20],[5,20]])])),encoding='utf-8')
    return original,mx,my


def test_preview_uses_original_pixels_and_whole_physical_cell(tmp_path):
    original,_,_=fixture(tmp_path)
    actual=cv2.imdecode(np.frombuffer(preview_png(tmp_path,1,'quantity'),np.uint8),cv2.IMREAD_COLOR)
    assert np.array_equal(actual,original[32:54,22:49])


def test_curved_map_uses_interior_not_only_corners(tmp_path):
    original,mx,my=fixture(tmp_path,curved=True)
    actual=cv2.imdecode(np.frombuffer(preview_png(tmp_path,1,'quantity'),np.uint8),cv2.IMREAD_COLOR)
    bottom=int(np.ceil(my[5:21,5:26].max()))+4
    top=int(np.floor(my[5:21,5:26].min()))-3
    assert np.array_equal(actual,original[top:bottom,22:49])


@pytest.mark.parametrize('row,field',[(0,'quantity'),(1,'../../upload.bin'),(2,'price'),(True,'name')])
def test_unknown_cell_is_rejected(tmp_path,row,field):
    fixture(tmp_path)
    with pytest.raises(ValueError):preview_png(tmp_path,row,field)


def test_layout_change_invalidates_cached_preview(tmp_path):
    fixture(tmp_path)
    before=preview_png(tmp_path,1,'quantity')
    p=tmp_path/'layout.json'
    data=json.loads(p.read_text())
    data['cells'][0]['box']=[[10,5],[30,5],[30,20],[10,20]]
    p.write_text(json.dumps(data)+'\n',encoding='utf-8')
    assert preview_png(tmp_path,1,'quantity')!=before
