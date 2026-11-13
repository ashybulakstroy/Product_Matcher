"""Offline synthetic + one actual photographed receipt table, no OCR/API."""
import cv2
import numpy as np
from pathlib import Path
from App.ocr.grid_curvature_v2 import diagnose, horizontal_candidate, two_axis_candidate

ROOT=Path(__file__).resolve().parents[3]
FIXTURE=Path(__file__).parent/'fixtures'/'curved_receipt_table.png'
if not FIXTURE.is_file():
    FIXTURE=ROOT / 'data' / 'ocr_regression' / 'grid_curvature_v2' / 'real_receipt_table.png'
if not FIXTURE.is_file():
    FIXTURE=ROOT / 'fixtures' / 'real_receipt_table.png'


def synthetic(*, curved):
    h, w = 330, 600
    image = np.full((h,w,3), 243, np.uint8)
    xs=np.arange(w)
    warp=10*np.sin(np.pi*xs/(w-1)) if curved else np.zeros(w)
    for row in [22,85,152,223,284]:
        for x in range(15,w-15):
            y=int(round(row+warp[x]+.014*x))
            cv2.circle(image,(x,y),0,(80,80,80),-1)
    # Small footer line across price area; must not become a body rule.
    cv2.line(image,(385,306),(585,306),(80,80,80),1)
    return image


def test_real_photo_identifies_five_coherent_curved_rules():
    photo=cv2.imread(str(FIXTURE))
    assert photo is not None
    result=diagnose(photo)
    assert result['decision']=='DEWARP'
    assert result['horizontal_rules']==5
    assert result['vertical_rules']>=3
    assert result['coherent_rules']>=3
    assert all(row['coverage'] >= .7 for row in result['rules'])


def test_prototype_straightens_real_photo_without_crossing_lines():
    photo=cv2.imread(str(FIXTURE))
    output,mx,my,quality=horizontal_candidate(photo)
    assert output.shape==photo.shape and mx.shape==photo.shape[:2]
    assert quality['min_jacobian']>0.4
    assert quality['max_jacobian']<2.2
    assert quality['before']['decision']=='DEWARP'
    assert quality['after']['decision']=='SKIP'
    assert quality['after']['horizontal_rules']==5
    assert max(row['curve_p95_px'] for row in quality['after']['rules']) < 2
    assert not np.array_equal(photo,output)


def test_straight_and_affine_slanted_lines_are_not_waves():
    result=diagnose(synthetic(curved=False))
    assert result['decision']!='DEWARP',result


def test_synthetic_curved_rules_have_dewarp_evidence():
    result=diagnose(synthetic(curved=True))
    assert result['decision']=='DEWARP',result
    out,mx,my,stats=horizontal_candidate(synthetic(curved=True))
    assert stats['after']['decision']!='DEWARP'


def test_blank_has_no_positive_result():
    img=np.full((350,620,3),255,np.uint8)
    assert diagnose(img)['decision']=='UNCERTAIN'


def test_footer_is_not_treated_as_full_width_table_boundary():
    img=cv2.imread(str(FIXTURE))
    out,mx,my,stats=horizontal_candidate(img)
    assert stats['after']['horizontal_rules']==5
    assert stats['after']['vertical_rules']>=3


def test_two_axis_map_is_safe_and_preserves_five_grid_columns():
    photo=cv2.imread(str(FIXTURE))
    result,mx,my,quality=two_axis_candidate(photo)
    assert quality['vertical_status']=='candidate_ready'
    assert quality['quality_2d']['min_jacobian']>.35
    assert quality['after_2d']['vertical_rules']>=5
    assert quality['after_2d']['horizontal_rules']==5
    assert result.shape==photo.shape
    assert np.isfinite(mx).all() and np.isfinite(my).all()
