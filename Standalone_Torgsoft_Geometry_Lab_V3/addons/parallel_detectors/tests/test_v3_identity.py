import sys
from pathlib import Path
import hashlib
import argparse
import json
import numpy as np
import pytest

ADDON=Path(__file__).resolve().parents[1]
LAB=ADDON.parents[1]
sys.path.insert(0,str(ADDON))
sys.path.insert(0,str(LAB))
import detectors
import worker
import metrics


def test_actual_v3_identity_and_no_legacy_substitution():
    function,identity=detectors.resolve_v3(LAB)
    assert function.__module__=='geometry_lab.physical_grid_graph_v3'
    assert identity['function']=='detect_physical_graph'
    assert Path(identity['module_path'])==LAB/'geometry_lab/physical_grid_graph_v3.py'
    assert identity['sha256']==hashlib.sha256(Path(identity['module_path']).read_bytes()).hexdigest()
    assert identity['sha256']==detectors.V3_SHA256
    assert 'V3' in identity['version']


def test_legacy_only_lab_is_unavailable(tmp_path):
    (tmp_path/'geometry_lab').mkdir()
    (tmp_path/'geometry_lab/detection.py').write_text('def detect_grid(img): return {}, {}')
    with pytest.raises(detectors.AlgorithmUnavailable):
        detectors.physical_grid_current(np.full((200,200,3),255,np.uint8),tmp_path)


def test_changed_source_cannot_pass_as_frozen_v3(monkeypatch):
    monkeypatch.setattr(detectors,'V3_SHA256','0'*64)
    with pytest.raises(detectors.AlgorithmUnavailable):detectors.resolve_v3(LAB)


def test_native_adapter_matches_direct_v3_without_calling_legacy(monkeypatch):
    from geometry_lab import detection
    def forbidden(*args):raise AssertionError('Legacy detect_grid called under V3 name')
    monkeypatch.setattr(detection,'detect_grid',forbidden)
    function,_=detectors.resolve_v3(LAB)
    image=np.full((400,300,3),255,np.uint8)
    grid,_=function(image)
    lines,info=detectors.physical_grid_current(image,LAB)
    assert grid is None and lines==[]
    assert info['backend']=='geometry_lab.physical_grid_graph_v3.detect_physical_graph'


def test_worker_reports_algorithm_unavailable(tmp_path):
    import cv2
    path=tmp_path/'source.png';cv2.imwrite(str(path),np.full((200,200,3),255,np.uint8))
    args=argparse.Namespace(method='physical_grid_current',job_id='case',sha='test',image=str(path),
        output=str(tmp_path/'out'),lab=str(tmp_path/'missing'),job_dir=str(tmp_path),gt='',proc_max_side=0)
    result=worker.perform(args)
    assert result['status']=='ALGORITHM_UNAVAILABLE'
    assert result['processed_shape']==result['source_shape']


@pytest.mark.parametrize('flag',['human_verified','user_confirmed_scope'])
def test_existing_confirmation_schema_is_respected(flag):
    result=metrics.compare({flag:True,'horizontal_polylines':[],'vertical_polylines':[]},[],(100,100))
    assert result['status']=='GT_EVALUATED'
